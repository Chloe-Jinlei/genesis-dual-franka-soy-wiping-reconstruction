"""Run reconstructed rigid/PBD/absorbing-film simulation on CUDA, save evidence."""
import os
os.environ.setdefault('PYOPENGL_PLATFORM','egl')
import argparse, json, time, csv
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
import trimesh
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont
from scipy.interpolate import CubicSpline
from scipy.spatial import cKDTree
import genesis as gs
from genesis.utils.geom import transform_by_trans_quat
from scene import build_scene, initial_configuration, solve_pose, ROOT, DT, TABLE_Z, CONTACT_THICKNESS
import motion
from fluid_model import ThinFilmAbsorption
from genesis_extensions import TableClothContact, FilmMeshVisual, prepare_cloth_colors, update_node_colors
from validation import SimulationValidator, Thresholds
from scipy.spatial.transform import Rotation

LABELS={'initial':'Initial state', 'approach':'Approach cloth edges',
 'lower_to_edges':'Lower to cloth', 'close_grippers':'Close both grippers',
 'lift_cloth':'Lift deformable cloth', 'transport':'Carry cloth above spill',
 'lay_on_spill':'Lay cloth onto soy sauce', 'begin_wipe':'Establish contact',
 'wipe':'Reciprocating wipe', 'final_lift':'Lift wet cloth', 'hold_wet_cloth':'Hold wet cloth'}

@lru_cache(maxsize=8)
def annotation_font(size):
    """Prefer the original font, with portable Pillow fallbacks."""
    for font in ('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 'DejaVuSans.ttf'):
        try:
            return ImageFont.truetype(font, size)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Older Pillow versions do not accept a size.
        return ImageFont.load_default()

def numpy(t):
    return t.detach().cpu().numpy() if isinstance(t,torch.Tensor) else np.asarray(t)

def plan(robots, duration, start_time=0., initial_commands=None):
    times=np.arange(start_time,duration+.05001,.05)
    qs=[r.get_qpos() for r in robots] if initial_commands is None else [torch.as_tensor(q,device=gs.device,dtype=torch.float32) for q in initial_commands]
    values=[]; errors=[]
    for t in times:
        poses,opening,_=motion.positions(min(t,motion.END_TIME))
        row=[]
        for k,r in enumerate(robots):
            q,err=solve_pose(r,poses[k],seed=qs[k],samples=1)
            e=numpy(err)
            if np.linalg.norm(e[:3])>.002 or np.linalg.norm(e[3:])>.02:
                raise RuntimeError(f'IK failed t={t:.3f} arm={k}: {e}')
            if np.max(np.abs(numpy(q-qs[k])[:7]))>.12:
                raise RuntimeError(f'IK branch jump at {t:.3f}, arm {k}')
            q[-2:]=opening; qs[k]=q;row.append(numpy(q));errors.append(e)
        if t==start_time and initial_commands is not None:
            row=np.array(initial_commands).copy()
        values.append(row)
    values=np.array(values)
    spline=CubicSpline(times,values,axis=0,bc_type='clamped')
    test=np.arange(start_time,duration+1e-5,DT)
    print('PLAN',json.dumps({'max_ik_position_error_m':float(np.max(np.linalg.norm(np.array(errors)[:,:3],axis=1))),
        'max_joint_speed_rad_s':float(np.abs(spline(test,1)[:,:,:7]).max()),
        'max_joint_acc_rad_s2':float(np.abs(spline(test,2)[:,:,:7]).max())}),flush=True)
    return spline, np.array(errors)

def film_mesh_state(film):
    v=numpy(film.film_vertices()).astype(np.float32)
    h=numpy(film.height).reshape(-1)
    colors=np.tile(np.array([.15,.065,.025,1],np.float32),(len(v),1))
    colors[:,3]=1-np.exp(-h/0.000025)
    # Rendering only: avoid z fighting of the dry grid against the table.
    v[:,2]=np.where(h>1e-9, v[:,2]+.00006,TABLE_Z-.001)
    return v,colors

def annotate(rgb,t,phase,diag):
    im=Image.fromarray(rgb)
    d=ImageDraw.Draw(im)
    w,h=im.size
    font=annotation_font(max(14,w//65))
    d.rectangle((0,h-65,w,h),fill=(19,28,37))
    d.text((18,h-57),f'{t:05.2f} s   {LABELS[phase]}',font=font,fill='white')
    remain=diag['free_liquid_mass_kg']*1000
    uptake=diag['absorbed_liquid_mass_kg']*1000
    d.text((18,h-30),f'Soy on table: {remain:.4f} g   |   In cloth: {uptake:.4f} g   |   CUDA physics',font=font,fill=(191,210,221))
    return np.array(im)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--duration',type=float,default=motion.END_TIME)
    ap.add_argument('--out',default='output/final')
    ap.add_argument('--width',type=int,default=1280)
    ap.add_argument('--fps',type=int,default=25)
    ap.add_argument('--no-video',action='store_true')
    args=ap.parse_args()
    out=ROOT/args.out;out.mkdir(parents=True,exist_ok=True)
    if not torch.cuda.is_available():raise RuntimeError('GPU required; CPU fallback is disabled')
    gs.init(backend=gs.gpu,precision='32',logging_level='warning',seed=42)
    scene,robots,cloth,cameras=build_scene(res=(args.width,args.width*3//4))
    initial_configuration(robots)
    rest=numpy(cloth.get_particles_pos()).copy()
    faces=np.asarray(cloth.mesh.faces)
    film=ThinFilmAbsorption(ROOT/'assets/soy_puddle.obj',rest,faces,device='cuda',contact_radius=CONTACT_THICKNESS/2)
    contact=TableClothContact(cloth,mu_static=.35,mu_kinetic=.28,contact_tolerance=.0005)
    validator=SimulationValidator(rest,faces,particle_radius=CONTACT_THICKNESS/2,
        thresholds=Thresholds(residual_fraction_success=.001))
    cloth_node=prepare_cloth_colors(scene,cloth)
    vmap=cKDTree(rest).query(cloth._vverts)[1]
    fv,fc=film_mesh_state(film)
    visual=FilmMeshVisual(scene,fv,film.film_faces,fc)
    path,ik_errors=plan(robots,args.duration)
    metadata={'description':'Independent reconstruction from four initial images, supplied assets, and task description.',
       'genesis_version':gs.__version__,'torch_version':torch.__version__,
       'backend':'CUDA','device':torch.cuda.get_device_name(0),'dt_s':DT,'substeps':40,
       'table_size_m':[1.6,1.,.05],'table_top_z_m':.75,
       'cloth_dimensions_m':[.28,.18],'cloth_particles':cloth.n_particles,
       'cloth_mesh_spacing_m':.004,'cloth_contact_diameter_m':CONTACT_THICKNESS,'physical_cloth_thickness_assumption_m':.0008,
       'stretch_compliance':1e-8,'bending_compliance':1e4,
       'film':film.parameters,'grasp_model':'Ideal no-slip attachment to actual hand links after closure; one-way constraint.',
       'limitations':['Photographs do not identify unique geometry, camera calibration, material, or robot states.',
          'Thin-film and capacity-limited absorption are reduced models, not native Genesis fluid or calibrated cotton porosity.',
          'Ideal grasps do not return exact constraint tension; no frictional grasp/slip validation.',
          'Table cloth friction is a weight-supported split Coulomb projection.',
          'Fluid pressure/shear reaction on cloth is omitted; textile wicking is a conservative saturation diffusion approximation.'],
       'sources':['Four images supplied in this task','assets/provenance.json','Installed Genesis 1.4.3 library and standard Panda asset']}
    (out/'configuration.json').write_text(json.dumps(metadata,indent=2))
    # Grasp material patches are selected once, never reseated during movement.
    patches=[np.where((rest[:,0]*side>.127)&(np.abs(rest[:,1]+.23)<.008))[0] for side in (-1,1)]
    grasp=False; grasp_offsets=[]; records=[]; snapshots=[]; film_states=[]; q_states=[]
    writer=None;topwriter=None;closewriter=None
    if not args.no_video:
        writer=imageio.get_writer(str(out/'simulation_overview.mp4'),fps=args.fps,codec='libx264',quality=8,macro_block_size=2)
        topwriter=imageio.get_writer(str(out/'simulation_top.mp4'),fps=args.fps,codec='libx264',quality=8,macro_block_size=2)
        closewriter=imageio.get_writer(str(out/'simulation_closeup.mp4'),fps=args.fps,codec='libx264',quality=8,macro_block_size=2)
    last_phase=None;wall=time.time();lift_start=None
    sample_frames=max(1,round(1/(args.fps*DT)))
    snapshot_times=[0,8.5,11,14,16,20,25,motion.WIPE_END,motion.WIPE_END+2,motion.END_TIME]
    snapshot_steps={round(t/DT):t for t in snapshot_times}
    # Initialize renderer buffers before attempting streamed color updates.
    for name,cam in cameras.items():
        Image.fromarray(cam.render()[0]).save(out/f'initial_{name}.png')
    for i in range(round(args.duration/DT)+1):
        t=i*DT
        poses,opening,phase=motion.positions(t)
        commands=path(t)
        for r,q in zip(robots,commands):r.control_dofs_position(q)
        if not grasp and t>=motion.GRASP_TIME:
            actual=numpy(cloth.get_particles_pos())
            grasp_log=[]
            for k,(r,ids) in enumerate(zip(robots,patches)):
                hand=r.get_link('hand');p=numpy(hand.get_pos());q=numpy(hand.get_quat())
                R=Rotation.from_quat(np.roll(q,-1)).as_matrix()
                local=(actual[ids]-p)@R
                # A geometric guard rejects attachment far from fingertip pads.
                if np.max(np.abs(local[:,0]))>.028 or np.max(np.abs(local[:,1]))>.025 or np.min(local[:,2])<.08 or np.max(local[:,2])>.12:
                    raise RuntimeError(f'Grasp patch is outside fingertip region: {k} {local.min(0)} {local.max(0)}')
                cloth.fix_particles_to_link(hand.idx,particles_idx_local=ids)
                grasp_offsets.append(local)
                grasp_log.append({'arm':k,'particles':len(ids),'local_bounds_m':[local.min(0).tolist(),local.max(0).tolist()]})
            (out/'grasp_event.json').write_text(json.dumps({'time_s':t,'patches':grasp_log},indent=2))
            print('GRASP',grasp_log,flush=True);grasp=True
        scene.step()
        contact.apply_friction(DT)
        positions=cloth.get_particles_pos()
        film.step(DT,positions,cloth_velocities=cloth.get_particles_vel())
        if film.absorbed_mass.sum().item()>0:
            contact.set_absorbed_masses(film.absorbed_mass,momentum_delta=film.momentum_transfer)
        # Validate actual integrated state at 100Hz, independent of video cadence.
        q=np.stack([numpy(r.get_dofs_position()) for r in robots])
        dq=np.stack([numpy(r.get_dofs_velocity()) for r in robots])
        torque=np.stack([numpy(r.get_dofs_control_force()) for r in robots])
        ddq=np.stack([numpy(r.get_dofs_acc()) for r in robots])
        gp=[]
        if grasp:
            for r,offset in zip(robots,grasp_offsets):
                hand=r.get_link('hand');R=Rotation.from_quat(np.roll(numpy(hand.get_quat()),-1)).as_matrix()
                gp.append(offset.mean(0)@R.T+numpy(hand.get_pos()))
        validator.update(t,phase,q,dq,torque,numpy(positions),
            grasp_points=gp if grasp else None,grasp_indices=patches if grasp else None,
            grasp_active=grasp,film_diag=film.diagnostics(),ddq=ddq)
        if i%10==0 or i==round(args.duration/DT):
            fd=film.diagnostics(); cd=contact.diagnostics()
            q=np.stack([numpy(r.get_dofs_position()) for r in robots])
            dq=np.stack([numpy(r.get_dofs_velocity()) for r in robots])
            torque=np.stack([numpy(r.get_dofs_control_force()) for r in robots])
            ddq=np.stack([numpy(r.get_dofs_acc()) for r in robots])
            row={'time_s':t,'phase':phase,**fd,**cd,
                'joint_speed_max_rad_s':float(np.abs(dq[:,:7]).max()),
                'joint_acc_max_rad_s2':float(np.abs(ddq[:,:7]).max()),
                'torque_max_Nm':float(np.abs(torque[:,:7]).max()),
                'tracking_joint_error_max_rad':float(np.abs(q[:,:7]-commands[:,:7]).max())}
            row['time_s']=t
            records.append(row);snapshots.append(numpy(positions));film_states.append(numpy(film.height).astype(np.float32));q_states.append(q)
            if not cd['cloth_state_finite']:raise RuntimeError('Nonfinite cloth state')
            if cd['cloth_table_penetration_max_m']>.004:raise RuntimeError('Cloth penetrated tabletop')
            if cd['cloth_edge_stretch_max']>1.35:
                (out/'failure.json').write_text(json.dumps(row,indent=2))
                Image.fromarray(cameras['top'].render()[0]).save(out/'failure_top.png')
                raise RuntimeError(f'Excessive cloth stretching at {t}: {cd["cloth_edge_stretch_max"]}')
        if phase!=last_phase or i%100==0:
            fd=film.diagnostics();cd=contact.diagnostics()
            print('PROGRESS',json.dumps({'time':round(t,2),'phase':phase,'wall_seconds':round(time.time()-wall,1),
                'residual':fd['residual_fraction'],'cloth_z_min':cd['cloth_z_min_m'],
                'stretch':cd['cloth_edge_stretch_max'],'joint_error':records[-1]['tracking_joint_error_max_rad']}),flush=True)
            last_phase=phase
        if (writer and i%sample_frames==0) or i in snapshot_steps:
            fv,fc=film_mesh_state(film);visual.update(fv,fc)
            wet=numpy(film.wetness)[vmap]
            stain=(1-np.exp(-wet*4)).clip(0,.92)
            colors=np.ones((len(vmap),4),np.float32)
            colors[:,:3]=(1-stain[:,None])+stain[:,None]*np.array([.22,.10,.035])
            update_node_colors(scene.visualizer.context,cloth_node,colors)
            fd=film.diagnostics()
            overview=cameras['overview'].render()[0]
            top=cameras['top'].render()[0]
            closeup=cameras['closeup'].render()[0]
            if writer and i%sample_frames==0:
                writer.append_data(annotate(overview,t,phase,fd));topwriter.append_data(annotate(top,t,phase,fd))
                closewriter.append_data(annotate(closeup,t,phase,fd))
            if i in snapshot_steps:
                Image.fromarray(annotate(overview,t,phase,fd)).save(out/f'stage_{t:05.1f}_{phase}.png')
                Image.fromarray(annotate(top,t,phase,fd)).save(out/f'top_{t:05.1f}_{phase}.png')
                Image.fromarray(annotate(closeup,t,phase,fd)).save(out/f'closeup_{t:05.1f}_{phase}.png')
        # Periodic durable metrics survive a failed test without losing evidence.
        if i%100==0:(out/'metrics.json').write_text(json.dumps(records,indent=2))
        # Finish a smooth complete cycle, then lift only when measured mass is clean.
        cycle=(t-17.)/5.
        if lift_start is None and t>=27. and abs(cycle-round(cycle))<1e-6:
            fd=film.diagnostics()
            if fd['residual_fraction']<=.001:
                lift_start=t;motion.WIPE_END=t;motion.END_TIME=t+5
                path,_=plan(robots,t+5,start_time=t,initial_commands=commands)
                snapshot_steps.update({round(t/DT):t,round((t+2)/DT):t+2,round((t+5)/DT):t+5})
                print('CLEAN_TRIGGER',json.dumps({'time_s':t,'cycles':round(cycle),'residual_fraction':fd['residual_fraction']}),flush=True)
            elif t>=57.:
                raise RuntimeError('Maximum wipe cycles reached without meeting the measured clean threshold')
        if lift_start is not None and t>=motion.END_TIME-1e-8:
            break
    if writer:writer.close();topwriter.close();closewriter.close()
    np.savez_compressed(out/'state_samples.npz',time=np.array([r['time_s'] for r in records]),
        cloth=np.array(snapshots),cloth_faces=faces,cloth_rest=rest,film_height=np.array(film_states),
        film_xy=numpy(film.xy),q=np.array(q_states),ik_errors=ik_errors,
        absorbed_mass=numpy(film.absorbed_mass),grasp_left=patches[0],grasp_right=patches[1])
    (out/'metrics.json').write_text(json.dumps(records,indent=2))
    with (out/'metrics.csv').open('w',newline='') as f:
        fields=[k for k,v in records[0].items() if not isinstance(v,(list,dict))]
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(records)
    summary={'completed_simulation_time_s':t,'wall_seconds':time.time()-wall,
        'clean_trigger_time_s':lift_start,'wipe_cycles':None if lift_start is None else round((lift_start-17)/5),
        'final':records[-1], 'max_cloth_stretch':max(r['cloth_edge_stretch_max'] for r in records),
        'max_table_penetration_m':max(r['cloth_table_penetration_max_m'] for r in records),
        'max_joint_speed_rad_s':max(r['joint_speed_max_rad_s'] for r in records),
        'max_tracking_joint_error_rad':max(r['tracking_joint_error_max_rad'] for r in records),
        'max_liquid_relative_mass_error':max(abs(r['relative_mass_error']) for r in records),
        'clean_threshold_fraction':.001,
        'clean':records[-1]['residual_fraction']<=.001,
        'final_lift_clearance_m':records[-1]['cloth_z_min_m']-TABLE_Z-CONTACT_THICKNESS/2,
        'limitations':metadata['limitations']}
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print('SUMMARY',json.dumps(summary),flush=True)
    validation=validator.finish(out/'validation.json',completed=lift_start is not None and t>=motion.END_TIME-1e-8)
    print('VALIDATION',json.dumps({'passed':validation['sampled_numerical_checks_passed'],
         'success':validation['task_success_under_stated_criteria'],'failures':validation['failures'],
         'cautions':validation['cautions']}),flush=True)

if __name__=='__main__':main()
