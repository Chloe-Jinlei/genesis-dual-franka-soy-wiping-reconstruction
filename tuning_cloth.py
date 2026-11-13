"""Isolated cloth lift convergence probe, independent of robot/controller contacts.

The 36 grasp particles receive continuously integrated prescribed velocities;
every other particle is advanced by Genesis PBD. This intentionally diagnoses
cloth convergence separately from hand collisions, IK, and link attachments.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
import genesis as gs
from genesis_extensions import TableClothContact


def ease(t):
    t = np.clip(t, 0, 1)
    return t * t * t * (10 + t * (-15 + 6 * t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stretch-iterations', type=int, default=24)
    ap.add_argument('--substeps', type=int, default=20)
    ap.add_argument('--relaxation', type=float, default=.2)
    ap.add_argument('--air-resistance', type=float, default=8e-6)
    ap.add_argument('--bending-relaxation', type=float, default=.1)
    ap.add_argument('--bending-compliance', type=float, default=2e-5)
    ap.add_argument('--wet-mass', type=float, default=0.0,
                    help='Additional liquid mass in kg, uniformly distributed for a load probe.')
    ap.add_argument('--duration', type=float, default=4.0)
    ap.add_argument('--lift-time', type=float, default=2.5)
    ap.add_argument('--lower-start', type=float, default=None)
    ap.add_argument('--lower-duration', type=float, default=2.0)
    ap.add_argument('--lower-height', type=float, default=.76545,
                    help='Final anchor height in metres, when --lower-start is set.')
    args = ap.parse_args()
    root = Path(__file__).resolve().parent
    gs.init(backend=gs.gpu, precision='32', logging_level='warning', seed=42)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=.01, substeps=args.substeps),
        pbd_options=gs.options.PBDOptions(
            particle_size=.004, max_stretch_solver_iterations=args.stretch_iterations,
            max_bending_solver_iterations=2, lower_bound=(-.9,-.6,.7),upper_bound=(.9,.6,1.9)),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Box(size=(1.6,1,.05),pos=(0,0,.725),fixed=True),
                     material=gs.materials.Rigid(needs_coup=True))
    cloth = scene.add_entity(gs.morphs.Mesh(file=str(root/'assets/cotton_cloth.obj'),
        pos=(0,-.23,.75045),decimate=False),
        material=gs.materials.PBD.Cloth(rho=.22,stretch_compliance=1e-8,
            bending_compliance=args.bending_compliance,stretch_relaxation=args.relaxation,
            bending_relaxation=args.bending_relaxation,
            static_friction=.35,kinetic_friction=.28,air_resistance=args.air_resistance))
    scene.sim.pbd_solver._particle_size=.0008
    scene.sim.pbd_solver.dist_scale=.001
    scene.build()
    rest=cloth.get_particles_pos().cpu().numpy()
    patches=[np.where((rest[:,0]*side>.127)&(np.abs(rest[:,1]+.23)<.008))[0] for side in (-1,1)]
    for patch in patches:cloth.fix_particles(patch)
    contact=TableClothContact(cloth,mu_static=.35,mu_kinetic=.28,contact_tolerance=.0005)
    if args.wet_mass>0:
        contact.set_absorbed_masses(np.full(cloth.n_particles,args.wet_mass/cloth.n_particles))
    wall=time.time(); peak=1.; peak95=1.;records=[]
    for i in range(round(args.duration/.01)):
        t=i*.01
        du=(ease((t+.01)/args.lift_time)-ease(t/args.lift_time))/.01
        lower_rate=0.0
        if args.lower_start is not None:
            lower_rate=(ease((t+.01-args.lower_start)/args.lower_duration)-
                        ease((t-args.lower_start)/args.lower_duration))/.01
        dz=.1718*du-(.92225-args.lower_height)*lower_rate
        for side,patch in zip((-1,1),patches):
            cloth.set_particles_vel(np.tile([-side*.006*du,0,dz],(len(patch),1)),patch)
        scene.step();contact.apply_friction(.01)
        if i%10==0:
            d=contact.diagnostics();peak=max(peak,d['cloth_edge_stretch_max']);peak95=max(peak95,d['cloth_edge_stretch_p95'])
            record={'t':t+.01,'stretch':d['cloth_edge_stretch_max'],'p95':d['cloth_edge_stretch_p95'],
                    'minz':d['cloth_z_min_m'],'maxz':d['cloth_z_max_m'],'speed':d['cloth_speed_max_m_s'],
                    'contact_particles':d['cloth_contact_particles']}
            records.append(record)
            if i%50==0: print('LIFT',json.dumps(record),flush=True)
            if not d['cloth_state_finite']:raise RuntimeError('Nonfinite state')
    result={'parameters':vars(args),'particles':cloth.n_particles,'patch_sizes':list(map(len,patches)),
            'peak_stretch':peak,'peak_stretch_p95':peak95,'wall_seconds':time.time()-wall,'records':records}
    print('TUNING_RESULT',json.dumps(result),flush=True)


if __name__=='__main__':main()
