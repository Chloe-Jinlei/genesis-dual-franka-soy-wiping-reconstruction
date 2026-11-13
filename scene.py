"""Independent scene reconstruction from the four supplied initial images only."""
from pathlib import Path
import numpy as np
import genesis as gs

ROOT = Path(__file__).resolve().parent
TABLE_Z = .75
DT = .01
PARTICLE_SIZE = .004
CONTACT_THICKNESS = .0008

def build_scene(res=(1280, 960), viewer=False, cloth_enabled=True):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT, substeps=40),
        rigid_options=gs.options.RigidOptions(enable_self_collision=True, enable_joint_limit=True),
        pbd_options=gs.options.PBDOptions(particle_size=PARTICLE_SIZE,
            max_stretch_solver_iterations=24, max_bending_solver_iterations=2,
            lower_bound=(-.9,-.6,.70), upper_bound=(.9,.6,1.9)),
        viewer_options=gs.options.ViewerOptions(camera_pos=(1.9,-2.5,2.2),
            camera_lookat=(0,0,.9), camera_fov=45, refresh_rate=60),
        vis_options=gs.options.VisOptions(background_color=(.12,.15,.19),
            ambient_light=(.35,.35,.35), show_world_frame=False),
        show_viewer=viewer,
    )
    scene.add_entity(gs.morphs.Plane(), material=gs.materials.Rigid(needs_coup=False),
                     surface=gs.surfaces.Rough(color=(.29,.32,.35,1)))
    tabletop = scene.add_entity(gs.morphs.Box(size=(1.6,1.0,.05),pos=(0,0,.725),fixed=True),
                     material=gs.materials.Rigid(friction=.5,coup_friction=.35,coup_softness=.0002),
                     surface=gs.surfaces.Rough(color=(.76,.56,.34,1)))
    for x in (-.7,.7):
        for y in (-.4,.4):
            scene.add_entity(gs.morphs.Box(size=(.06,.06,.7),pos=(x,y,.35),fixed=True),
                material=gs.materials.Rigid(needs_coup=False),surface=gs.surfaces.Rough(color=(.13,.17,.19,1)))
        scene.add_entity(gs.morphs.Box(size=(.045,.84,.045),pos=(x,0,.14),fixed=True),
                material=gs.materials.Rigid(needs_coup=False),surface=gs.surfaces.Rough(color=(.13,.17,.19,1)))
    scene.add_entity(gs.morphs.Box(size=(1.4,.045,.045),pos=(0,.1,.14),fixed=True),
                material=gs.materials.Rigid(needs_coup=False),surface=gs.surfaces.Rough(color=(.13,.17,.19,1)))
    robots=[]
    for x,yaw in ((-.575,-25),(.575,205)):
        scene.add_entity(gs.morphs.Box(size=(.20,.20,.02),pos=(x,.22,.76),fixed=True),
                material=gs.materials.Rigid(needs_coup=False),surface=gs.surfaces.Rough(color=(.12,.16,.19,1)))
        robot=scene.add_entity(gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml',
                   pos=(x,.22,.77),euler=(0,0,yaw)),
                   material=gs.materials.Rigid(friction=.8,coup_friction=.7,
                        coup_softness=.0001,gravity_compensation=1.0))
        robots.append(robot)
    cloth=None
    if cloth_enabled:
        cloth=scene.add_entity(
            gs.morphs.Mesh(file=str(ROOT/'assets/cotton_cloth.obj'),pos=(0,-.23,.75045),
                           decimate=False),
            material=gs.materials.PBD.Cloth(rho=.22,stretch_compliance=1e-8,
                bending_compliance=1e4,stretch_relaxation=.2,bending_relaxation=.1,
                static_friction=.35,kinetic_friction=.28,air_resistance=8e-6),
            surface=gs.surfaces.Rough(diffuse_texture=gs.textures.ImageTexture(
                image_path=str(ROOT/'assets/cotton_weave.png')),double_sided=True))
        # Genesis 1.4.3 otherwise uses mesh sampling length as contact thickness.
        # Sampling is complete in the entity constructor; retain its 4mm mesh,
        # but compile collision kernels with the assumed 0.8mm cotton thickness.
        scene.sim.pbd_solver._particle_size=CONTACT_THICKNESS
        scene.sim.pbd_solver.dist_scale=scene.sim.pbd_solver.particle_radius/.4
    cameras={
        'overview': scene.add_camera(res=res,pos=(1.5,-2.05,1.85),lookat=(0,0,.95),fov=42,GUI=False),
        'front': scene.add_camera(res=res,pos=(0,-2.45,1.72),lookat=(0,0,1.10),fov=42,GUI=False),
        'top': scene.add_camera(res=res,pos=(0,0,3.5),lookat=(0,0,.75),up=(0,1,0),fov=35,GUI=False),
        'back': scene.add_camera(res=res,pos=(0,2.35,1.65),lookat=(0,0,1.10),fov=45,GUI=False),
        'closeup': scene.add_camera(res=res,pos=(.20,-.95,1.15),lookat=(0,-.025,.78),fov=44,GUI=False),
    }
    scene.build()
    for robot in robots:
        robot.set_dofs_kp(np.array([1800,1800,1600,1600,900,900,600,350,350]))
        robot.set_dofs_kv(np.array([100,100,90,90,50,50,35,15,15]))
        force=np.array([87,87,87,87,12,12,12,20,20])
        robot.set_dofs_force_range(-force,force)
    return scene,robots,cloth,cameras

def solve_pose(robot, target, seed=None, quat=None, samples=1):
    if quat is None:
        quat=robot._reconstruction_quat
    return robot.inverse_kinematics(robot.get_link('hand'),pos=np.array(target),
        quat=np.array(quat),local_point=np.array([0,0,.105]),init_qpos=seed,
        max_samples=samples,max_solver_iters=60,damping=.02,
        pos_tol=.0002,rot_tol=.002,return_error=True)

def initial_configuration(robots):
    qs=[]
    for side,robot in zip((-1,1),robots):
        robot._reconstruction_quat=np.array([0,2**-.5,-side*2**-.5,0])
        initial=np.array([0,-.4,0,-1.75,0,1.35,side*.785,.025,.025])
        robot.set_qpos(initial)
        q,err=solve_pose(robot,(side*.34,-.04,1.37),seed=initial,samples=10)
        q[-2:]=.025
        robot.set_qpos(q)
        robot.control_dofs_position(q)
        qs.append(q)
        print('INITIAL_IK',side,q.cpu().numpy().tolist(),err.cpu().numpy().tolist(),flush=True)
    return qs

if __name__=='__main__':
    import os, json
    os.environ.setdefault('PYOPENGL_PLATFORM','egl')
    from PIL import Image
    gs.init(backend=gs.gpu,precision='32',logging_level='warning',seed=42)
    scene,robots,cloth,cameras=build_scene(res=(1000,750))
    initial_configuration(robots)
    for i in range(20): scene.step()
    for name,cam in cameras.items():
        Image.fromarray(cam.render()[0]).save(ROOT/'output'/f'initial_{name}.png')
    print('cloth_particles',cloth.n_particles,'bounds',cloth.get_particles_pos().amin(0),cloth.get_particles_pos().amax(0))
    for side,robot in zip((-1,1),robots):
        q=robot.get_qpos()
        for target in [(side*.132,-.23,.87),(side*.132,-.23,.758),(side*.128,.04,.77),(side*.128,.10,.77)]:
            q,err=solve_pose(robot,target,seed=q,samples=10)
            print('REACH',side,target,'error',err.cpu().numpy().tolist(),'q',q.cpu().numpy().tolist(),flush=True)
