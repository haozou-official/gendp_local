# Defines table, cube placement, cube randomization, and lighting
import numpy as np
import sapien.core as sapien
import transforms3d.euler

from sapien_env.sim_env.base import BaseSimulationEnv
from sapien_env.sim_env.constructor import add_default_scene_light
from sapien_env.gui.gui_base import GUIBase, YX_TABLE_TOP_CAMERAS

class CubePickingEnv(BaseSimulationEnv):
    def __init__(self, use_gui=True, frame_skip=5, randomness_scale=1, friction=0.3, seed=None,
                 use_ray_tracing=True, **renderer_kwargs):
        super().__init__(use_gui=use_gui, frame_skip=frame_skip, use_ray_tracing=use_ray_tracing, **renderer_kwargs)

        self.randomness_scale = randomness_scale
        scene_config = sapien.SceneConfig()
        self.scene = self.engine.create_scene(config=scene_config)
        self.scene.set_timestep(0.004)
        self.friction = friction

        # # # Load table
        # # #table_surface_z = 0.6
        # # self.table = self.create_table(table_height=0.025, table_half_size=[0.35, 0.7, 0.025])  # thickness is 5cm total
        
        # # Aligned with calibration frame
        # table_surface_z = -0.6
        # table_thickness = 0.05  
        # table_center_z = table_surface_z - table_thickness / 2
        # self.table = self.create_table(
        #     table_height=table_center_z,
        #     table_half_size=[0.35, 0.7, table_thickness / 2]
        # )

        # table_surface_z = -0.01
        # table_thickness = 0.05
        # table_center_z = table_surface_z - table_thickness / 2

        # self.table = self.create_table(
        #     table_height=table_center_z,
        #     table_half_size=[0.35, 0.7, table_thickness / 2]
        # )

        # # !!Flip table to face upward in calibration frame
        # flip_q = transforms3d.euler.euler2quat(np.pi, 0, 0)  # 180° rotation around X
        # self.table.set_pose(sapien.Pose([0, 0, table_center_z], flip_q))
        # #self.table.set_pose(sapien.Pose([0, 0, table_center_z]))
        
        # Load object
        #self.cube = self._create_cube()
        #self.cube.set_pose(sapien.Pose([0.1, 0, 0.025 + 0.025]))
        #cube_center_pos = np.array([0.0, 0.0, table_surface_z + 0.025 + 0.005])  
        #cube_center_pos = np.array([0.0, 0.0, table_surface_z - 0.025 - 0.005]) 
        #cube_center_pos = np.array([0.0, 0.0, -4])  
        #cube_center_pos = np.array([0.0, 0.0, -2])

        # Flip the cube 180° rotation around X
        #self.cube.set_pose(sapien.Pose(cube_center_pos, flip_q))
        #self.cube.set_pose(sapien.Pose(cube_center_pos))

        # cube_half_height = 0.02
        # cube_center_pos = np.array([0.0, 0.0, table_center_z - cube_half_height])  # ON TOP of table
        # cube_quat = transforms3d.euler.euler2quat(0, 0, 0) 
        # self.cube.set_pose(sapien.Pose(cube_center_pos, cube_quat))

        # self.original_object_pos = np.zeros(3)
        # print("Cube pose:", self.cube.get_pose())
        # print("Table center z:", table_center_z)

        # Load table
        self.table = self.create_table(table_height=0.4, table_half_size=[0.5, 1.0, 0.025])
        #self.table = self.create_table(table_height=0.5, table_half_size=[0.5, 1.0, 0.025])
        #self.table = self.create_table(table_height=0.5, table_half_size=[0.35, 0.7, 0.025])  # lower 1cm for cuba picking env
        # self.table = self.create_table(table_height=0.6, table_half_size=[0.35, 0.7, 0.025])
        
        # Load object
        self.cube = self._create_cube()
        self.cube.set_pose(sapien.Pose([0.1, 0, 0.42]))
        #self.cube.set_pose(sapien.Pose([0.1, 0, 0.52]))
        #self.cube.set_pose(sapien.Pose([0.1, 0, 0.62]))
        self.original_object_pos = np.zeros(3)

        # set up workspace boundary
        self.wkspc_half_w = 0.18
        self.wkspc_half_l = 0.18

    def _create_cube(self):
        builder = self.scene.create_actor_builder()
        builder.add_box_collision(half_size=[0.02, 0.02, 0.02])
        builder.add_box_visual(half_size=[0.02, 0.02, 0.02], color=[1.0, 0.1, 0.1, 1])
        cube = builder.build(name="cube")
        return cube

    def reset_env(self):
        pose = self.generate_random_init_pose(self.randomness_scale)
        self.cube.set_pose(pose)
        self.original_object_pos = pose.p

    def generate_random_init_pose(self, randomness_scale):
        pos = np.array([
            self.np_random.uniform(0.0, 0.15),
            self.np_random.uniform(-0.15, 0.15),
            0.42
            #0.52
        ])
        quat = transforms3d.euler.euler2quat(0, 0, 0)  # flip to match calibration frame
        return sapien.Pose(pos, quat)

    # def generate_random_init_pose(self, randomness_scale):
    #     # Random XY position within workspace bounds
    #     pos = np.array([
    #         self.np_random.uniform(0.0, 0.15),
    #         self.np_random.uniform(-0.15, 0.15),
    #         #-0.01 + 0.025 + 0.005
    #         -0.01 - 0.02  # cube_half_height = 0.02
    #         #-2
    #     ])
    #     # No rotation needed for cube (identity quaternion)
    #     quat = transforms3d.euler.euler2quat(0, 0, 0)  # It converts Euler angles (roll=0, pitch=0, yaw=0) → quaternion representation of no rotation
    #     return sapien.Pose(pos, quat)

    def get_init_poses(self):
        return np.stack([self.cube.get_pose().to_transformation_matrix()])


def env_test():
    from sapien.utils import Viewer
    env = CubePickingEnv(use_ray_tracing=False)
    env.reset_env()
    viewer = Viewer(env.renderer)
    viewer.set_scene(env.scene)
    viewer.set_camera_xyz(x=0, y=0.5, z=0.5)
    viewer.set_camera_rpy(r=0, p=-0.5, y=np.pi / 2)
    viewer.set_fovy(2.0)
    add_default_scene_light(env.scene, env.renderer)
    env.viewer = viewer

    while not viewer.closed:
        env.simple_step()
        env.render()


def get_init_pic():
    import cv2, os
    output_dir = f"/home/hz2999/gendp/data/cube"
    os.makedirs(output_dir, exist_ok=True)
    env = CubePickingEnv(use_ray_tracing=False, seed=0)
    env.reset_env()
    add_default_scene_light(env.scene, env.renderer)
    gui = GUIBase(env.scene, env.renderer, headless=True)

    for name, params in YX_TABLE_TOP_CAMERAS.items():
        if 'rotation' in params:
            gui.create_camera_from_pos_rot(**params)
        else:
            gui.create_camera(**params)

    for _ in range(50):
        env.simple_step()
        rgbs = gui.render()

    for i, rgb in enumerate(rgbs):
        cv2.imwrite(f'{output_dir}/{i}.png', rgb)


if __name__ == '__main__':
    #env_test()
    get_init_pic()