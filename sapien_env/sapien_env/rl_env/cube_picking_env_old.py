import numpy as np
import sapien.core as sapien
import transforms3d.euler

from sapien_env.sim_env.base import BaseSimulationEnv
from sapien_env.utils.yx_object_utils import load_yx_obj
from sapien_env.utils.common_robot_utils import load_robot
from constructor import add_default_scene_light


class CubePickingEnv(BaseSimulationEnv):
    def __init__(self, use_gui=True, frame_skip=5, object_scale=1.0, friction=0.3, seed=None, **kwargs):
        super().__init__(use_gui=use_gui, frame_skip=frame_skip, **kwargs)

        # Setup scene
        self.scene = self.engine.create_scene()
        self.scene.set_timestep(0.004)
        self.random_state = np.random.RandomState(seed)
        self.friction = friction
        self.object_scale = object_scale

        # Load table
        self.table = self.create_table(table_height=0.6, table_half_size=[0.35, 0.35, 0.025])

        # Load Panda robot
        self.robot = load_robot(
            scene=self.scene,
            renderer=self.renderer,
            robot_uid="panda",
            control_freq=240 // frame_skip,
            fix_root_link=True,
            reconfig=False
        )

        # Load cube object
        self.cube = load_yx_obj(self.scene, 'cube', density=1000, scale=self.object_scale)

        # Define workspace
        self.workspace_bounds = {
            'x': (-0.15, 0.15),
            'y': (-0.15, 0.15),
            'z': 0.6
        }

        # Set up cameras
        self._setup_cameras()

        # Add lighting
        add_default_scene_light(self.scene, self.renderer)

    def _setup_cameras(self):
        self.cameras = []
        cam_positions = [
            ([0.3, 0.0, 0.5], [0, 0, 0]),      # top-down
            ([0.2, 0.2, 0.4], [0, 0, 0.6]),    # diagonal
            ([0.0, 0.3, 0.3], [0, 0, 0.6]),    # side view
            ([-0.2, 0.2, 0.4], [0, 0, 0.6])    # angled
        ]

        for i, (pos, lookat) in enumerate(cam_positions):
            cam = self.scene.add_camera(
                name=f"camera_{i}",
                width=128,
                height=128,
                fovy=1.57,
                near=0.01,
                far=10
            )
            q = sapien.Pose().look_at(pos, lookat).q
            cam.set_pose(sapien.Pose(pos, q))
            self.cameras.append(cam)

    def reset(self):
        # Reset robot state
        self.robot.robot.set_qpos(np.zeros(self.robot.robot.dof))
        self.robot.robot.set_qvel(np.zeros(self.robot.robot.dof))

        # Reset cube pose
        self._randomize_cube_pose()

        # Take initial images
        self.render()

        return self.get_observation()

    def _randomize_cube_pose(self):
        x = self.random_state.uniform(*self.workspace_bounds['x'])
        y = self.random_state.uniform(*self.workspace_bounds['y'])
        z = self.workspace_bounds['z']
        yaw = self.random_state.uniform(0, 2 * np.pi)
        q = transforms3d.euler.euler2quat(0, 0, yaw)
        pose = sapien.Pose([x, y, z], q)
        self.cube.set_pose(pose)

    def get_observation(self):
        obs = {}

        # Robot state
        obs['qpos'] = self.robot.robot.get_qpos().copy()
        obs['qvel'] = self.robot.robot.get_qvel().copy()
        obs['ee_pose'] = self.robot.robot.get_links()[-1].get_pose().p.tolist()

        # Camera images
        for i, cam in enumerate(self.cameras):
            rgb = cam.get_color_rgba()[:, :, :3]  # [H, W, 3]
            depth = cam.get_depth()               # [H, W]
            obs[f'camera_{i}_rgb'] = rgb
            obs[f'camera_{i}_depth'] = depth

        return obs

    def step(self, action):
        self.robot.robot.set_qf(action)  # assuming torque control
        for _ in range(self.frame_skip):
            self.scene.step()
        self.render()
        obs = self.get_observation()
        return obs, 0.0, False, {}

    def render(self):
        for cam in self.cameras:
            cam.take_picture()
        if self.use_gui:
            self.renderer.render()

    def simple_step(self):
        self.scene.step()


def env_test():
    from sapien.utils import Viewer

    env = CubePickingEnv(use_gui=True)
    env.reset()
    viewer = Viewer(env.renderer)
    viewer.set_scene(env.scene)
    viewer.set_camera_xyz(0, 0.5, 0.5)
    viewer.set_camera_rpy(0, -0.5, np.pi / 2)
    viewer.set_fovy(2.0)
    env.viewer = viewer

    while not viewer.closed:
        env.simple_step()
        env.render()


if __name__ == '__main__':
    env_test()
