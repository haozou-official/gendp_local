from functools import cached_property
import os
import sys
import numpy as np
import sapien.core as sapien
import transforms3d
from sapien_env.rl_env.base import BaseRLEnv
from sapien_env.sim_env.cube_picking_env import CubePickingEnv
from sapien_env.utils.common_robot_utils import (
    generate_arm_robot_hand_info,
    generate_panda_info,
    generate_free_robot_hand_info,
    load_robot,
)
from sapien_env.gui.gui_base import GUIBase, YX_TABLE_TOP_CAMERAS

class CubePickingRLEnv(CubePickingEnv, BaseRLEnv):
    def __init__(self, use_gui=False, frame_skip=5, robot_name="panda", **renderer_kwargs):
        super().__init__(use_gui=use_gui, frame_skip=frame_skip, **renderer_kwargs)
        self.setup(robot_name)

        self.robot_name = robot_name
        self.is_robot_free = "free" in robot_name
        self.is_xarm = "xarm" in robot_name
        self.is_panda = "panda" in robot_name

        if self.is_robot_free:
            self.robot_info = generate_free_robot_hand_info()[robot_name]
        elif self.is_xarm:
            self.robot_info = generate_arm_robot_hand_info()[robot_name]
        elif self.is_panda:
            self.robot_info = generate_panda_info()[robot_name]
        else:
            raise NotImplementedError(f"Unsupported robot: {robot_name}")

        self.robot = load_robot(self.scene, robot_name)
        self.palm_link_name = self.robot_info.palm_name
        self.palm_link = [link for link in self.robot.get_links() if link.get_name() == self.palm_link_name][0]

        self.cam_names = list(YX_TABLE_TOP_CAMERAS.keys())
        self.gui = GUIBase(self.scene, self.renderer, headless=True)
        for name, params in YX_TABLE_TOP_CAMERAS.items():
            if 'rotation' in params:
                self.gui.create_camera_from_pos_rot(**params)
            else:
                self.gui.create_camera(**params)
        self.cameras = {cam.get_name(): cam for cam in self.gui.cams}
        self.object_episode_init_pose = self.cube.get_pose()

    def reset(self):
        qpos = self.robot_info.arm_init_qpos.copy()
        self.robot.set_qpos(qpos)
        self.robot.set_drive_target(qpos)
        self.robot.set_pose(sapien.Pose([0, -0.5, 0], transforms3d.euler.euler2quat(0, 0, np.pi / 2)))

        self.object_episode_init_pose = self.generate_random_init_pose(self.randomness_scale)
        self.cube.set_pose(self.object_episode_init_pose)

        self.current_step = 0
        return self.get_robot_state()

    def get_reward(self, action):
        cube_pos = self.cube.get_pose().p
        ee_pos = self.palm_link.get_pose().p
        dist = np.linalg.norm(cube_pos - ee_pos)
        return float(dist < 0.05)

    def is_done(self):
        return self.current_step >= 200

    def get_robot_state(self):
        qpos = self.robot.get_qpos()
        ee_pose = self.palm_link.get_pose()
        return np.concatenate([qpos, ee_pose.p])

    def get_oracle_state(self):
        robot_qpos_vec = self.robot.get_qpos()
        cube_pose = self.cube.get_pose()
        cube_pose_vec = np.concatenate([cube_pose.p, cube_pose.q])
        linear_v = self.cube.get_velocity()
        angular_v = self.cube.get_angular_velocity()
        palm_pose = self.palm_link.get_pose()
        cube_in_palm = cube_pose.p - palm_pose.p
        z_axis = cube_pose.to_transformation_matrix()[:3, 2]
        theta_cos = np.dot(z_axis, np.array([0, 0, 1]))
        oracle_state = np.concatenate([
            robot_qpos_vec,
            cube_pose_vec,
            linear_v,
            angular_v,
            cube_in_palm,
            np.array([theta_cos])
        ])
        return oracle_state

    def set_init(self, init_states):
        cube_pose = sapien.Pose.from_transformation_matrix(init_states[0])
        self.cube.set_pose(cube_pose)

    @cached_property
    def obs_dim(self):
        return len(self.get_robot_state())

    @cached_property
    def horizon(self):
        return 200

def main_env():
    import imageio
    import os
    from sapien_env.gui.gui_base import GUIBase, YX_TABLE_TOP_CAMERAS
    import cv2

    env = CubePickingRLEnv(
        use_gui=False,
        use_visual_obs=True,
        need_offscreen_render=True,
        frame_skip=5,
        robot_name="panda"
    )
    env.seed(42)
    env.reset()

    gui = GUIBase(env.scene, env.renderer, headless=True)
    for name, params in YX_TABLE_TOP_CAMERAS.items():
        if 'rotation' in params:
            gui.create_camera_from_pos_rot(position=params['position'], rotation=params['rotation'], name=name)
        else:
            gui.create_camera(position=params['position'], look_at_dir=params['look_at_dir'], right_dir=params['right_dir'], name=name)

    rgbs = gui.render()
    panel = np.concatenate([cv2.resize(rgb, (320, 240)) for rgb in rgbs], axis=1)

    save_dir = "./video_output"
    os.makedirs(save_dir, exist_ok=True)
    cv2.imwrite(os.path.join(save_dir, "frame0.png"), panel[:, :, ::-1])  # BGR for OpenCV
    print("✅ One frame rendered and saved.")

if __name__ == '__main__':
    main_env()
