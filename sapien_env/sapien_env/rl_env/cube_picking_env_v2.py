# RL-style wrapper 
# Robot setup
# Reward computation
# Vis obs access
# Oracle state access
# main_env() test entry point for rollout
from functools import cached_property
from typing import Optional

import sys
sys.path.append('/home/hz2999/gendp/sapien_env')

import numpy as np
import sapien.core as sapien
import transforms3d
import os
import imageio
from sapien_env.rl_env.base import BaseRLEnv
from sapien_env.sim_env.cube_picking_env import CubePickingEnv
from sapien_env.rl_env.para import ARM_INIT
from sapien_env.utils.common_robot_utils import (
    generate_arm_robot_hand_info,
    generate_panda_info,
    generate_free_robot_hand_info,
)
from sapien_env.utils.common_robot_utils import load_robot
from sapien_env.gui.gui_base import GUIBase, YX_TABLE_TOP_CAMERAS

class CubePickingRLEnv(CubePickingEnv, BaseRLEnv):
    def __init__(self, use_gui=False, frame_skip=5, robot_name="panda", **renderer_kwargs):
        super().__init__(use_gui=use_gui, frame_skip=frame_skip, **renderer_kwargs)
        self.setup(robot_name)

        self.object_episode_init_pose = self.cube.get_pose()

        self.cam_names = list(YX_TABLE_TOP_CAMERAS.keys())
        self.cameras = {}
        self.gui = GUIBase(self.scene, self.renderer, headless=True)
        for name, params in YX_TABLE_TOP_CAMERAS.items():
            if 'rotation' in params:
                self.gui.create_camera_from_pos_rot(**params)
            else:
                self.gui.create_camera(**params)
        self.cameras = {cam.get_name(): cam for cam in self.gui.cams}

        self.robot_name = robot_name
        self.is_robot_free = "free" in robot_name
        self.is_xarm = "xarm" in robot_name
        self.is_panda = "panda" in robot_name

        # Choose the correct info source based on robot type
        if self.is_robot_free:
            self.robot_info = generate_free_robot_hand_info()[robot_name]
        elif self.is_xarm:
            # Returns a dictionary of robot info structs (ArmRobotInfo)
            self.robot_info = generate_arm_robot_hand_info()[robot_name]
        elif self.is_panda:
            self.robot_info = generate_panda_info()[robot_name]
        else:
            raise NotImplementedError(f"Unsupported robot: {robot_name}")

        # Actually build the robot
        self.robot = load_robot(self.scene, robot_name)
        self.palm_link_name = self.robot_info.palm_name
        self.palm_link = [link for link in self.robot.get_links() if link.get_name() == self.palm_link_name][0]

    def reset(self):
        qpos = self.robot_info.arm_init_qpos.copy()
        self.robot.set_qpos(qpos)
        self.robot.set_drive_target(qpos)
        self.robot.set_pose(sapien.Pose([0, -0.5, 0], transforms3d.euler.euler2quat(0, 0, np.pi / 2)))

        self.object_episode_init_pose = self.generate_random_init_pose(self.randomness_scale)
        self.cube.set_pose(self.object_episode_init_pose)

        self.current_step = 0
        return self.get_observation()

    def step(self, action):
        qpos = self.robot.get_qpos()
        qpos[:len(action)] += action
        self.robot.set_qpos(qpos)
        self.robot.set_drive_target(qpos)

        for _ in range(self.frame_skip):
            self.scene.step()

        self.current_step += 1
        obs = self.get_observation()
        reward = self.get_reward(action)
        done = self.is_done()
        info = {}
        return obs, reward, done, info

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
        # 1. Robot joint positions
        robot_qpos_vec = self.robot.get_qpos()

        # 2. Cube pose (position + quaternion)
        cube_pose = self.cube.get_pose()
        cube_pose_vec = np.concatenate([cube_pose.p, cube_pose.q])

        # 3. Cube velocities
        linear_v = self.cube.get_velocity()
        angular_v = self.cube.get_angular_velocity()

        # 4. Distance between cube and robot palm
        palm_pose = self.palm_link.get_pose()
        cube_in_palm = cube_pose.p - palm_pose.p

        # 5. Cube uprightness: how well aligned is cube's z-axis with world z-axis
        z_axis = cube_pose.to_transformation_matrix()[:3, 2]  # Local z in world
        theta_cos = np.dot(z_axis, np.array([0, 0, 1]))

        # Combine all into oracle state
        oracle_state = np.concatenate([
            robot_qpos_vec,         # Robot joint config
            cube_pose_vec,          # Cube position + orientation
            linear_v,               # Cube linear velocity
            angular_v,              # Cube angular velocity
            cube_in_palm,           # Relative position to palm
            np.array([theta_cos])   # Uprightness indicator
        ])
        return oracle_state

    def set_init(self, init_states):
        cube_pose = sapien.Pose.from_transformation_matrix(init_states[0])
        self.cube.set_pose(cube_pose)

    @cached_property
    def obs_dim(self):
        if not self.use_visual_obs:
            return self.robot.dof + 7 + 6 + 3 + 1
        else:
            return len(self.get_robot_state())

    @cached_property
    def horizon(self):
        return 200


def main_env():
    env = CubePickingRLEnv(
        use_gui=False,
        use_visual_obs=True,
        need_offscreen_render=True,
        frame_skip=5,
        robot_name="panda"
    )
    env.seed(42)
    env.reset()

    save_dir = "./video_output"
    os.makedirs(save_dir, exist_ok=True)
    video_path = os.path.join(save_dir, "rollout_cube.mp4")
    writer = imageio.get_writer(video_path, fps=20)

    for _ in range(100):
        action = np.zeros(env.arm_dof + 1)
        action[2] = 0.01
        obs, reward, done, _ = env.step(action)
        rgbs = env.gui.render(horizontal=False)
        writer.append_data(rgbs)

    writer.close()
    print(f"Saved video to {video_path}")


if __name__ == '__main__':
    main_env()