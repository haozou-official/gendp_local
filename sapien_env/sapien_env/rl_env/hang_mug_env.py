from functools import cached_property
from typing import Optional

import sys
sys.path.append('/home/bing4090/yixuan_old_branch/general_dp/sapien_env')

import numpy as np
import sapien.core as sapien
import transforms3d
from sapien.utils import Viewer

from sapien_env.rl_env.base import BaseRLEnv
from sapien_env.sim_env.hang_mug_env import HangMugEnv
from sapien_env.rl_env.para import ARM_INIT
from sapien_env.utils.common_robot_utils import generate_free_robot_hand_info, generate_arm_robot_hand_info, generate_panda_info


class HangMugRLEnv(HangMugEnv, BaseRLEnv):
    def __init__(self, use_gui=False, frame_skip=5, robot_name="xarm6", constant_object_state=False,
                 object_scale=1.0, randomness_scale=1, friction=1, object_pose_noise=0.01, manip_obj='nescafe_mug', **renderer_kwargs):
        super().__init__(use_gui, frame_skip, object_scale, randomness_scale, friction, use_ray_tracing=False, manip_obj=manip_obj, **renderer_kwargs)
        self.setup(robot_name)

        self.constant_object_state = constant_object_state
        self.object_pose_noise = object_pose_noise

        # Parse link name
        if self.is_robot_free:
            info = generate_free_robot_hand_info()[robot_name]
        elif self.is_xarm:
            info = generate_arm_robot_hand_info()[robot_name]
        elif self.is_panda:
            info = generate_panda_info()[robot_name]
        else:
            raise NotImplementedError
        self.palm_link_name = info.palm_name
        self.palm_link = [link for link in self.robot.get_links() if link.get_name() == self.palm_link_name][0]
        
        # Finger tip: thumb, index, middle, ring
        finger_tip_names = ["panda_leftfinger", "panda_rightfinger"]
        
        robot_link_names = [link.get_name() for link in self.robot.get_links()]
        self.finger_tip_links = [self.robot.get_links()[robot_link_names.index(name)] for name in finger_tip_names]

        # Object init pose
        self.object_episode_init_pose = sapien.Pose()

    def get_oracle_state(self):
        robot_qpos_vec = self.robot.get_qpos()
        object_pose = self.object_episode_init_pose if self.constant_object_state else self.manipulated_object.get_pose()
        object_pose_vec = np.concatenate([object_pose.p, object_pose.q])
        z_axis = object_pose.to_transformation_matrix()[:3, 2]
        theta_cos = np.sum(np.array([0, 0, 1]) * z_axis)
        palm_pose = self.palm_link.get_pose()
        object_in_palm = object_pose.p - palm_pose.p
        v = self.manipulated_object.get_velocity()
        w = self.manipulated_object.get_angular_velocity()
        return np.concatenate([robot_qpos_vec, object_pose_vec, v, w, object_in_palm, np.array([theta_cos])])

    def get_robot_state(self):
        robot_qpos_vec = self.robot.get_qpos()
        palm_pose = self.palm_link.get_pose()
        return np.concatenate([robot_qpos_vec, palm_pose.p])

    def get_reward(self, action):
        # 1.0 if success, 0.0 otherwise
        obj_pose = self.manipulated_object.get_pose()
        tree_pose = self.mug_tree.get_pose()
        
        h_thresh_upper = 0.3
        h_thresh_lower = 0.1
        
        obj_pos = obj_pose.p
        tree_pos = tree_pose.p
        obj_box_dist = np.linalg.norm(obj_pos[:2] - tree_pos[:2]) # ignore z axis
        obj_box_dist_thresh = 0.1
        
        reward = (obj_box_dist < obj_box_dist_thresh) and (obj_pos[-1] > h_thresh_lower) and (obj_pos[-1] < h_thresh_upper)
        return float(reward)

    def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None):
        # super().reset(seed=seed)
        if self.is_xarm:
            qpos = np.zeros(self.robot.dof)
            arm_qpos = self.robot_info.arm_init_qpos
            qpos[:self.arm_dof] = arm_qpos
            self.robot.set_qpos(qpos)
            self.robot.set_drive_target(qpos)
            init_pos = ARM_INIT + self.robot_info.root_offset
            init_pose = sapien.Pose(init_pos, transforms3d.euler.euler2quat(0, 0, 0))
        elif self.is_trossen_arm:
            print("trossen_arm")
            qpos = np.zeros(self.robot.dof)
            qpos[self.arm_dof:] =[0.021,-0.021]
            arm_qpos = self.robot_info.arm_init_qpos
            qpos[:self.arm_dof] = arm_qpos
            self.robot.set_qpos(qpos)
            self.robot.set_drive_target(qpos)
            init_pos = ARM_INIT + self.robot_info.root_offset
            init_pose = sapien.Pose(init_pos, transforms3d.euler.euler2quat(0, 0, 0))
        
        elif self.is_panda:
            qpos = self.robot_info.arm_init_qpos.copy()
            self.robot.set_qpos(qpos)
            self.robot.set_drive_target(qpos)
            init_pos = np.array([0.0, -0.5, 0.0])
            init_ori = transforms3d.euler.euler2quat(0, 0, np.pi / 2)
            init_pose = sapien.Pose(init_pos, init_ori)
        else:
            init_pose = sapien.Pose(np.array([-0.4, 0, 0.2]), transforms3d.euler.euler2quat(0, np.pi / 2, 0))
        self.robot.set_pose(init_pose)
        self.reset_internal()
        for i in range(100):
            self.robot.set_qf(self.robot.compute_passive_force(external=False, coriolis_and_centrifugal=False))
            self.scene.step()
        self.object_episode_init_pose = self.manipulated_object.get_pose()
        random_quat = transforms3d.euler.euler2quat(*(self.np_random.randn(3) * self.object_pose_noise * 10))
        random_pos = self.np_random.randn(3) * self.object_pose_noise
        self.object_episode_init_pose = self.object_episode_init_pose * sapien.Pose(random_pos, random_quat)

        return self.get_observation()

    def set_init(self, init_states):
        init_pose = sapien.Pose.from_transformation_matrix(init_states[0])
        self.manipulated_object.set_pose(init_pose)
        init_box_pose = sapien.Pose.from_transformation_matrix(init_states[1])
        self.mug_tree.set_pose(init_box_pose)

    @cached_property
    def obs_dim(self):
        if not self.use_visual_obs:
            return self.robot.dof + 7 + 6 + 3 + 1
        else:
            return len(self.get_robot_state())

    def is_done(self):
        return self.current_step >= self.horizon

    @cached_property
    def horizon(self):
        return 10000


# def main_env():
#     env = HangMugRLEnv(use_gui=True, robot_name="panda", frame_skip=10, use_visual_obs=False)
#     base_env = env
#     robot_dof = env.arm_dof + 1
#     env.seed(0)
#     env.reset()
#     viewer = Viewer(base_env.renderer)
#     viewer.set_scene(base_env.scene)
#     viewer.set_camera_xyz(x=0, y=0.5, z = 0.5)
#     viewer.set_camera_rpy(r=0, p=-0.5, y=np.pi/2)
#     viewer.set_fovy(2.0)
#     base_env.viewer = viewer

#     viewer.toggle_pause(False)
#     from tqdm import tqdm
#     for i in tqdm(range(200)):
#         action = np.zeros(robot_dof)
#         action[2] = 0.01
#         obs, reward, done, _ = env.step(action)
#         env.render()

#     viewer.toggle_pause(True)
#     while not viewer.closed:
#         env.simple_step()
#         env.render()

# def main_env():
#     import imageio
#     import os
#     import numpy as np
#     import transforms3d.euler
#     from sapien_env.gui.gui_base import YX_TABLE_TOP_CAMERAS
#     from sapien_env.rl_env.hang_mug_env import HangMugRLEnv
#     from sapien_env.sim_env.constructor import add_default_scene_light
#     from sapien.core import Pose
#     import sapien

#     env = HangMugRLEnv(
#         use_gui=False,
#         use_visual_obs=True,
#         need_offscreen_render=True,
#         frame_skip=5,
#         robot_name="panda"
#     )
#     env.seed(42)
#     env.reset()

#     # Add light
#     add_default_scene_light(env.scene, env.renderer)

#     # Setup camera
#     for cam_name, params in YX_TABLE_TOP_CAMERAS.items():
#         if 'rotation' in params:
#             position = np.array(params['position'])
#             euler_angles = np.array(params['rotation'])  # [roll, pitch, yaw]
#             #quat = transforms3d.euler.euler2quat(*euler_angles)  # x, y, z, w
#             quat = transforms3d.euler.euler2quat(euler_angles[0], euler_angles[1], euler_angles[2], axes='sxyz')
#             pose = Pose(position, quat)
#             env.create_camera_from_pose(
#                 pose=pose,
#                 name=cam_name,
#                 resolution=params['resolution'],
#                 fov=params['fov']
#             )
#         else:
#             env.create_camera(
#                 position=params['position'],
#                 look_at_dir=params['look_at_dir'],
#                 right_dir=params['right_dir'],
#                 name=cam_name,
#                 resolution=params['resolution'],
#                 fov=params['fov']
#             )

#     cam_name = list(env.cameras.keys())[0]
#     print(f"✅ Using camera: {cam_name}")

#     save_dir = "./video_output"
#     os.makedirs(save_dir, exist_ok=True)
#     video_path = os.path.join(save_dir, "rollout.mp4")
#     writer = imageio.get_writer(video_path, fps=20)

#     for i in range(100):
#         action = np.zeros(env.arm_dof + 1)
#         action[2] = 0.01  # simple upward motion
#         obs, reward, done, _ = env.step(action)
#         rgb = env.cameras[cam_name].get_rgb()
#         frame = (rgb * 255).astype(np.uint8)
#         writer.append_data(frame)

#     writer.close()
#     print(f"🎥 Saved video to: {video_path}")

def add_default_scene_light(scene: sapien.Scene, renderer: sapien.VulkanRenderer):
    direction = np.array([0, -1, -1], dtype=np.float32)
    color = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    scene.add_directional_light(
        direction=direction,
        color=color,
        shadow=True,
        position=np.array([0, 0, 2], dtype=np.float32),  # optional
        scale=2.0,
        near=-2.0,
        far=5.0,
        shadow_map_size=2048
    )

    scene.set_ambient_light([0.3, 0.3, 0.3])

def main_env():
    import imageio
    import os
    from sapien_env.gui.gui_base import GUIBase, YX_TABLE_TOP_CAMERAS
    #from sapien_env.utils.render_scene_utils import add_default_scene_light
    import sapien.core as sapien
    import cv2  # For resizing

    # Setup environment
    env = HangMugRLEnv(
        use_gui=False,
        use_visual_obs=True,
        need_offscreen_render=True,
        frame_skip=5,
        robot_name="panda"
    )
    env.seed(42)
    env.reset()

    # Set up camera system (headless rendering)
    add_default_scene_light(env.scene, env.renderer)
    gui = GUIBase(env.scene, env.renderer, headless=True)

    # Add cameras defined in YX_TABLE_TOP_CAMERAS
    for name, params in YX_TABLE_TOP_CAMERAS.items():
        if 'rotation' in params:
            gui.create_camera_from_pos_rot(position=params['position'], rotation=params['rotation'], name=name)
        else:
            gui.create_camera(position=params['position'], look_at_dir=params['look_at_dir'], right_dir=params['right_dir'], name=name)

    # Prepare to write video
    save_dir = "./video_output"
    os.makedirs(save_dir, exist_ok=True)
    video_path = os.path.join(save_dir, "hang_mug_env.mp4")

    writer = imageio.get_writer(video_path, fps=20)

    for i in range(200):
        action = np.zeros(env.arm_dof + 1)
        action[2] = 0.01  # small vertical motion
        obs, reward, done, _ = env.step(action)

        # List of RGB frames from all mounted cameras
        rgbs = gui.render()

        # Resize all views to the same shape (optional but recommended)
        resized_rgbs = [cv2.resize(rgb, (320, 240)) for rgb in rgbs]

        # 6 resized_rgbs, reshape to 2 rows × 3 columns
        row1 = np.concatenate(resized_rgbs[:3], axis=1)
        row2 = np.concatenate(resized_rgbs[3:], axis=1)
        panel = np.concatenate([row1, row2], axis=0)

        # Write to video
        writer.append_data(panel)

        # for rgb in rgbs:
        #     writer.append_data(rgb)

    writer.close()
    print(f"Saved video to {video_path}")


if __name__ == '__main__':
    main_env()
