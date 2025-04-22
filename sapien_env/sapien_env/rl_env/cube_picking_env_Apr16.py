from functools import cached_property
from typing import Optional

import sys
sys.path.append('/home/bing4090/yixuan_old_branch/general_dp/sapien_env')

import numpy as np
import sapien.core as sapien
import transforms3d
from sapien.utils import Viewer

from sapien_env.rl_env.base import BaseRLEnv
from sapien_env.sim_env.cube_picking_env import CubePickingEnv
from sapien_env.rl_env.para import ARM_INIT
from sapien_env.utils.common_robot_utils import generate_free_robot_hand_info, generate_arm_robot_hand_info, generate_panda_info

import pickle
#from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Rotation as SciRotation

class CubePickingRLEnv(CubePickingEnv, BaseRLEnv):
    def __init__(self, use_gui=False, frame_skip=5, randomness_scale=1, friction=1, robot_name="xarm6", **renderer_kwargs):
        super().__init__(use_gui, frame_skip, randomness_scale, friction, use_ray_tracing=False, **renderer_kwargs)
        self.setup(robot_name)

        # Parse link name
        if self.is_robot_free:
            info = generate_free_robot_hand_info()[robot_name]
        elif self.is_xarm:
            info = generate_arm_robot_hand_info()[robot_name]
        elif self.is_panda:
            info = generate_panda_info()[robot_name]
        else:
            raise NotImplementedError
        
        # print("[cube_picking_env] All robot link names:")
        # for link in self.robot.get_links():
        #     print(f"- {link.get_name()}")
        # print(f"[cube_picking_env] info: {info}")
        self.palm_link_name = info.palm_name
        self.palm_link = [link for link in self.robot.get_links() if link.get_name() == self.palm_link_name][0]
        
        # print(f"[cube_picking_env] palm_link_name: {self.palm_link_name}")
        # print(f"[cube_picking_env] palm_link: {self.palm_link}")
        
        #finger_tip_names = ["panda_leftfinger", "panda_rightfinger"]
        finger_tip_names = ["left_finger", "right_finger"]
        
        robot_link_names = [link.get_name() for link in self.robot.get_links()]
        self.finger_tip_links = [self.robot.get_links()[robot_link_names.index(name)] for name in finger_tip_names]

        self.object_episode_init_pose = sapien.Pose()

        # Get end-effector link pose
        eef_link = self.robot.get_links()[-1]  # The last link is the gripper
        eef_pose = eef_link.get_pose()
        print("📍 EE pose:", eef_pose)
        print("📍 EE position:", eef_pose.p)

    # def draw_calibration_axes(self, R_base2world, t_base2world):
    #     from scipy.spatial.transform import Rotation as R

    #     axis_length = 0.3
    #     axis_radius = 0.004
    #     cone_length = 0.05
    #     cone_radius = 0.01

    #     # Convert to sapien pose
    #     q_base2world = R.from_matrix(R_base2world).as_quat()
    #     base_pose = sapien.Pose(p=t_base2world, q=q_base2world)

    #     def draw_arrow(direction: np.ndarray, color: list):
    #         direction = direction / np.linalg.norm(direction)

    #         offset = direction * axis_length / 2
    #         rot = R.align_vectors([direction], [[1, 0, 0]])[0].as_quat()
    #         pose = base_pose * sapien.Pose(p=offset, q=rot)

    #         # Draw capsule (axis shaft)
    #         builder = self.scene.create_actor_builder()
    #         builder.add_capsule_visual(radius=axis_radius, half_length=axis_length / 2, color=color)
    #         axis = builder.build_static()
    #         axis.set_pose(pose)

    #         # Draw sphere at the tip (arrowhead)
    #         tip_offset = direction * axis_length
    #         sphere_pose = base_pose * sapien.Pose(p=tip_offset)
    #         sphere_builder = self.scene.create_actor_builder()
    #         sphere_builder.add_sphere_visual(radius=axis_radius * 2, color=color)
    #         sphere = sphere_builder.build_static()
    #         sphere.set_pose(sphere_pose)

    #     # Draw +X, +Y, +Z
    #     draw_arrow(np.array([1, 0, 0]), [1, 0, 0, 1])  # Red
    #     draw_arrow(np.array([0, 1, 0]), [0, 1, 0, 1])  # Green
    #     draw_arrow(np.array([0, 0, 1]), [0, 0, 1, 1])  # Blue

    # # Works partially (Z is not visiable and no direction arrow)
    # def draw_calibration_axes(self, R_base2world, t_base2world):
    #     axis_length = 0.3
    #     axis_radius = 0.004

    #     # Convert rotation matrix to quaternion
    #     from scipy.spatial.transform import Rotation as SciRotation
    #     quat_xyzw = SciRotation.from_matrix(R_base2world).as_quat()  # [x, y, z, w] SciRotation
    #     q_base2world = np.roll(quat_xyzw, 1)  # convert to [w, x, y, z]
    #     #print(f"[draw_calibration_axes()] q_base2world {q_base2world}") # [0. 1. 0. 0.]

    #     # Pose in world frame
    #     base_pose = sapien.Pose(p=t_base2world, q=q_base2world)  # This sets calibration world frame
    #     print(f"[draw_calibration_axes()] base_pose {base_pose}") # [-0.095, 0.085, -0.01], [1, 0, 0, 0]

    #     def draw_axis(direction: np.ndarray, color: list):
    #         # direction: a unit vector in calibration frame (e.g. [1,0,0] for X)
    #         # color: RGBA
    #         direction = direction / np.linalg.norm(direction)
    #         offset = direction * axis_length / 2  # Move half-length along the direction

    #         builder = self.scene.create_actor_builder()
    #         builder.add_capsule_visual(radius=axis_radius, half_length=axis_length / 2, color=color)

    #         # Compute rotation that aligns +X to the desired direction
    #         from scipy.spatial.transform import Rotation as R
    #         #rot = R.align_vectors([direction], [[1, 0, 0]])[0].as_quat()  # align `direction` to +X
    #         #rot = R.align_vectors([[1, 0, 0]], [direction])[0].as_quat()  # Might be wrong

    #         if np.allclose(direction, [1, 0, 0]):
    #             rot = [0, 0, 0, 1]
    #         elif np.allclose(direction, [0, 1, 0]):
    #             rot = transforms3d.euler.euler2quat(0, 0, np.pi / 2)
    #         elif np.allclose(direction, [0, 0, 1]):
    #             rot = transforms3d.euler.euler2quat(0, -np.pi / 2, 0)
    #         # Rotate the local offset into world direction
    #         rot_mat = R.from_quat(rot).as_matrix()
    #         rotated_offset = rot_mat @ (np.array([axis_length / 2, 0, 0]))  # Local X becomes world offset

    #         #pose = base_pose * sapien.Pose(p=[0, 0, 0], q=rot)
    #         #pose = base_pose * sapien.Pose(p=offset, q=rot)
    #         pose = base_pose * sapien.Pose(p=rotated_offset, q=rot)
    #         print(f"[draw_axis()] pose {pose}")
            
    #         axis = builder.build_static(name=f"axis_{direction}")
    #         print(f"Built axis {direction} with color {color} and pose {pose}")
    #         axis.set_pose(pose)

    #     # Draw X (forward), Y (right), Z (downward)
    #     draw_axis(np.array([1, 0, 0]), [1, 0, 0, 1])  # Red: X --> Blue --> Forward
    #     draw_axis(np.array([0, 1, 0]), [0, 1, 0, 1])  # Green: Y --> Green
    #     # Pose([-0.095, 0.235, -0.01], [0, 0, 0.707107, -0.707107]), rotation of -90° around Z axis

    #     draw_axis(np.array([0, 0, 1]), [0, 0, 1, 1])  # Blue: Z --> Red --> Right


    # def draw_calibration_axes(self, origin: sapien.Pose):
    #     axis_length = 0.15  # Adjust to your scene scale
    #     axis_radius = 0.004

    #     def build_axis(direction_quat, color, offset_vec):
    #         builder = self.scene.create_actor_builder()
    #         builder.add_capsule_visual(radius=axis_radius, half_length=axis_length / 2, color=color)
    #         actor = builder.build_static()

    #         # Apply offset along capsule's local +X axis (because capsule is aligned with X by default)
    #         local_pose = sapien.Pose(p=offset_vec, q=direction_quat)
    #         world_pose = origin * local_pose
    #         actor.set_pose(world_pose)

    #     # X-axis: red, no rotation needed (aligned with +X)
    #     build_axis(
    #         direction_quat=transforms3d.euler.euler2quat(0, 0, 0),
    #         color=[1, 0, 0, 1],
    #         offset_vec=[axis_length / 2, 0, 0]
    #     )

    #     # Y-axis: green, rotate +Z (yaw) 90° to point capsule toward +Y
    #     build_axis(
    #         direction_quat=transforms3d.euler.euler2quat(0, 0, np.pi / 2),
    #         color=[0, 1, 0, 1],
    #         offset_vec=[axis_length / 2, 0, 0]
    #     )

    #     # Z-axis: blue, rotate -Y (pitch) 90° to point capsule toward +Z
    #     build_axis(
    #         direction_quat=transforms3d.euler.euler2quat(0, -np.pi / 2, 0),
    #         color=[0, 0, 1, 1],
    #         offset_vec=[axis_length / 2, 0, 0]
    #     )

    # def draw_calibration_axes(self, origin: sapien.Pose):
    #     axis_length = 0.3
    #     axis_radius = 0.003

    #     def draw_axis(axis: str, color):
    #         builder = self.scene.create_actor_builder()
    #         builder.add_capsule_visual(radius=axis_radius, half_length=axis_length / 2, color=color)
    #         actor = builder.build_static()

    #         if axis == "x":
    #             rot = transforms3d.euler.euler2quat(0, 0, 0)
    #             offset = [axis_length / 2, 0, 0]
    #         elif axis == "y":
    #             rot = transforms3d.euler.euler2quat(0, 0, np.pi / 2)  # Rotate around Z
    #             offset = [axis_length / 2, 0, 0]
    #         elif axis == "z":
    #             rot = transforms3d.euler.euler2quat(0, -np.pi / 2, 0)  # Rotate around Y
    #             offset = [axis_length / 2, 0, 0]
    #         else:
    #             raise ValueError("Axis must be x, y, or z")

    #         pose = origin * sapien.Pose(p=offset, q=rot)
    #         actor.set_pose(pose)

    #     # Draw red X, green Y, blue Z
    #     draw_axis("x", [1, 0, 0, 1])  # Red
    #     draw_axis("y", [0, 1, 0, 1])  # Green
    #     draw_axis("z", [0, 0, 1, 1])  # Blue

    def get_oracle_state(self):
        # Robot joint positions
        robot_qpos_vec = self.robot.get_qpos()

        # Cube pose (position + quaternion)
        cube_pose = self.cube.get_pose()
        cube_pose_vec = np.concatenate([cube_pose.p, cube_pose.q])

        # Cube velocities
        linear_v = self.cube.get_velocity()
        angular_v = self.cube.get_angular_velocity()

        # Distance between cube and robot palm
        palm_pose = self.palm_link.get_pose()
        cube_in_palm = cube_pose.p - palm_pose.p

        # Cube uprightness (alignment of z-axis)
        z_axis = cube_pose.to_transformation_matrix()[:3, 2]
        theta_cos = np.dot(z_axis, np.array([0, 0, 1]))

        # Combine into oracle state vector
        return np.concatenate([
            robot_qpos_vec,
            cube_pose_vec,
            linear_v,
            angular_v,
            cube_in_palm,
            np.array([theta_cos])
        ])

    def get_robot_state(self):
        robot_qpos_vec = self.robot.get_qpos()
        palm_pose = self.palm_link.get_pose()
        return np.concatenate([robot_qpos_vec, palm_pose.p])

    def get_reward(self, action):
        # Reward is 1.0 if the cube is close to the gripper and lifted above a height threshold
        cube_pose = self.cube.get_pose()
        palm_pose = self.palm_link.get_pose()

        # Distance threshold in XY plane
        xy_dist = np.linalg.norm(cube_pose.p[:2] - palm_pose.p[:2])
        xy_thresh = 0.05

        # Height threshold for lifting
        z_thresh = 0.65  # Table is at z=0.6, cube lifted if > 0.65

        reward = (xy_dist < xy_thresh) and (cube_pose.p[2] > z_thresh)
        return float(reward)

    def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None):
        # Set robot initial joint config and pose
        if self.is_xarm:
            qpos = np.zeros(self.robot.dof)
            arm_qpos = self.robot_info.arm_init_qpos
            qpos[:self.arm_dof] = arm_qpos
            self.robot.set_qpos(qpos)
            self.robot.set_drive_target(qpos)

            # init_pos = ARM_INIT + self.robot_info.root_offset
            # init_ori = transforms3d.euler.euler2quat(0, np.pi, np.pi)  # derived from R_base2world
            # init_pose = sapien.Pose(init_pos, init_ori)

            print(f"[cube_picking_env] hardcoded offset: {ARM_INIT}")
            print(f"[cube_picking_env] model-specific offset: {self.robot_info.root_offset}")
            # # Set the robot base using base.pkl
            # #init_pose = sapien.Pose(init_pos, transforms3d.euler.euler2quat(0, 0, 0))  # Facing toward X+
            # init_pos = np.array([-0.095, 0.085, -0.01])
            # #init_pos = np.array([-0.095, 0.085, 0.0])
            # #init_ori = transforms3d.euler.euler2quat(0, np.pi, np.pi)  # derived from R_base2world
            # #init_ori = transforms3d.euler.euler2quat(0, 0, 0)  # No rotation; works
            # #init_ori = transforms3d.euler.euler2quat(0, -np.pi / 2, 0) # Wrong
            # #init_ori = transforms3d.euler.euler2quat(0, 0, -np.pi / 2)
            # #init_ori = transforms3d.euler.euler2quat(0, 0, np.pi / 2)

            # init_ori = [0.0, 1.0, 0.0, 0.0]
            # #init_ori = [0.0, 0.0, 0.0, 1.0]
            # #init_ori = [1.0, 0.0, 0.0, 0.0]
            # init_pose = sapien.Pose(init_pos, init_ori)
            # print("📍 [Debug] Set xArm pose from base.pkl:", init_pose)

            # === Load base pose from base.pkl ===
            with open("/home/hz2999/gendp/data/cube_picking_processed/episode_0000/calibration/base.pkl", "rb") as f:
                base_info = pickle.load(f)

            R_base2world = base_info["R_base2world"]
            t_base2world = base_info["t_base2world"]

            # Convert rotation matrix to quaternion [w, x, y, z]
            quat_xyzw = SciRotation.from_matrix(R_base2world).as_quat()
            quat_wxyz = np.roll(quat_xyzw, 1)
            print(f"[reset()] quat_xyzw {quat_wxyz}")

            # Set pose
            init_pos = np.array(t_base2world)
            init_ori = quat_wxyz
            init_pose = sapien.Pose(init_pos, init_ori)

            print("📍 [Debug] Set xArm pose from base.pkl:", init_pose)

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
        print("📍 [Debug] Set robot init pose to:", init_pose.p)
        
        # Draw XYZ Axes
        #self.draw_calibration_axes(init_pose)
        self.draw_calibration_axes(R_base2world, t_base2world)

        self.reset_env()
        self.reset_internal()

        # Stabilize simulation before placing objects
        for _ in range(100):
            self.robot.set_qf(self.robot.compute_passive_force(external=False, coriolis_and_centrifugal=False))
            self.scene.step()

        # if hasattr(self, "init_states") and self.init_states is not None:
        #     print(f"[init_states]")
        #     self.set_init(self.init_states)
        # else:
        #     # cube_center_pos = np.array([0.0, 0.0, 0.3])  # Z = table height (0.6) + half cube height (0.025)
        #     # cube_center_quat = transforms3d.euler.euler2quat(0, 0, 0)
        #     cube_center_pos = np.array([0.12, -0.18, -0.475])  # in meters
        #     cube_center_quat = np.array([0.0, 0.0, 0.38268343, 0.92387953])  # 45° around Z
        #     fixed_pose = sapien.Pose(cube_center_pos, cube_center_quat)
        #     self.object_episode_init_pose = fixed_pose
        #     self.cube.set_pose(fixed_pose)


        return self.get_observation()


    def set_init(self, init_states):
        cube_pose = sapien.Pose.from_transformation_matrix(init_states[0])
        self.cube.set_pose(cube_pose)
        self.object_episode_init_pose = cube_pose


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

# Dynamically layout camera views in a grid
def layout_grid(images, cols=3):
    import math
    rows = math.ceil(len(images) / cols)
    blank = np.zeros_like(images[0])
    padded = images + [blank] * (rows * cols - len(images))  # pad to full grid
    grid_rows = [np.concatenate(padded[i * cols:(i + 1) * cols], axis=1) for i in range(rows)]
    return np.concatenate(grid_rows, axis=0)

def main_env():
    import imageio
    import os
    from sapien_env.gui.gui_base import GUIBase, YX_TABLE_TOP_CAMERAS, DEFAULT_TABLE_TOP_CAMERAS
    import sapien.core as sapien
    import cv2  # For resizing

    # Setup environment
    env = CubePickingRLEnv(
        use_gui=False,
        use_visual_obs=True,
        need_offscreen_render=True,
        frame_skip=5,
        robot_name="xarm6_with_gripper"
        #robot_name="panda"
    )
    env.seed(42)
    env.reset()

    # # Draw the calibration frame after environment reset
    # base_pose = sapien.Pose(p=[-0.095, 0.085, -0.01], q=[1, 0, 0, 0])
    # env.draw_calibration_axes(base_pose)  

    # Set up camera system (headless rendering)
    add_default_scene_light(env.scene, env.renderer)
    gui = GUIBase(env.scene, env.renderer, headless=True)

    # # Add predefined cameras (if you still want them)
    # for name, params in YX_TABLE_TOP_CAMERAS.items():
    #     if 'rotation' in params:
    #         gui.create_camera_from_pos_rot(position=params['position'], rotation=params['rotation'], name=name)
    #     else:
    #         gui.create_camera(position=params['position'], look_at_dir=params['look_at_dir'], right_dir=params['right_dir'], name=name)

    # DEFAULT_TABLE_TOP_CAMERAS
    for name, params in DEFAULT_TABLE_TOP_CAMERAS.items():
        if 'rotation' in params:
            gui.create_camera_from_pos_rot(position=params['position'], rotation=params['rotation'], name=name)
        else:
            gui.create_camera(position=params['position'], look_at_dir=params['look_at_dir'], right_dir=params['right_dir'], name=name)

    # # === ✅ Add custom debug camera pointing at robot base ===
    # robot_base_pos = [-0.095, 0.085, -0.01]  # From base.pkl
    # cam_pos = [robot_base_pos[0], robot_base_pos[1], 0.4]  # 40cm above robot base
    # look_at = np.array(robot_base_pos)
    # camera_pos = np.array(cam_pos)
    # look_dir = look_at - camera_pos

    # print("Camera Position:", cam_pos)
    # print("Camera Look Dir:", look_dir)

    # gui.create_camera(position=cam_pos, look_at_dir=look_dir.tolist(), right_dir=[1, 0, 0], name="debug_topdown")

    # Prepare to write video
    save_dir = "./video_output"
    os.makedirs(save_dir, exist_ok=True)
    video_path = os.path.join(save_dir, "cube_picking_env.mp4")

    writer = imageio.get_writer(video_path, fps=20)

    for i in range(1):
        action = np.zeros(env.arm_dof + 1)
        action[2] = 0.01  # small vertical motion
        obs, reward, done, _ = env.step(action)

        # List of RGB frames from all mounted cameras
        rgbs = gui.render()

        # Resize all views to the same shape
        resized_rgbs = [cv2.resize(rgb, (320, 240)) for rgb in rgbs]

        # # Split views into 2 rows × N/2 columns
        # n = len(resized_rgbs)
        # half = (n + 1) // 2
        # row1 = np.concatenate(resized_rgbs[:half], axis=1)
        # row2 = np.concatenate(resized_rgbs[half:], axis=1)
        # panel = np.concatenate([row1, row2], axis=0)

        panel = layout_grid(resized_rgbs, cols=3)

        writer.append_data(panel)

    writer.close()
    print(f"Saved video to {video_path}")


# def main_env():
#     import imageio
#     import os
#     from sapien_env.gui.gui_base import GUIBase, YX_TABLE_TOP_CAMERAS
#     #from sapien_env.utils.render_scene_utils import add_default_scene_light
#     import sapien.core as sapien
#     import cv2  # For resizing

#     # Setup environment
#     env = CubePickingRLEnv(
#         use_gui=False,
#         use_visual_obs=True,
#         need_offscreen_render=True,
#         frame_skip=5,
#         robot_name="xarm6_with_gripper"
#     )
#     env.seed(42)
#     env.reset()

#     # Set up camera system (headless rendering)
#     add_default_scene_light(env.scene, env.renderer)
#     gui = GUIBase(env.scene, env.renderer, headless=True)

#     # Add cameras defined in YX_TABLE_TOP_CAMERAS
#     for name, params in YX_TABLE_TOP_CAMERAS.items():
#         if 'rotation' in params:
#             gui.create_camera_from_pos_rot(position=params['position'], rotation=params['rotation'], name=name)
#         else:
#             gui.create_camera(position=params['position'], look_at_dir=params['look_at_dir'], right_dir=params['right_dir'], name=name)

#     # Prepare to write video
#     save_dir = "./video_output"
#     os.makedirs(save_dir, exist_ok=True)
#     video_path = os.path.join(save_dir, "cube_picking_env.mp4")

#     writer = imageio.get_writer(video_path, fps=20)

#     # for i in range(200):
#     for i in range(200):
#         action = np.zeros(env.arm_dof + 1)
#         action[2] = 0.01  # small vertical motion
#         obs, reward, done, _ = env.step(action)

#         # List of RGB frames from all mounted cameras
#         rgbs = gui.render()

#         # Resize all views to the same shape (optional but recommended)
#         resized_rgbs = [cv2.resize(rgb, (320, 240)) for rgb in rgbs]

#         # 6 resized_rgbs, reshape to 2 rows × 3 columns
#         row1 = np.concatenate(resized_rgbs[:3], axis=1)
#         row2 = np.concatenate(resized_rgbs[3:], axis=1)
#         panel = np.concatenate([row1, row2], axis=0)

#         # Write to video
#         writer.append_data(panel)

#         # for rgb in rgbs:
#         #     writer.append_data(rgb)

#     writer.close()
#     print(f"Saved video to {video_path}")


if __name__ == '__main__':
    main_env()
