import h5py
import numpy as np

# Load actions from training episode
#episode_path = "/home/hz2999/gendp/data/cube_picking_hdf5_Mar28/episode_0.hdf5"
episode_path = "/home/hz2999/gendp/data/cube_picking_hdf5_Apr9/episode_0.hdf5"
#episode_path = "/home/hz2999/gendp/data/sapien_demo/episode_0.hdf5"
with h5py.File(episode_path, "r") as f:
    np.set_printoptions(precision=4, suppress=True)

    cartesian_actions = f["cartesian_action"][:]  # shape: (T, 10)
    print(f"Loaded actions: {cartesian_actions.shape}")
    print(cartesian_actions[:5])

    # ee_pos = f["observations/ee_pos"][:]  # shape: (T, 7)
    # print(f"Loaded ee_pos: {ee_pos.shape}")
    # print(ee_pos[:5])

    # ee_vel = f["observations/robot_base_pose_in_world"][:]  # shape: (T, 7)
    # print(f"Loaded robot_base_pose_in_world: {ee_vel.shape}")
    # print(ee_vel[:5])






# # Inside run()
# start_frame = 0
# for step_count in range(max_steps):
#     used_action = np.zeros((self.n_action_steps, 7))  # (horizon, 7)
#     for j in range(self.n_action_steps):
#         global_step = step_count * self.n_action_steps + j
#         if global_step + 1 >= len(cartesian_actions):
#             break
#         action_full = cartesian_actions[global_step + 1]  # skip frame 0
#         used_action[j] = action_full[:7]  # [dx, dy, dz, 6D_rot[:3], gripper]

#     obs, reward, done, info = env.step(used_action)
#     print(f"Step {step_count}, used_action:\n{used_action}")
