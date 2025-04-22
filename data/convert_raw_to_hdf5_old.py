import os
import h5py
import numpy as np
from PIL import Image
import cv2

def load_robot_state(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    data = [float(val) for line in lines for val in line.strip().split()]
    return np.array(data, dtype=np.float32)

def rotation_matrix_to_6d(matrix):
    return matrix[:, :2].reshape(-1)  # (6,)

def compute_action(prev, curr):
    pos_prev, pos_curr = prev[0:3], curr[0:3]
    rot_prev = prev[3:12].reshape(3, 3)
    rot_curr = curr[3:12].reshape(3, 3)
    grip_prev, grip_curr = prev[12], curr[12]

    dpos = pos_curr - pos_prev
    drot = rotation_matrix_to_6d(rot_curr) - rotation_matrix_to_6d(rot_prev)
    dgrip = np.array([grip_curr - grip_prev], dtype=np.float32)
    return np.concatenate([dpos, drot, dgrip])

def create_hdf5_from_episode(episode_dir, output_path):
    robot_txts = sorted(os.listdir(os.path.join(episode_dir, "robot")))
    num_frames = len(robot_txts)
    cameras = [f"camera_{i}" for i in range(4)]

    robot_data_all = [load_robot_state(os.path.join(episode_dir, "robot", f)) for f in robot_txts]

    with h5py.File(output_path, "w") as hf:
        hf.create_dataset("cartesian_action", shape=(num_frames, 10), dtype=np.float32)
        hf.create_dataset("joint_action", shape=(num_frames, 7), dtype=np.float32)  # dummy

        obs = hf.create_group("observations")
        obs.create_dataset("ee_pos", shape=(num_frames, 7), dtype=np.float32)
        obs.create_dataset("joint_pos", shape=(num_frames, 7), dtype=np.float32)
        obs.create_dataset("full_joint_pos", shape=(num_frames, 8), dtype=np.float32)
        obs.create_dataset("robot_base_pose_in_world", shape=(num_frames, 1, 4, 4), dtype=np.float32)

        image_grp = obs.create_group("images")
        for cam in cameras:
            image_grp.create_dataset(f"{cam}_color", shape=(num_frames, 480, 640, 3), dtype=np.uint8)
            image_grp.create_dataset(f"{cam}_depth", shape=(num_frames, 480, 640), dtype=np.float32)
            image_grp.create_dataset(f"{cam}_intrinsics", shape=(num_frames, 3, 3), dtype=np.float32)
            image_grp.create_dataset(f"{cam}_extrinsics", shape=(num_frames, 4, 4), dtype=np.float32)

        hf.create_dataset("timestamp", shape=(num_frames,), dtype=np.float64)
        hf.create_dataset("stage", shape=(num_frames,), dtype=np.float32)

        for i in range(num_frames):
            idx_str = f"{i:06d}"
            robot_data = robot_data_all[i]
            pos, rot, grip = robot_data[0:3], robot_data[3:12].reshape(3, 3), robot_data[12]
            ee_pose_7d = np.concatenate([pos, rot[:, 0], [grip]])
            obs["ee_pos"][i] = ee_pose_7d

            T = np.eye(4, dtype=np.float32)
            T[:3, :3] = rot
            T[:3, 3] = pos
            obs["robot_base_pose_in_world"][i, 0] = T

            # Real Action
            if i == 0:
                hf["cartesian_action"][i] = np.zeros(10, dtype=np.float32)
            else:
                action = compute_action(robot_data_all[i-1], robot_data_all[i])
                hf["cartesian_action"][i] = action

            # Dummy values
            hf["joint_action"][i] = np.zeros(7)
            obs["joint_pos"][i] = np.zeros(7)
            obs["full_joint_pos"][i] = np.zeros(8)
            hf["timestamp"][i] = i * 0.1
            hf["stage"][i] = 0.0

            for cam_id in range(4):
                cam = f"camera_{cam_id}"
                rgb_path = os.path.join(episode_dir, cam, "rgb", f"{idx_str}.jpg")
                depth_path = os.path.join(episode_dir, cam, "depth", f"{idx_str}.png")

                rgb = np.array(Image.open(rgb_path).resize((640, 480)))
                depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)

                #print(f"depth_raw: {depth_raw}")
                #print(f"[{idx_str}] depth_raw stats → min: {depth_raw.min()}, max: {depth_raw.max()}, mean: {depth_raw.mean()}, dtype: {depth_raw.dtype}")
                #depth_m = cv2.resize(depth_raw, (640, 480)) / 1000.0 
                depth_m = cv2.resize(depth_raw, (640, 480)) 
                #print(f"depth_m in meters: {depth_m}")
                #print(f"[{idx_str}] depth_m stats → min: {depth_m.min()}, max: {depth_m.max()}, mean: {depth_m.mean()}")

                intrinsics = np.load(os.path.join(episode_dir, "calibration", "intrinsics.npy"))[cam_id] 
                extrinsics = np.load(os.path.join(episode_dir, "calibration", "extrinsics.npy"))[cam_id] 
                #extrinsics = np.load(os.path.join(episode_dir, "calibration", "extrinsics.npy"))[0]
                image_grp[f"{cam}_intrinsics"][i] = intrinsics
                image_grp[f"{cam}_extrinsics"][i] = extrinsics

                # # Debug: project depth to 3D point cloud and print info
                # fx, fy = intrinsics[0, 0], intrinsics[1, 1]
                # cx, cy = intrinsics[0, 2], intrinsics[1, 2]

                # height, width = depth_m.shape
                # points_3d = []

                # for v in range(height):
                #     for u in range(width):
                #         Z = depth_m[v, u]
                #         if Z > 0 and Z < 5:  # ignore invalid depth
                #             X = (u - cx) * Z / fx
                #             Y = (v - cy) * Z / fy
                #             points_3d.append([X, Y, Z])

                # points_3d = np.array(points_3d)
                # print(f"[Frame {i} | {cam}] Projected 3D points: {points_3d.shape}")
                # if points_3d.shape[0] > 0:
                #     print(f"First 5 points: {points_3d[:5]}")
                # else:
                #     print("No valid 3D points found!")


                image_grp[f"{cam}_color"][i] = rgb
                image_grp[f"{cam}_depth"][i] = depth_m

    print(f"Saved: {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode_dir")
    parser.add_argument("--output_path")
    args = parser.parse_args()
    print(f"📂 Converting {args.episode_dir} episode to: {args.output_path}")
    #os.makedirs(args.output_path, exist_ok=True)
    #os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    create_hdf5_from_episode(args.episode_dir, args.output_path)
