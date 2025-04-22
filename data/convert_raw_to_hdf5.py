from scipy.spatial.transform import Rotation as R
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

def extract_euler_angles_from_matrix(rot_matrix):
    return R.from_matrix(rot_matrix).as_euler('xyz', degrees=False)

def create_hdf5_from_episode(episode_dir, output_path):
    robot_txts = sorted(os.listdir(os.path.join(episode_dir, "robot")))
    num_frames = len(robot_txts)
    cameras = [f"camera_{i}" for i in range(4)]

    robot_data_all = [load_robot_state(os.path.join(episode_dir, "robot", f)) for f in robot_txts]

    with h5py.File(output_path, "w") as hf:
        hf.create_dataset("cartesian_action", shape=(num_frames, 7), dtype=np.float32)  # x, y, z, rx, ry, rz, grip
        hf.create_dataset("joint_action", shape=(num_frames, 7), dtype=np.float32)  # dummy

        obs = hf.create_group("observations")
        obs.create_dataset("ee_pos", shape=(num_frames, 7), dtype=np.float32)  # same format
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

        T_base = np.eye(4, dtype=np.float32)
        T_base[:3, :3] = np.array([
            [ 0., -1.,  0.],
            [ 1.,  0.,  0.],
            [ 0.,  0.,  1.]
        ])
        T_base[:3, 3] = np.array([0.0, -0.5, 0.0], dtype=np.float32)

        # Load calibration
        import pickle
        with open(os.path.join(episode_dir, "calibration", "base.pkl"), "rb") as f:
            base = pickle.load(f)
        R_base2world = base["R_base2world"]
        t_base2world = base["t_base2world"]
        T_base2world = np.eye(4)
        T_base2world[:3, :3] = R_base2world
        T_base2world[:3, 3] = t_base2world

        R_calib2sim = np.array([
            [ 1.,  0.,  0.],
            [ 0., -1.,  0.],
            [ 0.,  0., -1.]
        ])  # flip Y and Z

        t_calib2sim = np.array([-0.305, 0.085, -0.01])  # translation shift

        for i in range(num_frames):
            idx_str = f"{i:06d}"
            robot_data = robot_data_all[i]
            pos, rot, grip = robot_data[0:3], robot_data[3:12].reshape(3, 3), robot_data[12]
            
            # Convert pos from calibration world → sim world
            pos_sim = R_calib2sim @ pos + t_calib2sim

            # Convert orientation
            rot_sim = R_calib2sim @ rot

            euler_sim = extract_euler_angles_from_matrix(rot_sim)
            ee_pose_7d = np.concatenate([pos_sim, euler_sim, [grip]])

            obs["ee_pos"][i] = ee_pose_7d
            hf["cartesian_action"][i] = ee_pose_7d
            #print("pos_world:", pos_world)
            #print("euler_world:\n", euler_world)
            #print("T_base2world:", T_base2world)
            
            T_sim_base = np.eye(4)
            T_sim_base[:3, 3] = np.array([-0.4, 0.0, 0.0])
            obs["robot_base_pose_in_world"][i, 0] = T_sim_base
            #obs["robot_base_pose_in_world"][i, 0] = T_base2world
            #print("hardcoded T_base:", T_base)

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
                depth_m = cv2.resize(depth_raw, (640, 480))  # Optional: scale to meters

                intrinsics = np.load(os.path.join(episode_dir, "calibration", "intrinsics.npy"))[cam_id]
                extrinsics = np.load(os.path.join(episode_dir, "calibration", "extrinsics.npy"))[cam_id]
                #print("extrinsics:", extrinsics)
                image_grp[f"{cam}_intrinsics"][i] = intrinsics
                image_grp[f"{cam}_extrinsics"][i] = extrinsics
                image_grp[f"{cam}_color"][i] = rgb
                image_grp[f"{cam}_depth"][i] = depth_m

    print(f"✅ Saved HDF5 to: {output_path}")

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