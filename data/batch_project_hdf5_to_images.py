import os
import numpy as np
import cv2
import h5py
from tqdm import tqdm

def project_points_to_image(pcd, K, pose, image):
    # Project 3D pcd onbto 2D image using K and pose
    N = pcd.shape[0]
    ones = np.ones((N, 1))
    pcd_h = np.concatenate([pcd, ones], axis=-1)

    # Transform to camera space
    pose_mm = pose.copy()
    pose_mm[:3, 3] *= 1000.0
    pcd_cam = (np.linalg.inv(pose_mm) @ pcd_h.T).T[:, :3]  # World Frame to Cam
    print(f"[Debug] pcd_cam Z-range: {pcd_cam[:, 2].min()} to {pcd_cam[:, 2].max()}")

    # Project using intrinsics
    uv = (K @ pcd_cam.T).T
    uv[:, :2] /= uv[:, 2:3]

    H, W = image.shape[:2]
    mask = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H) & (pcd_cam[:, 2] > 0)
    uv_valid = uv[mask]
    print(f"[Debug] uv sample: {uv[:5]}")
    print(f"[Debug] image size: H={H}, W={W}")
    print(f"[Debug] valid projections: {mask.sum()} / {mask.shape[0]}")

    image_out = image.copy()
    for pt in uv_valid.astype(np.int32):
        cv2.circle(image_out, (pt[0], pt[1]), radius=2, color=(0, 255, 0), thickness=-1)
    return image_out

def depth_to_pcd(depth, K):
    H, W = depth.shape
    x, y = np.meshgrid(np.arange(W), np.arange(H))
    x, y = x.flatten(), y.flatten()
    depth_flat = depth.flatten()

    X = (x - K[0, 2]) * depth_flat / K[0, 0]
    Y = (y - K[1, 2]) * depth_flat / K[1, 1]
    Z = depth_flat
    pcd = np.stack([X, Y, Z], axis=-1)
    valid = (Z > 0)
    return pcd[valid]

def batch_visualize_all_cams(hdf5_path, out_dir, n_frames=5):
    os.makedirs(out_dir, exist_ok=True)

    with h5py.File(hdf5_path, "r") as f:
        for i in tqdm(range(n_frames), desc="Rendering frames"):
            for cam_id in range(4):
                cam = f"camera_{cam_id}"

                # Load data
                color = f[f"observations/images/{cam}_color"][i]
                depth = f[f"observations/images/{cam}_depth"][i]
                K = f[f"observations/images/{cam}_intrinsics"][i]
                pose = f[f"observations/images/{cam}_extrinsics"][i]

                # Project
                pcd = depth_to_pcd(depth, K)
                projected = project_points_to_image(pcd, K, pose, color)

                # Save
                out_path = os.path.join(out_dir, f"proj_frame_{i}_{cam}.png")
                cv2.imwrite(out_path, cv2.cvtColor(projected, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    hdf5_path = "/home/hz2999/gendp/data/cube_picking_hdf5_Mar28/episode_0.hdf5"
    #hdf5_path = "/home/hz2999/gendp/data/real_aloha_demo/knife_real/episode_0.hdf5"
    out_dir = "./projection_vis"
    batch_visualize_all_cams(hdf5_path, out_dir, n_frames=5)
