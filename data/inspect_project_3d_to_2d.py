import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import h5py

def project_points_to_image(pcd, K, pose, image):
    """
    Project 3D points into 2D image space.
    pcd: [N, 3]
    K: [3, 3]
    pose: [4, 4]
    image: [H, W, 3]
    """
    # Convert 3D points to homogeneous
    N = pcd.shape[0]
    ones = np.ones((N, 1))
    pcd_h = np.concatenate([pcd, ones], axis=-1)  # [N, 4]

    # Transform points from world to camera
    pcd_cam = (np.linalg.inv(pose) @ pcd_h.T).T[:, :3]  # [N, 3]

    # Project to 2D using intrinsics
    uv = (K @ pcd_cam.T).T  # [N, 3]
    uv[:, :2] /= uv[:, 2:3]  # Normalize

    # Filter points inside image bounds
    H, W = image.shape[:2]
    mask = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H) & (pcd_cam[:, 2] > 0)
    uv_valid = uv[mask]
    
    # Draw on image
    image_out = image.copy()
    for pt in uv_valid.astype(np.int32):
        cv2.circle(image_out, (pt[0], pt[1]), radius=2, color=(0, 255, 0), thickness=-1)

    return image_out

# Load episode_0
hdf5_path = "/home/hz2999/gendp/data/cube_picking_hdf5_Mar28/episode_0.hdf5"
with h5py.File(hdf5_path, "r") as f:
    depth = f["observations/images/camera_0_depth"][3]  # (H, W)
    color = f["observations/images/camera_0_color"][3]  # (H, W, 3)
    K = f["observations/images/camera_0_intrinsics"][3]  # (3, 3)
    pose = f["observations/images/camera_0_extrinsics"][3]  # (4, 4)

# Convert depth to 3D point cloud
H, W = depth.shape
x, y = np.meshgrid(np.arange(W), np.arange(H))
x, y = x.flatten(), y.flatten()
depth_flat = depth.flatten()
X = (x - K[0, 2]) * depth_flat / K[0, 0]
Y = (y - K[1, 2]) * depth_flat / K[1, 1]
Z = depth_flat
pcd = np.stack([X, Y, Z], axis=-1)
valid = (Z > 0)
pcd = pcd[valid]

# Visualize
image_out = project_points_to_image(pcd, K, pose, color)

# Project and save
image_out = project_points_to_image(pcd, K, pose, color)
save_path = "./camera_3_proj_frame_0000.png"
cv2.imwrite(save_path, cv2.cvtColor(image_out, cv2.COLOR_RGB2BGR))
print(f"✅ Saved visualization to {save_path}")