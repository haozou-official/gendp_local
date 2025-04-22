import os
import numpy as np
import open3d as o3d
import imageio
from tqdm import tqdm
import scipy.spatial.transform as st

from gendp.common.data_utils import load_dict_from_hdf5
from gendp.common.kinematics_utils import KinHelper
from d3fields.utils.draw_utils import aggr_point_cloud_from_data, np2o3d, ImgEncoding
from open3d.visualization.rendering import OffscreenRenderer, MaterialRecord

# === Config ===
episode_id = 0
data_path = f"/home/hz2999/gendp/sapien_dataset/episode_{episode_id}.hdf5"
output_dir = f"/home/hz2999/gendp/vis_outputs/mug/episode_{episode_id}_3d"
os.makedirs(output_dir, exist_ok=True)

robot_name = "panda"
cam_keys = ['right_bottom_view', 'left_bottom_view', 'right_top_view', 'left_top_view']
boundaries = {
    'x_lower': -2.0,
    'x_upper': 2.0,
    'y_lower': -2.0,
    'y_upper': 2.0,
    'z_lower': -0.1,
    'z_upper': 2.0,
}

# === Load Data ===
data_dict, _ = load_dict_from_hdf5(data_path)
T = data_dict['observations']['images'][f'{cam_keys[0]}_color'].shape[0]
robot_base_seq = data_dict['observations']['robot_base_pose_in_world'][()]
kin_helper = KinHelper(robot_name=robot_name)
rendered_images = []

# === Setup Headless Renderer ===
width, height = 640, 480
renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
scene = renderer.scene
scene.set_background([1, 1, 1, 1])  # white background

# Materials
pcd_material = MaterialRecord()
pcd_material.shader = "defaultUnlit"

mesh_material = MaterialRecord()
mesh_material.shader = "defaultLit"

for t in tqdm(range(T), desc="Rendering Frames"):
    scene.clear_geometry()

    # --- Point Cloud ---
    colors = np.stack([data_dict['observations']['images'][f"{k}_color"][t] for k in cam_keys])
    depths = np.stack([data_dict['observations']['images'][f"{k}_depth"][t] for k in cam_keys]) / 1000.0
    intrinsics = np.stack([data_dict['observations']['images'][f"{k}_intrinsic"][t] for k in cam_keys])
    extrinsics = np.stack([data_dict['observations']['images'][f"{k}_extrinsic"][t] for k in cam_keys])

    pts, colors_pcd = aggr_point_cloud_from_data(
        colors, depths, intrinsics, extrinsics,
        out_o3d=False, downsample=False, boundaries=boundaries,
        color_fmt=ImgEncoding.BGR_UINT8
    )
    pts_homo = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=-1).T
    pts_robot = np.linalg.inv(robot_base_seq[t]) @ pts_homo
    pts_robot = pts_robot[:3].T

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_robot)
    pcd.colors = o3d.utility.Vector3dVector(colors_pcd)
    scene.add_geometry("pcd", pcd, pcd_material)

    # --- Robot Meshes ---
    robot_meshes = kin_helper.gen_robot_meshes(data_dict['observations']['full_joint_pos'][t])
    for j, mesh in enumerate(robot_meshes):
        scene.add_geometry(f"mesh_{j}", mesh, mesh_material)

    # --- Camera Settings ---
    center = np.array([0, 0, 0])
    eye = np.array([1.5, 0, 1.0])
    up = np.array([0, 0, 1])
    scene.camera.look_at(center, eye, up)

    # --- Render Frame ---
    img = renderer.render_to_image()
    image_path = os.path.join(output_dir, f"{t:04d}.png")
    o3d.io.write_image(image_path, img)
    rendered_images.append(imageio.imread(image_path))

# === Save MP4 ===
video_path = os.path.join(output_dir, "episode_vis.mp4")
imageio.mimsave(video_path, rendered_images, fps=10)
print(f"Saved MP4 to: {video_path}")
