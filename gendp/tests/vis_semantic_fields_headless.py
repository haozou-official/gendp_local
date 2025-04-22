import os
import numpy as np
import torch
import imageio
from tqdm import tqdm
from matplotlib import colormaps
import scipy.spatial.transform as st

from gendp.common.data_utils import load_dict_from_hdf5, d3fields_proc
from gendp.common.kinematics_utils import KinHelper
from d3fields.fusion import Fusion
from d3fields.utils.draw_utils import np2o3d
import open3d as o3d

### hyper param
episode_id = 0
data_path = f"/home/hz2999/gendp/sapien_dataset/episode_{episode_id}.hdf5"
output_dir = f"/home/hz2999/gendp/vis_outputs/mug/episode_{episode_id}_semantic"
os.makedirs(output_dir, exist_ok=True)

vis_robot = True
vis_action = True
robot_name = 'panda'
cam_keys = ['right_bottom_view', 'left_bottom_view', 'right_top_view', 'left_top_view']

shape_meta = {
    'shape': [6, 4000],
    'type': 'spatial',
    'info': {
        'reference_frame': 'world',
        'distill_dino': True,
        'distill_obj': 'mug',
        'view_keys': cam_keys,
        'N_gripper': 400,
        'boundaries': {
            'x_lower': -0.35,
            'x_upper': 0.35,
            'y_lower': -0.3,
            'y_upper': 0.5,
            'z_lower': 0.01,
            'z_upper': 0.5
        },
        'resize_ratio': 0.5
    }
}

kin_helper = KinHelper(robot_name='panda')
fusion = Fusion(num_cam=len(cam_keys), dtype=torch.float16)

data_dict, _ = load_dict_from_hdf5(data_path)
T = data_dict['observations']['images'][f'{cam_keys[0]}_color'].shape[0]
robot_base_seq = data_dict['observations']['robot_base_pose_in_world'][()]
rendered_images = []

for t in tqdm(range(T), desc="Rendering Frames"):
    robot_base = robot_base_seq[t]
    colors = np.stack([data_dict['observations']['images'][f"{k}_color"][t:t+1] for k in cam_keys], axis=1)
    depths = np.stack([data_dict['observations']['images'][f"{k}_depth"][t:t+1] for k in cam_keys], axis=1) / 1000.0
    intrinsics = np.stack([data_dict['observations']['images'][f"{k}_intrinsic"][t:t+1] for k in cam_keys], axis=1)
    extrinsics = np.stack([data_dict['observations']['images'][f"{k}_extrinsic"][t:t+1] for k in cam_keys], axis=1)

    pcd, pcd_feats = d3fields_proc(
        fusion=fusion,
        shape_meta=shape_meta,
        color_seq=colors,
        depth_seq=depths,
        extri_seq=extrinsics,
        intri_seq=intrinsics,
        robot_base_pose_in_world_seq=robot_base_seq,
        teleop_robot=kin_helper,
        qpos_seq=data_dict['observations']['full_joint_pos'][t:t+1],
    )

    pcd = pcd[0]
    pcd_feats = pcd_feats[0]
    pcd = np.linalg.inv(robot_base) @ np.concatenate([pcd, np.ones((pcd.shape[0], 1))], axis=-1).T
    pcd = pcd.T[:, :3]

    feats_cmap = colormaps.get_cmap('viridis')
    pcd_colors = feats_cmap(pcd_feats[:, 0])[:, :3]
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
    pcd_o3d.colors = o3d.utility.Vector3dVector(pcd_colors)

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(pcd_o3d)

    if vis_robot:
        meshes = kin_helper.gen_robot_meshes(qpos=data_dict['observations']['full_joint_pos'][t])
        for mesh in meshes:
            vis.add_geometry(mesh)

    vis.poll_events()
    vis.update_renderer()

    img_path = os.path.join(output_dir, f"frame_{t:04d}.png")
    vis.capture_screen_image(img_path)
    vis.destroy_window()

    rendered_images.append(imageio.v3.imread(img_path))

# Save GIF
gif_path = os.path.join(output_dir, "semantic_vis.gif")
imageio.mimsave(gif_path, rendered_images, duration=0.1)
print(f"Saved semantic field GIF to: {gif_path}")
