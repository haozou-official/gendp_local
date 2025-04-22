#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import cv2
import matplotlib
from gendp.common.data_utils import load_dict_from_hdf5

# === Modify here ===
episode_idx = 0
# data_path = f"/home/hz2999/gendp/sapien_dataset/episode_{episode_idx}.hdf5"
# output_dir = f"./vis_outputs/mug/episode_{episode_idx}"
data_path = f"/home/hz2999/gendp/data/real_aloha_demo/knife_real/episode_{episode_idx}.hdf5"
output_dir = f"./vis_outputs/knife/episode_{episode_idx}"
os.makedirs(output_dir, exist_ok=True)

# obs_keys = [
#     'front_view_color',
#     'front_view_depth',
# ]
# obs_format = [
#     'bgr',
#     'uint16depth',
# ]

obs_keys = [
    'camera_0_color',
    'camera_0_depth',
]
obs_format = [
    'bgr',
    'uint16depth',
]

def vis_format(img, format):
    if format == 'bgr':
        return img
    elif format == 'rgb':
        return img[..., ::-1]
    elif format == 'uint16depth':
        cmap = matplotlib.colormaps.get_cmap('plasma')
        depth_norm = img / 1000.0
        depth_norm = np.clip(depth_norm, 0, 1)
        depth_vis = cmap(depth_norm)[:, :, :3]
        return (depth_vis * 255).astype(np.uint8)[..., ::-1]
    elif format == 'float32depth':
        cmap = matplotlib.colormaps.get_cmap('plasma')
        depth_norm = img / 1000.0
        depth_norm = np.clip(depth_norm, 0, 1)
        depth_vis = cmap(depth_norm)[:, :, :3]
        return (depth_vis * 255).astype(np.uint8)[..., ::-1]

print(f'Visualizing episode {episode_idx}')
data_dict, _ = load_dict_from_hdf5(data_path)

print('All obs keys:', list(data_dict['observations']['images'].keys()))

obs_ls = [data_dict['observations']['images'][k] for k in obs_keys]
T = obs_ls[0].shape[0]

for obs_i, obs in enumerate(obs_ls):
    obs_key = obs_keys[obs_i]
    fmt = obs_format[obs_i]
    key_dir = os.path.join(output_dir, obs_key)
    os.makedirs(key_dir, exist_ok=True)

    for t in range(T):
        img = vis_format(obs[t], fmt)
        out_path = os.path.join(key_dir, f"{t:04d}.png")
        cv2.imwrite(out_path, img)

print(f"Saved visualizations to {output_dir}")
