#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import cv2
import matplotlib
from gendp.common.data_utils import load_dict_from_hdf5

epi_range = [0]

curr_dir = os.path.dirname(os.path.abspath(__file__))
#data_dir = f'{curr_dir}/../../data/sapien_demo/'
data_dir = "/home/hz2999/gendp/data/cube_picking_hdf5_Apr16"
obs_keys = [
            'camera_0_color',
            'camera_0_depth',
            ]
obs_format = [
            'bgr',
            'uint16depth',
            ]

def vis_format(img, format):
    # img: (H, W, C) or (H, W) numpy array
    # format: 'bgr', 'rgb' or 'uint16depth' or 'float32depth'
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

for i in epi_range:
    if i==1:
        break
    print(f'visualizing episode {i}')
    data_path = f'{data_dir}/episode_{i}.hdf5'

    data_dict, _ = load_dict_from_hdf5(data_path)

    obs_ls = []
    print('all obs keys:', list(data_dict['observations']['images'].keys()))
    for obs_key in obs_keys:
        obs_ls.append(data_dict['observations']['images'][obs_key])

    T = obs_ls[0].shape[0]

    for obs_i, obs in enumerate(obs_ls):
        for t in range(T):
            img = vis_format(obs[t], obs_format[obs_i])
            # cv2.imshow('img', img)
            # cv2.waitKey(1)
            save_dir = f"./vis_output/episode_{i}/obs_{obs_keys[obs_i]}"
            os.makedirs(save_dir, exist_ok=True)
            cv2.imwrite(f"{save_dir}/frame_{t:04d}.png", img)

