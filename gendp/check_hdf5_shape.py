import h5py
import numpy as np
from pprint import pprint

episode_path = "/home/hz2999/gendp/data/real_aloha_demo/knife_real/episode_0.hdf5"

with h5py.File(episode_path, "r") as f:
    def show(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"{name}: shape={obj.shape}, dtype={obj.dtype}")
    
    print("🎯 Structure and Shapes:")
    f.visititems(show)

    # Preview actual data (optional):
    print("\n📷 Sample RGB shape and value (camera_0):")
    rgb = f["observations/images/camera_0_color"][0]
    print("Shape:", rgb.shape)
    print("Pixel [0,0]:", rgb[0, 0])

    print("\n🤖 EE Pose sample:")
    ee_pos = f["observations/ee_pos"][0]
    print(ee_pos)

    print("\n🎮 Cartesian Action sample:")
    cart_act = f["cartesian_action"][0]
    print(cart_act)

