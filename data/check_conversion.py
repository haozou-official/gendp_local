import h5py
import numpy as np

path = './cube_picking/episode_0.hdf5'
with h5py.File(path, 'r') as f:
    print("\nStructure and Shapes:")
    def print_group(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"{name}: shape={obj.shape}, dtype={obj.dtype}")
    f.visititems(print_group)

# Check EE Position and Gripper Openness
with h5py.File('./cube_picking/episode_0.hdf5', 'r') as f:
    ee = f['observations/ee_pos'][0]
    print(f"EE Pose [pos + dummy orient + gripper]:\n{ee}")
