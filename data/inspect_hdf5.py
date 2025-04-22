import h5py
import numpy as np

#hdf5_path = "cube_picking_hdf5_Mar28/episode_0.hdf5" 
hdf5_path = "real_aloha_demo/knife_real/episode_0.hdf5" 

def print_dataset_info(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"[{name}] shape: {obj.shape}, dtype: {obj.dtype}")
        data = obj[()]
        if np.issubdtype(data.dtype, np.number):
            print(f"  └── min: {np.min(data)}, max: {np.max(data)}, mean: {np.mean(data):.3f}")

with h5py.File(hdf5_path, "r") as f:
    print(f"📂 Inspecting {hdf5_path}...\n")
    print("📂 HDF5 File Structure:\n")
    f.visititems(print_dataset_info)

