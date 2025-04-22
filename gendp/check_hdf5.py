import h5py

path = "/home/hz2999/gendp/data/sapien_demo/episode_0.hdf5"

with h5py.File(path, "r") as f:
    def print_structure(name, obj):
        print(f"{name} - {'Group' if isinstance(obj, h5py.Group) else 'Dataset'}")
    f.visititems(print_structure)

