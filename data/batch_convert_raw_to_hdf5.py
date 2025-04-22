import os
import argparse
from convert_raw_to_hdf5 import create_hdf5_from_episode

def batch_convert(root_input, root_output, episode_range=(0, 15)):
    os.makedirs(root_output, exist_ok=True)

    for i in range(*episode_range):
        ep_name = f"episode_{i:04d}"
        input_dir = os.path.join(root_input, ep_name)
        output_path = os.path.join(root_output, f"episode_{i}.hdf5")

        if not os.path.exists(input_dir):
            print(f"Skipping {ep_name}: input folder not found.")
            continue

        print(f"Converting {ep_name}...")
        try:
            create_hdf5_from_episode(input_dir, output_path)
            print(f"Saved to {output_path}\n")
        except Exception as e:
            print(f"Failed to convert {ep_name}: {e}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", help="Path to root folder of raw episodes")
    parser.add_argument("--output_root", help="Path to save HDF5 outputs")
    args = parser.parse_args()

    batch_convert(args.input_root, args.output_root)

