import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

def inspect_depth_images(base_dir, camera='camera_0', show=False):
    depth_dir = os.path.join(base_dir, camera, 'depth')
    depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith('.png')])

    print(f"📂 Inspecting {len(depth_files)} depth images in: {depth_dir}")

    for i, f in enumerate(depth_files):
        path = os.path.join(depth_dir, f)
        depth_img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        depth_img = depth_img / 1000.0

        if depth_img is None:
            print(f"[Warning] Failed to load {path}")
            continue

        depth_img = depth_img.astype(np.float32)

        min_val = np.min(depth_img)
        max_val = np.max(depth_img)
        mean_val = np.mean(depth_img)
        nonzero_ratio = np.count_nonzero(depth_img) / depth_img.size

        print(f"[{f}] min: {min_val:.3f}, max: {max_val:.3f}, mean: {mean_val:.3f}, non-zero: {nonzero_ratio:.2%}")

        if show:
            plt.imshow(depth_img, cmap='gray')
            plt.title(f"{f} — min={min_val:.2f}, max={max_val:.2f}")
            plt.colorbar()
            plt.show()

        if i >= 10:  # Only inspect first 10 images unless changed
            break

if __name__ == "__main__":
    base_dir = "/home/hz2999/gendp/data/cube_picking_processed/episode_0000"
    inspect_depth_images(base_dir, camera='camera_0', show=True)

