import numpy as np
import cv2
import os

def generate_extrinsics(calib_dir):
    rvecs = np.load(os.path.join(calib_dir, 'rvecs.npy'))  # (4, 3)
    tvecs = np.load(os.path.join(calib_dir, 'tvecs.npy'))  # (4, 3)

    extrinsics = np.zeros((4, 4, 4), dtype=np.float32)

    for i in range(4):
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = cv2.Rodrigues(rvecs[i])[0]
        T[:3, 3] = tvecs[i].reshape(3) 
        extrinsics[i] = T 
        #print(T.shape)
    #print(extrinsics.shape)

    save_path = os.path.join(calib_dir, 'extrinsics.npy')
    np.save(save_path, extrinsics)
    print(f"Saved extrinsics to: {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--calib_dir", help="Path to calibration/ folder (containing rvecs.npy and tvecs.npy)")
    args = parser.parse_args()

    generate_extrinsics(args.calib_dir)
