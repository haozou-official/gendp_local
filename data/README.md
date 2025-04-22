# Convert Raw Real World Data to HDF5

## 💾  Generate Dataset
### Generate Camera Extrinsics
```console
python generate_extrinsics.py --calib_dir [PATH_TO_CALIBRATION]
```
`[PATH_TO_CALIBRATION]`: Path to calibration/ folder (containing rvecs.npy and tvecs.npy).

### Convert Single Episode Raw File
```console
python convert_raw_to_hdf5.py --episode_dir cube_picking_processed/episode_0000 --output_path [OUTPUT_DIR]/episode_0.hdf5
```

### Convert Whole Episodes
```console
python batch_convert_raw_to_hdf5.py --input_root cube_picking_processed/ --output_root [OUTPUT_DIR]
```

### Code Explanation
`cube_picking_processed` is the raw dataset folder collected from real robot, which has `episode_0000` to `episode_N` subfolders, make sure each episode subfolder has:
- `calibration`: contains `base.pkl` to convert robot base to world frame; `intrinsics.npy`; `rvecs.npy` and `tvecs.npy` to generate `extrinsics.npy`.
- `camera_0` to `camera_N`, `0` to `6` cameras at most. Each camera folder contains `/depth` and `/rgb` folders for images.
- `robot`: `000000.txt` to `N_steps.txt` to record robot pose.
```console
1.978241334213923786e-01 1.320325309243938761e-01 -2.688415424187221014e-01
9.952046651029050617e-01 -1.143580994124750519e-02 9.714369155226811048e-02
1.322502194812277820e-02 9.997542005365072093e-01 -1.779430538349144331e-02
-9.691632139060073203e-02 1.899370318281683873e-02 9.951112831676248716e-01
8.010000000000000000e+02 0.000000000000000000e+00 0.000000000000000000e+00
```