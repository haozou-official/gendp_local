import numpy as np
from scipy.spatial.transform import Rotation as R

# Given: camera poses in SAPIEN world frame (position, quaternion [w, x, y, z])
camera_poses_world = {
    "right_top": {
        "position": np.array([-0.300793, -0.03741587, 0.2789142]) * 1.1,
        "rotation": np.array([0.96431008, -0.03549781, 0.17547506, 0.19507563])
    },
    "left_top": {
        "position": np.array([0.32175508, -0.09930836, 0.20163289]) * 1.1,
        "rotation": np.array([0.26004855, -0.19390239, 0.05339162, 0.94441833])
    },
    "right_bottom": {
        "position": np.array([-0.31482142, 0.2923913, 0.26412952]) * 1.1,
        "rotation": np.array([0.94703722, 0.04895124, 0.26465097, -0.17516899])
    },
    "left_bottom": {
        "position": np.array([0.3791156, 0.27693018, 0.270775]) * 1.1,
        "rotation": np.array([0.13047577, 0.20659657, 0.0278099, -0.96928869])
    }
}

# Given: Calibration world to SAPIEN world transform
R_base2world = np.array([
    [1., 0., 0.],
    [0., -1., 0.],
    [0., 0., -1.]
])
t_base2world = np.array([-0.095, 0.085, -0.01])  # in meters

# Compute inverse transformation: world to calibration
R_world2base = R_base2world.T
t_world2base = -R_world2base @ t_base2world

# Apply transform to each camera pose
transformed_cameras = {}
for name, cam in camera_poses_world.items():
    p_world = cam["position"]
    q_wxyz = cam["rotation"]  # [w, x, y, z]
    
    # Convert to rotation matrix
    q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])  # [x, y, z, w]
    R_cam_world = R.from_quat(q_xyzw).as_matrix()

    # Transform position
    p_base = R_world2base @ p_world + t_world2base

    # Transform orientation
    R_cam_base = R_world2base @ R_cam_world
    q_cam_base_xyzw = R.from_matrix(R_cam_base).as_quat()
    q_cam_base_wxyz = [q_cam_base_xyzw[3], q_cam_base_xyzw[0], q_cam_base_xyzw[1], q_cam_base_xyzw[2]]

    transformed_cameras[name] = {
        "position": p_base,
        "rotation": q_cam_base_wxyz
    }

import pandas as pd
import ace_tools as tools
tools.display_dataframe_to_user("Transformed Camera Poses in Calibration Frame", pd.DataFrame(transformed_cameras).T)
