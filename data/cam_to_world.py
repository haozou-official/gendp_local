import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

i = 0
rvec = np.load("cube_picking_processed/episode_0000/calibration/rvecs.npy")[i]  # for camera i
tvec = np.load("cube_picking_processed/episode_0000/calibration/tvecs.npy")[i]

R_wc, _ = cv2.Rodrigues(rvec)  # Rotation from world to camera
t_wc = tvec.reshape(3)

# Invert to get camera pose in world frame
R_cw = R_wc.T
t_cw = -R_wc.T @ t_wc

print(R_cw)
print(t_cw)

from transforms3d.quaternions import mat2quat
print(mat2quat(R_cw))

r = R.from_matrix(R_cw)
rpy = r.as_euler('xyz', degrees=False)  # returns [roll, pitch, yaw]
print(rpy)
