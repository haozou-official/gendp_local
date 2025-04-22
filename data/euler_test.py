import numpy as np
from scipy.spatial.transform import Rotation as R

rot = np.array([
    [ 0.9952047, -0.01143581,  0.09714369],
    [ 0.01322502, 0.9997542,  -0.01779431],
    [-0.09691632, 0.0189937,  0.9951113]
])

euler_xyz = R.from_matrix(rot).as_euler('xyz', degrees=False)
euler_zyx = R.from_matrix(rot).as_euler('zyx', degrees=False)

print("Euler xyz:", euler_xyz.round(6))
print("Euler zyx:", euler_zyx.round(6))

