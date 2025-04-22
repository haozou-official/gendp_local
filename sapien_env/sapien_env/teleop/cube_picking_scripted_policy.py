import numpy as np
from pyquaternion import Quaternion
import transforms3d
import sapien.core as sapien

from sapien_env.rl_env.cube_picking_env import CubePickingRLEnv

class SingleArmPolicy:
    def __init__(self, inject_noise=False):
        self.inject_noise = inject_noise
        self.step_count = 0
        self.trajectory = None

    @staticmethod
    def interpolate(curr_waypoint, next_waypoint, t):
        t_frac = (t - curr_waypoint["t"]) / (next_waypoint["t"] - curr_waypoint["t"])
        curr_xyz = curr_waypoint['xyz']
        curr_quat = curr_waypoint['quat']
        curr_grip = curr_waypoint['gripper']
        next_xyz = next_waypoint['xyz']
        next_quat = next_waypoint['quat']
        next_grip = next_waypoint['gripper']
        xyz = curr_xyz + (next_xyz - curr_xyz) * t_frac
        # quat = curr_quat + (next_quat - curr_quat) * t_frac
        # interpolate quaternion using slerp
        curr_quat_obj = Quaternion(curr_quat)
        next_quat_obj = Quaternion(next_quat)
        quat = Quaternion.slerp(curr_quat_obj, next_quat_obj, t_frac).elements
        gripper = curr_grip + (next_grip - curr_grip) * t_frac
        return xyz, quat, gripper

    def single_trajectory(self,env, ee_link_pose, mode='straight'):
        # generate trajectory at first timestep, then open-loop execution
        if self.step_count == 0:
            self.generate_trajectory(env, ee_link_pose, mode)


        if self.trajectory[0]['t'] == self.step_count:
            self.curr_waypoint = self.trajectory.pop(0)
        
        if len(self.trajectory) == 0:
            quit = True
            return None, quit
        else:
            quit = False

        next_waypoint = self.trajectory[0]

        # interpolate between waypoints to obtain current pose and gripper command
        xyz, quat, gripper = self.interpolate(self.curr_waypoint, next_waypoint, self.step_count)


        # Inject noise
        if self.inject_noise:
            scale = 0.01
            xyz = xyz + np.random.uniform(-scale, scale, xyz.shape)

        self.step_count += 1
        cartisen_action_dim =6
        grip_dim = 1
        cartisen_action = np.zeros(cartisen_action_dim+grip_dim)
        eluer = transforms3d.euler.quat2euler(quat,axes='sxyz')
        cartisen_action[0:3] = xyz
        cartisen_action[3:6] = eluer
        cartisen_action[6] = gripper
        return cartisen_action, quit
    
    def generate_trajectory(self, env : CubePickingRLEnv, ee_link_pose, mode='straight'):
        cube_pose = env.cube.get_pose()
        cube_pos = cube_pose.p

        # Define grasp orientation
        grasp_quat = transforms3d.euler.euler2quat(np.pi, 0, np.pi / 2, axes='sxyz')

        pre_grasp = cube_pos + np.array([0.0, 0.0, 0.1])
        lift = cube_pos + np.array([0.0, 0.0, 0.15])
        place = cube_pos + np.array([0.2, 0.0, 0.15])
        place_drop = cube_pos + np.array([0.2, 0.0, 0.025])

        self.trajectory = [
            {"t": 0, "xyz": ee_pose.p, "quat": ee_pose.q, "gripper": 0.08},
            {"t": 20, "xyz": pre_grasp, "quat": grasp_quat, "gripper": 0.08},
            {"t": 40, "xyz": cube_pos, "quat": grasp_quat, "gripper": 0.08},
            {"t": 60, "xyz": cube_pos, "quat": grasp_quat, "gripper": 0.0},  # close
            {"t": 80, "xyz": lift, "quat": grasp_quat, "gripper": 0.0},
            {"t": 100, "xyz": place, "quat": grasp_quat, "gripper": 0.0},
            {"t": 120, "xyz": place_drop, "quat": grasp_quat, "gripper": 0.0},
            {"t": 140, "xyz": place_drop, "quat": grasp_quat, "gripper": 0.08},  # release
            {"t": 160, "xyz": place, "quat": grasp_quat, "gripper": 0.08},
        ]