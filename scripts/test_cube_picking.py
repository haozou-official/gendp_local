from sapien_env.rl_env.cube_picking_env import CubePickingEnv
from diffusion_policy.workspace.base_workspace import BaseWorkspace
import torch

def run_policy():
    # Load the environment
    env = CubePickingEnv(use_gui=True)

    # Reset the env
    obs = env.reset()
    
    # Load the trained GenDP policy
    workspace = BaseWorkspace()
    workspace.load_ckpt('<path_to_your_checkpoint>.ckpt')
    policy = workspace.model
    policy.eval()
    
    # Rollout policy
    for _ in range(200):  # roll out 200 timesteps
        action = policy(obs)  # shape: (act_dim,)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()

if __name__ == '__main__':
    run_policy()
