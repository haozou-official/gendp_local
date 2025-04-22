from pytorch3d.ops import sample_farthest_points
import torch

empty = torch.empty((1, 0, 3), device="cuda")
sample_farthest_points(empty, K=1)

