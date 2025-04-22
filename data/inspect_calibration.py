import pickle
from pprint import pprint  # prettier print for nested dicts

path = "/home/hz2999/gendp/data/cube_picking_processed/episode_0000/calibration/base.pkl"

with open(path, 'rb') as f:
    base = pickle.load(f)

print("📦 Content of base.pkl:")
pprint(base)