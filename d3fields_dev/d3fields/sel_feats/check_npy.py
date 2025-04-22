import numpy as np

# Load the file
data = np.load("/home/hz2999/gendp/d3fields_dev/d3fields/sel_feats/mug.npy")

# Check shape and dtype
print("Shape:", data.shape)
print("Dtype:", data.dtype)

# Preview the first few entries
print("First row (feature vector):\n", data[0])
