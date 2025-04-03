# Read checkpoint.npy on output folder

import numpy as np

# Load checkpoint data
checkpoint_path = 'output_no_face/checkpoint_2.npy'

processed_data = np.load(checkpoint_path, allow_pickle=True).tolist()

print(f"Loaded checkpoint with {len(processed_data)} frames.")
print(processed_data[1])

# Output:
# Loaded checkpoint with 100 frames.
