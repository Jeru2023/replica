import torch
import time

# You can verify that PyTorch will utilize the GPU (if present) as follows:
# check for gpu
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print(x)
else:
    print("MPS device not found.")

# Run the next script in a virtual environment with and without GPU support to measure the performance:
# GPU
start_time = time.time()

# synchronize time with cpu, otherwise only time for loading data to gpu would be measured
torch.mps.synchronize()

a = torch.ones(4000, 4000, device="mps")
for _ in range(200):
    a += a

elapsed_time = time.time() - start_time
print("GPU Time: ", elapsed_time)
