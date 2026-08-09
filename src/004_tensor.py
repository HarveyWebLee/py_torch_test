import numpy as np
import torch

data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
print(f"x_data: {x_data}")

np_array = np.array(data)

x_np = torch.from_numpy(np_array)
print(f"x_np: {x_np}")

x_ones = torch.ones_like(x_data)  # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

# overrides the datatype of x_data
x_rand = torch.rand_like(x_data, dtype=torch.float)
print(f"Random Tensor: \n {x_rand} \n")
