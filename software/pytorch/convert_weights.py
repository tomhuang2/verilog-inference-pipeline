import torch
import os 
from model import MLP
import numpy as np

os.makedirs("software/numpy/exported_weights",exist_ok = True)

model = MLP()
model.load_state_dict(torch.load("software/pytorch/saved_weights/MLP_weights.pth"))
model.eval()

for name, param in model.named_parameters():
    weight = param.detach().numpy()
    print(f"Saving {name}, with shape {weight.shape}")
    np.save(f"software/numpy/exported_weights/{name}.npy", weight)


