import mlflow
import torch
import sys

# Patch torch.load
original_load = torch.load
def patched_load(*args, **kwargs):
    kwargs['map_location'] = 'cpu'
    return original_load(*args, **kwargs)
torch.load = patched_load

try:
    model = mlflow.pytorch.load_model("models:/mlp_basic_model/latest")
    print("Success!")
except Exception as e:
    print(f"Error: {e}")
