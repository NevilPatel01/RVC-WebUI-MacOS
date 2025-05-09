import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os


def get_rmvpe(model_path, device, is_half=True):
    try:
        # Load the model using torch.load instead of torch.jit.load
        state_dict = torch.load(model_path, map_location=device, weights_only=False)
        model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
        )
        model.load_state_dict(state_dict)
        model.eval()
        if is_half:
            model = model.half()
        model = model.to(device)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load RMVPE model: {str(e)}")
