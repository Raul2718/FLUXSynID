# Based on: Deng, Jiankang, et al. "Arcface: Additive angular margin loss for deep face recognition."

import os
import torch
import torch.nn as nn
from models.face_recognition.networks import Backbone


class ArcFace(nn.Module):
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            # Create the instance if it doesn't exist
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "initialized"):  # Prevent __init__ from running more than once
            super().__init__()
            self.initialized = True  # Mark the instance as initialized

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = Backbone(num_layers=50, mode="ir_se", drop_ratio=0.6, affine=True)
            self.model.to(device)

            current_dir = os.path.dirname(__file__)
            checkpoint_path = os.path.join(current_dir, "checkpoint", "ArcFace.pth")
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=False))

            self.model.eval()

    def forward(self, x, enable_grad=False):
        with torch.set_grad_enabled(enable_grad):
            return self.model(x)
