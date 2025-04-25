import os
import torch
import torch.nn as nn
import models.face_recognition.adaface.net as net


class AdaFace(nn.Module):
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

            self.model = net.build_model("ir_101")
            self.model.to(device)
            current_dir = os.path.dirname(__file__)
            checkpoint_path = os.path.join(current_dir, "checkpoint", "adaface_ir101_webface12m.ckpt")

            statedict = torch.load(checkpoint_path, map_location=device, weights_only=False)["state_dict"]
            model_statedict = {key[6:]: val for key, val in statedict.items() if key.startswith("model.")}
            self.model.load_state_dict(model_statedict)
            self.model.eval()

    def forward(self, x, enable_grad=False):
        with torch.set_grad_enabled(enable_grad):
            # Swap RGB → BGR
            x = x[:, [2, 1, 0], :, :]
            features, _ = self.model(x)
            return features
