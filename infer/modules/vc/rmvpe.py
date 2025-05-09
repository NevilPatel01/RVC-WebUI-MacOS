import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os


class RMVPE(nn.Module):
    def __init__(self, model_path, device="cpu"):
        super().__init__()
        self.device = device
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        try:
            self.model = torch.jit.load(
                model_path, map_location=device, weights_only=False
            )
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")

    def forward(self, x):
        with torch.no_grad():
            x = x.to(self.device)
            return self.model(x)

    @staticmethod
    def convert_to_onnx(model_path, onnx_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model = RMVPE(model_path, device="cpu")
        dummy_input = torch.randn(1, 1, 128, 128)

        torch.onnx.export(
            model.model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )
