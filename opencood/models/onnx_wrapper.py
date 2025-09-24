import torch
import torch.nn as nn

class OnnxWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.eval()
        torch.set_grad_enabled(False)

    def forward_normal(self, data_dict):
        return self.model(data_dict)

    def forward(self, *args):
        if hasattr(self.model, 'forward_onnx_export'):
            try:
                return self.model.forward_onnx_export(*args)
            except TypeError:
                return self.model.forward_onnx_export(args)
        else:
            return self.model(*args)