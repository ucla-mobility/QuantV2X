# -*- coding: utf-8 -*-
# Author: Zhaowei Li


import torch
import torch.nn as nn

class OnnxWrapper(nn.Module):
    def __init__(self, model):
        super(OnnxWrapper, self).__init__()
        self.model = model
        self.model.eval()
        torch.set_grad_enabled(False)  # Disable gradients for the entire instance
    
    def forward_normal(self, data_dict):
        output = self.model(data_dict)
        return output

    def forward(self, *args):
        output = self.model.forward_onnx_export(*args)
        return output