import torch
import torch.nn as nn
import torch.nn.init as init
from opencood.quant.quant_block import specials_unquantized_names
try: # spconv1
    from spconv import SparseSequential, SubMConv3d, SparseConv3d, SparseInverseConv3d, SparseConvTensor
except: # spconv2
    from spconv.pytorch import  SparseSequential, SubMConv3d, SparseConv3d, SparseInverseConv3d, SparseConvTensor


class StraightThrough(nn.Module):
    def __int__(self):
        super().__init__()

    def forward(self, input):
        return input


def _fold_bn(conv_module, bn_module):
    """
    BatchNorm folding for Conv2d, ConvTranspose2d, and Linear layers.
    """
    w = conv_module.weight.data
    y_mean = bn_module.running_mean
    y_var = bn_module.running_var
    safe_std = torch.sqrt(y_var + bn_module.eps)

    if isinstance(conv_module, nn.Conv2d):
        return _fold_bn_conv(conv_module, bn_module, w, y_mean, safe_std)
    elif isinstance(conv_module, nn.ConvTranspose2d):
        return _fold_bn_transpose(conv_module, bn_module, w, y_mean, safe_std)
    elif isinstance(conv_module, nn.Linear):
        return _fold_bn_linear(conv_module, bn_module, w, y_mean, safe_std)
    elif isinstance(conv_module, (SubMConv3d, SparseConv3d, SparseInverseConv3d)):
        return _fold_bn_spconv(conv_module, bn_module, w, y_mean, safe_std)
    else:
        raise TypeError(f"Unsupported module type {type(conv_module)} in BN folding")


def _fold_bn_conv(conv_module, bn_module, w, y_mean, safe_std):
    """
    BatchNorm folding for standard Conv2d layers.
    """
    w_view = (conv_module.out_channels, 1, 1, 1)  # Expand to match shape
    if bn_module.affine:
        weight = w * (bn_module.weight / safe_std).view(w_view)
        beta = bn_module.bias - bn_module.weight * y_mean / safe_std
        if conv_module.bias is not None:
            bias = bn_module.weight * conv_module.bias / safe_std + beta
        else:
            bias = beta
    else:
        weight = w / safe_std.view(w_view)
        beta = -y_mean / safe_std
        if conv_module.bias is not None:
            bias = conv_module.bias / safe_std + beta
        else:
            bias = beta
    return weight, bias


def _fold_bn_transpose(conv_module, bn_module, w, y_mean, safe_std):
    """
    BatchNorm folding for ConvTranspose2d layers.
    """
    w_view = (1, conv_module.out_channels, 1, 1)  # Adjusted for transposed convolution
    if bn_module.affine:
        weight = w * (bn_module.weight / safe_std).view(w_view)
        beta = bn_module.bias - bn_module.weight * y_mean / safe_std
        if conv_module.bias is not None:
            bias = bn_module.weight * conv_module.bias / safe_std + beta
        else:
            bias = beta
    else:
        weight = w / safe_std.view(w_view)
        beta = -y_mean / safe_std
        if conv_module.bias is not None:
            bias = conv_module.bias / safe_std + beta
        else:
            bias = beta
    return weight, bias


def _fold_bn_linear(linear_module, bn_module, w, y_mean, safe_std):
    """
    BatchNorm folding for Linear layers.
    """
    w_view = (linear_module.out_features, 1)  # Fully connected layers reshape
    if bn_module.affine:
        weight = w * (bn_module.weight / safe_std).view(w_view)
        beta = bn_module.bias - bn_module.weight * y_mean / safe_std
        if linear_module.bias is not None:
            bias = bn_module.weight * linear_module.bias / safe_std + beta
        else:
            bias = beta
    else:
        weight = w / safe_std.view(w_view)
        beta = -y_mean / safe_std
        if linear_module.bias is not None:
            bias = linear_module.bias / safe_std + beta
        else:
            bias = beta
    return weight, bias


def _fold_bn_spconv(conv_module, bn_module, w, y_mean, safe_std):
    """
    BatchNorm folding for spconv layers like SubMConv3d, SparseConv3d, etc.
    """
    w_view = (conv_module.out_channels, 1, 1, 1, 1)  # Shape for 3D conv weights

    if bn_module.affine:
        weight = w * (bn_module.weight / safe_std).view(w_view)
        beta = bn_module.bias - bn_module.weight * y_mean / safe_std
        if conv_module.bias is not None:
            bias = bn_module.weight * conv_module.bias / safe_std + beta
        else:
            bias = beta
    else:
        weight = w / safe_std.view(w_view)
        beta = -y_mean / safe_std
        if conv_module.bias is not None:
            bias = conv_module.bias / safe_std + beta
        else:
            bias = beta
    return weight, bias


def fold_bn_into_conv(conv_module, bn_module):
    w, b = _fold_bn(conv_module, bn_module)
    if conv_module.bias is None:
        conv_module.bias = nn.Parameter(b)
    else:
        conv_module.bias.data = b
    conv_module.weight.data = w
    # set bn running stats
    bn_module.running_mean = bn_module.bias.data
    bn_module.running_var = bn_module.weight.data ** 2


def reset_bn(module: nn.BatchNorm2d):
    if module.track_running_stats:
        module.running_mean.zero_()
        module.running_var.fill_(1-module.eps)
        # we do not reset numer of tracked batches here
        # self.num_batches_tracked.zero_()
    if module.affine:
        init.ones_(module.weight)
        init.zeros_(module.bias)


def is_bn(m):
    return isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d)


def is_absorbing(m):
    return isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d) \
              or isinstance(m, SubMConv3d) or isinstance(m, SparseConv3d) or isinstance(m, SparseInverseConv3d)


def search_fold_and_remove_bn(model):
    model.eval()
    prev = None
    for n, m in model.named_children():
        if n in specials_unquantized_names:
            continue
        if is_bn(m) and is_absorbing(prev): # if the previous layer is conv or linear, and the current layer is bn1d or bn2d
            fold_bn_into_conv(prev, m)
            # set the bn module to straight through
            setattr(model, n, StraightThrough())
        elif is_absorbing(m):
            prev = m
        else:
            prev = search_fold_and_remove_bn(m)
    return prev


def search_fold_and_reset_bn(model):
    model.eval()
    prev = None
    for n, m in model.named_children():
        if is_bn(m) and is_absorbing(prev):
            fold_bn_into_conv(prev, m)
            # reset_bn(m)
        else:
            search_fold_and_reset_bn(m)
        prev = m

