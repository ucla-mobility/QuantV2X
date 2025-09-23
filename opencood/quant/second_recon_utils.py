import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

from opencood.tools import train_utils
from .quant_layer import QuantModule, Union, lp_loss
from .quant_model import QuantModel
from .quant_block import BaseQuantBlock
from tqdm import trange
from icecream import ic

try: # spconv1
    from spconv import SparseSequential, SubMConv3d, SparseConv3d, SparseInverseConv3d, SparseConvTensor
except: # spconv2
    from spconv.pytorch import  SparseSequential, SubMConv3d, SparseConv3d, SparseInverseConv3d, SparseConvTensor


def stack_tensors_channelwise(tensor_list):
    """
        Stack tensors channel-wise. All tensors must have the same spatial dimensions.
        :param tensor_list: list of tensors (C, N, H, W)
        :return: stacked tensor
    """
    n, h, w = tensor_list[0].shape[1:]
    for t in tensor_list:
        assert t.shape[1:] == (n, h, w), f"Tensor shape {t.shape} mismatches expected {(n, h, w)}"
    
    return torch.cat(tensor_list, dim=0)

def get_encoder_out(model: QuantModel, layer: Union[QuantModule, BaseQuantBlock], cali_data: list,
                    batch_size: int = 32, keep_gpu: bool = True,
                    input_prob: bool = False, lamb=50, bn_lr=1e-3):
    """Activation after correction"""
    device = torch.device('cuda')
    get_inp_out = GetDcFpLayerInpOut(model, layer, device=device, input_prob=input_prob, lamb=lamb, bn_lr=bn_lr)
    cached_batches = []

    print("Start correcting {} batches of data!".format(len(cali_data)))
    for i in range(len(cali_data)):
        if input_prob:
            cur_out, out_fp, cur_sym = get_inp_out(cali_data[i])
            cached_batches.append((cur_out.cpu(), out_fp.cpu(), cur_sym))
        else:
            cur_out, out_fp = get_inp_out(cali_data[i])
            cached_batches.append((cur_out.cpu(), out_fp.cpu()))
    cached_outs = list(x[0] for x in cached_batches)
    cached_outputs = torch.cat([x[1] for x in cached_batches])
    if input_prob:
        cached_sym = list(x[2] for x in cached_batches) 
    torch.cuda.empty_cache()
    if keep_gpu:
        cached_outs = train_utils.to_device(cached_outs, device)
        cached_outputs = cached_outputs.to(device)
        if input_prob:
            cached_sym = train_utils.to_device(cached_sym, device)
    # if input_prob:
    #     cached_outs.requires_grad = False
    #     cached_sym.requires_grad = False
        return cached_outs, cached_outputs, cached_sym
    return cached_outs, cached_outputs


def get_encoder_input(model: QuantModel, block: Union[QuantModule, BaseQuantBlock], cali_data: list,
                      batch_size: int = 1, keep_gpu: bool = True,
                      input_prob: bool = False):
    """
    Save input data and output data of a particular layer/block over calibration dataset.

    :param model: QuantModel
    :param layer: QuantModule or QuantBlock
    :param cali_data: calibration data set (list of dict)
    :param weight_quant: use weight_quant quantization
    :param act_quant: use act_quant quantization
    :param batch_size: mini-batch size for calibration
    :param keep_gpu: put saved data on GPU for faster optimization
    :return: input and output data
    """
    device = next(model.parameters()).device
    get_inp_out = GetLayerInpOut(model, block, device=device, input_prob=input_prob)
    cached_batches = []
    is_dict = False
    channel_sizes = []

    for i in range(len(cali_data)):
        cur_inp = get_inp_out(cali_data[i])
        assert type(cur_inp) in [SparseConvTensor]
        cached_batches.append(cur_inp)

    torch.cuda.empty_cache()
    return cached_batches


class StopForwardException(Exception):
    """
    Used to throw and catch an exception to stop traversing the graph
    """
    pass


class DataSaverHook:
    """
    Forward hook that stores the input and output of a block
    """

    def __init__(self, store_input=False, store_output=False, stop_forward=False):
        self.store_input = store_input
        self.store_output = store_output
        self.stop_forward = stop_forward

        self.input_store = None
        self.output_store = None

    def __call__(self, module, input_batch, output_batch):
        if self.store_input:
            self.input_store = input_batch
        if self.store_output:
            self.output_store = output_batch
        if self.stop_forward:
            raise StopForwardException


class input_hook(object):
    """
	Forward_hook used to get the output of the intermediate layer. 
	"""
    def __init__(self, stop_forward=False):
        super(input_hook, self).__init__()
        self.inputs = None

    def hook(self, module, input, output):
        self.inputs = input

    def clear(self):
        self.inputs = None

class GetLayerInpOut:
    def __init__(self, model: QuantModel, layer: Union[QuantModule, BaseQuantBlock],
                 device: torch.device, input_prob: bool = False):
        self.model = model
        self.layer = layer
        self.device = device
        self.data_saver = DataSaverHook(store_input=True, store_output=False, stop_forward=True)
        self.input_prob = input_prob

    def __call__(self, model_input):

        handle = self.layer.register_forward_hook(self.data_saver)
        with torch.no_grad():
            self.model.set_quant_state(weight_quant=True, act_quant=True)
            try:
                model_input = train_utils.to_device(model_input, self.device)
                _ = self.model(model_input)
            except StopForwardException:
                pass

        handle.remove()
        return self.data_saver.input_store[0] # this is the input spconv tensor to SECOND

class GetDcFpLayerInpOut:
    def __init__(self, model: QuantModel, layer: Union[QuantModule, BaseQuantBlock],
                 device: torch.device, input_prob: bool = False, lamb=50, bn_lr=1e-3):
        self.model = model
        self.layer = layer
        self.device = device
        self.data_saver = DataSaverHook(store_input=True, store_output=True, stop_forward=False)
        self.input_prob = input_prob
        self.bn_stats = []
        self.eps = 1e-6
        self.lamb=lamb
        self.bn_lr=bn_lr
        for n, m in self.layer.named_modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            # get the statistics in the BatchNorm layers
                self.bn_stats.append(
                    (m.running_mean.detach().clone().flatten().cuda(),
                    torch.sqrt(m.running_var +
                                self.eps).detach().clone().flatten().cuda()))
    
    def own_loss(self, A, B):
        return (A - B).norm()**2 / B.size(0)
    
    def relative_loss(self, A, B):
        return (A-B).abs().mean()/A.abs().mean()

    def __call__(self, model_input):
        self.model.set_quant_state(False, False)
        handle = self.layer.register_forward_hook(self.data_saver)
        hooks = []
        hook_handles = []
        for name, module in self.layer.named_modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                hook = input_hook()
                hooks.append(hook)
                hook_handles.append(module.register_forward_hook(hook.hook))
        assert len(hooks) == len(self.bn_stats)

        with torch.no_grad():
            try:
                model_input = train_utils.to_device(model_input, self.device)
                output_fp = self.model(model_input)
                output_fp = output_fp['preds_tensor']
            except StopForwardException:
                pass
            if self.input_prob:
                if(isinstance(self.data_saver.input_store[0], SparseConvTensor)):
                    input_sym = self.data_saver.input_store[0] # now the input is Spconv Tensor
                else: # tensor
                    raise NotImplementedError
            
        handle.remove()

        para_features = input_sym.features.clone().detach().requires_grad_(True)
        para_features = train_utils.to_device(para_features, self.device)
        optimizer = optim.Adam([para_features], lr=self.bn_lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                        min_lr=1e-5,
                                                        verbose=False,
                                                        patience=100)
        iters=500
        for iter in range(iters):
            self.layer.zero_grad()
            optimizer.zero_grad()
            for hook in hooks:
                hook.clear()

            # Replace the features inside SparseConvTensor
            modified_input = input_sym.replace_feature(para_features)

            _ = self.layer(modified_input)
            mean_loss, std_loss = 0, 0
            for bn_stat, hook in zip(self.bn_stats, hooks):
                tmp_input = hook.inputs[0]  # (B, C, H, W) or similar
                bn_mean, bn_std = bn_stat # shape [16]

                # Transpose to [C, N] so we can compute per-channel stats
                tmp_input_t = tmp_input.transpose(0, 1)  # [16, 32000]
                tmp_mean = tmp_input_t.mean(dim=1)  # [16]
                tmp_var = tmp_input_t.var(dim=1, unbiased=False)  # [16]
                tmp_var = torch.clamp(tmp_var, min=1e-6)
                tmp_std = torch.sqrt(tmp_var)  # [16]

                mean_loss += self.own_loss(bn_mean, tmp_mean)
                std_loss += self.own_loss(bn_std, tmp_std)

            # IMPORTANT: constraint_loss on the features only
            constraint_loss = lp_loss(para_features, input_sym.features) / self.lamb
            total_loss = mean_loss + std_loss + constraint_loss
            total_loss.backward()
            optimizer.step()
            scheduler.step(total_loss.item())
            if (iter+1) % 500 == 0:
                print('Total loss:\t{:.3f} (mse:{:.3f}, mean:{:.3f}, std:{:.3f})\tcount={}'.format(
                float(total_loss), float(constraint_loss), float(mean_loss), float(std_loss), iter))
                
        with torch.no_grad():
            modified_input = input_sym.replace_feature(para_features)
            out_fp = self.layer(input_sym).dense().detach()

        if self.input_prob:
            return out_fp, output_fp.detach(), modified_input # tensor, tensor, spconv tensor
        return out_fp, output_fp.detach()