import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

from opencood.tools import train_utils
from .quant_layer import QuantModule, Union, lp_loss
from .quant_model import QuantModel
from .quant_block import BaseQuantBlock
from tqdm import trange

def get_pyramid_out(model: QuantModel, layer: Union[QuantModule, BaseQuantBlock], 
                    cali_data: list, keep_gpu: bool = True,
                    input_prob: bool = False, lamb=50, bn_lr=1e-3):
    """Activation after correction"""
    device = torch.device('cuda')
    get_inp_out = get_fp_inpout(model, layer, device=device, input_prob=input_prob, lamb=lamb, bn_lr=bn_lr)
    cached_batches = []

    print("Start correcting {} batches of data!".format(len(cali_data)))
    for i in range(len(cali_data)):
        if input_prob:
            cur_out, out_fp, cur_sym = get_inp_out(cali_data[i])
            cached_batches.append((cur_out.cpu(), out_fp.cpu(), cur_sym.cpu()))
        else:
            cur_out, out_fp = get_inp_out(cali_data[i])
            cached_batches.append((cur_out.cpu(), out_fp.cpu()))
    cached_outs = torch.cat([x[0] for x in cached_batches])
    cached_outputs = torch.cat([x[1] for x in cached_batches])
    if input_prob:
        cached_sym = torch.cat([x[2] for x in cached_batches])
    torch.cuda.empty_cache()
    if keep_gpu:
        cached_outs = cached_outs.to(device)
        cached_outputs = cached_outputs.to(device)
        if input_prob:
            cached_sym = cached_sym.to(device)
    if input_prob:
        cached_outs.requires_grad = False
        cached_sym.requires_grad = False
        return cached_outs, cached_outputs, cached_sym
    return cached_outs, cached_outputs


def get_pyramid_input(model: QuantModel, 
                      block: Union[QuantModule, BaseQuantBlock], 
                      cali_data: list,
                      keep_gpu: bool = True,
                      input_prob: bool = False):
    
    device = next(model.parameters()).device
    get_inp_out = GetBlockInpOut(model, block, device=device, input_prob=input_prob)
    cached_b1 = []
    cached_b2 = []
    cached_b3 = []
    cached_b4 = []
    cached_b5 = []
    is_dict = False

    for i in range(len(cali_data)):
        inp1, inp2, inp3, inp4, inp5 = get_inp_out(cali_data[i])
        cached_b1.append(inp1.unsqueeze(0).cpu())
        cached_b2.append(inp2.unsqueeze(0).cpu())
        cached_b3.append(inp3.unsqueeze(0).cpu())
        cached_b4.append(inp4)
        cached_b5.append(inp5)

        cached_inp1 = torch.cat([x for x in cached_b1])
        cached_inp2 = torch.cat([x for x in cached_b2])
        cached_inp3 = torch.cat([x for x in cached_b3])
        torch.cuda.empty_cache()
        if keep_gpu:
            cached_inp1 = cached_inp1.to(device)
            cached_inp2 = cached_inp2.to(device)
            cached_inp3 = cached_inp3.to(device)
        return cached_inp1, cached_inp2, cached_inp3, cached_b4, cached_b5


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

class GetBlockInpOut:
    def __init__(self, model: QuantModel, block: Union[QuantModule, BaseQuantBlock],
                 device: torch.device, input_prob: bool = False):
        self.model = model
        self.block = block
        self.device = device
        self.data_saver = DataSaverHook(store_input=True, store_output=False, stop_forward=True)
        self.input_prob = input_prob

    def __call__(self, model_input):

        handle = self.block.register_forward_hook(self.data_saver)
        with torch.no_grad():
            self.model.set_quant_state(weight_quant=True, act_quant=True)
            try:
                model_input = train_utils.to_device(model_input, self.device)
                _ = self.model(model_input)
            except StopForwardException:
                pass

        handle.remove()

        inp1 = self.data_saver.input_store[0] # float32 (2, 64, 256, 256)
        inp2 = self.data_saver.input_store[1] # int64 (2)
        inp3 = self.data_saver.input_store[2] # float64 (1, 6, 6, 2, 3)
        inp4 = self.data_saver.input_store[3] # list of strings
        inp5 = self.data_saver.input_store[4] # None for LiDAR

        return inp1, inp2, inp3, inp4, inp5

class get_fp_inpout:
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
            if isinstance(m, nn.BatchNorm2d):
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
            if isinstance(module, nn.BatchNorm2d):
                hook = input_hook()
                hooks.append(hook)
                hook_handles.append(module.register_forward_hook(hook.hook))
        assert len(hooks) == len(self.bn_stats)

        with torch.no_grad():
            try:
                model_input = train_utils.to_device(model_input, self.device)
                output_fp = self.model(model_input)
                output_fp = output_fp['reg_preds']
            except StopForwardException:
                pass
            if self.input_prob:
                input_sym = self.data_saver.input_store[0].detach() # float32 (2, 64, 256, 256) heter_feature_2d
                inp2 = self.data_saver.input_store[1].detach()
                inp3 = self.data_saver.input_store[2].detach()
                inp4 = self.data_saver.input_store[3]
                inp5 = self.data_saver.input_store[4]
            
        handle.remove()
        para_input = input_sym.data.clone()
        para_input = train_utils.to_device(para_input, self.device)
        para_input.requires_grad = True
        optimizer = optim.Adam([para_input], lr=self.bn_lr)
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
            _, _ = self.layer(para_input, inp2, inp3, inp4, inp5)
            mean_loss = 0
            std_loss = 0
            for num, (bn_stat, hook) in enumerate(zip(self.bn_stats, hooks)):
                tmp_input = hook.inputs[0]
                bn_mean, bn_std = bn_stat[0], bn_stat[1]
                tmp_mean = torch.mean(tmp_input.view(tmp_input.size(0),
                                                    tmp_input.size(1), -1),
                                    dim=2)
                tmp_std = torch.sqrt(
                    torch.var(tmp_input.view(tmp_input.size(0),
                                            tmp_input.size(1), -1),
                            dim=2) + self.eps)
                mean_loss += self.own_loss(bn_mean, tmp_mean)
                std_loss += self.own_loss(bn_std, tmp_std)
            constraint_loss = lp_loss(para_input, input_sym) / self.lamb
            total_loss = mean_loss + std_loss + constraint_loss
            total_loss.backward()
            optimizer.step()
            scheduler.step(total_loss.item())
            if (iter+1) % 500 == 0:
                print('Total loss:\t{:.3f} (mse:{:.3f}, mean:{:.3f}, std:{:.3f})\tcount={}'.format(
                float(total_loss), float(constraint_loss), float(mean_loss), float(std_loss), iter))
                
        with torch.no_grad():
            fused_feature, occ_map_list = self.layer(para_input, inp2, inp3, inp4, inp5)
        
        fused_feature = fused_feature.unsqueeze(0)
        if self.input_prob:
            para_input = para_input.unsqueeze(0)

        if self.input_prob:
            return  fused_feature.detach(), output_fp.detach(), para_input.detach()
        return fused_feature.detach(), output_fp.detach()