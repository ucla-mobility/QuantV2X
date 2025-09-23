import torch
from .quant_layer import QuantModule
from .quant_block import BaseQuantBlock
from .quant_model import QuantModel
from typing import Union

def set_act_quantize_params(module: Union[QuantModel, QuantModule, BaseQuantBlock],
                            cached_inps: Union[list, torch.Tensor], channel_sizes: list=None, batch_size: int = 8):
    module.set_quant_state(True, True)

    for t in module.modules():
        if isinstance(t, (QuantModule, BaseQuantBlock)) and hasattr(t, 'act_quantizer'):
            t.act_quantizer.set_inited(False) # so that the quantizer will be re-inited

    '''set or init step size and zero point in the activation quantizer'''
    if isinstance(cached_inps, torch.Tensor):
        with torch.no_grad():
            for i in range(cached_inps.size(0)):
                module(cached_inps[i].cuda())
    else:
        for i in range(0, min(len(cached_inps), batch_size)):
            with torch.no_grad():
                module(cached_inps[i].cuda())
        
    torch.cuda.empty_cache()

    for t in module.modules():
        if isinstance(t, (QuantModule, BaseQuantBlock)) and \
                hasattr(t, 'act_quantizer'):
            t.act_quantizer.set_inited(True)
