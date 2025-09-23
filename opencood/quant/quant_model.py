import torch.nn as nn
from opencood.quant.quant_block import specials, opencood_specials, specials_unquantized_names, BaseQuantBlock
from opencood.quant.quant_layer import QuantModule, StraightThrough, UniformAffineQuantizer
from .fold_bn import search_fold_and_remove_bn


class QuantModel(nn.Module):

    def __init__(self, model: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}, is_fusing=True):
        super().__init__()
        if is_fusing:
            search_fold_and_remove_bn(model)
            self.model = model
            self.quant_module_refactor(self.model, weight_quant_params, act_quant_params)
        else:
            self.model = model
            self.quant_module_refactor_wo_fuse(self.model, weight_quant_params, act_quant_params)

    def quant_module_refactor(self, module: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        """
        Recursively replace only `opencood_specials` modules with their quantized versions and fuse BatchNorm.
        :param module: nn.Module with submodules in `opencood_specials`
        :param weight_quant_params: quantization parameters for weights
        :param act_quant_params: quantization parameters for activations
        """
        prev_quantmodule = None  # Store last quantized module to attach BatchNorm
        for name, child_module in module.named_children():
            # Skip unquantized layers
            if name in specials_unquantized_names:
                continue

            # If the module is in `opencood_specials`, replace it with its quantized counterpart
            if type(child_module) in opencood_specials:
                setattr(module, name, opencood_specials[type(child_module)](child_module, weight_quant_params, act_quant_params))

            elif isinstance(child_module, (nn.Conv2d, nn.Linear)):
                setattr(module, name, QuantModule(child_module, weight_quant_params, act_quant_params))
                prev_quantmodule = getattr(module, name)

            elif isinstance(child_module, (nn.ReLU, nn.ReLU6)):
                if prev_quantmodule is not None:
                    prev_quantmodule.activation_function = child_module
                    setattr(module, name, StraightThrough())
                else:
                    continue

            elif isinstance(child_module, StraightThrough):
                continue

            else:
                self.quant_module_refactor(child_module, weight_quant_params, act_quant_params)


    def quant_module_refactor_wo_fuse(self, module: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        """
        Recursively replace only `opencood_specials` modules with their quantized versions but leave BatchNorm unchanged.
        :param module: nn.Module with submodules in `opencood_specials`
        :param weight_quant_params: quantization parameters for weights
        :param act_quant_params: quantization parameters for activations
        """
        prev_quantmodule = None  # Store last quantized module to attach BatchNorm
        for name, child_module in module.named_children():

            if name in specials_unquantized_names:
                continue

            if type(child_module) in opencood_specials:
                setattr(module, name, opencood_specials[type(child_module)](child_module, weight_quant_params, act_quant_params))

            elif isinstance(child_module, (nn.Conv2d, nn.Linear)):
                setattr(module, name, QuantModule(child_module, weight_quant_params, act_quant_params))
                prev_quantmodule = getattr(module, name)

            elif isinstance(child_module, nn.BatchNorm2d):
                if prev_quantmodule is not None:
                    prev_quantmodule.norm_function = child_module
                    setattr(module, name, StraightThrough())
                else:
                    continue
            
            elif isinstance(child_module, (nn.ReLU, nn.ReLU6)):
                if prev_quantmodule is not None:
                    prev_quantmodule.activation_function = child_module
                    setattr(module, name, StraightThrough())
                else:
                    continue

            elif isinstance(child_module, StraightThrough):
                continue

            else:
                self.quant_module_refactor_wo_fuse(child_module, weight_quant_params, act_quant_params)


    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        for m in self.model.modules():
            if isinstance(m, (QuantModule, BaseQuantBlock)):
                m.set_quant_state(weight_quant, act_quant)

    def forward(self, input):
        return self.model(input)

    def set_first_last_layer_to_8bit(self):
        w_list, a_list = [], []
        for module in self.model.modules():
            if isinstance(module, UniformAffineQuantizer):
                if module.leaf_param:
                    a_list.append(module)
                else:
                    w_list.append(module)
        w_list[0].bitwidth_refactor(8)
        w_list[-1].bitwidth_refactor(8)
        'the image input has been in 0~255, set the last layer\'s input to 8-bit'
        a_list[-2].bitwidth_refactor(8)
        # a_list[0].bitwidth_refactor(8)

    def disable_network_output_quantization(self):
        module_list = []
        for m in self.model.modules():
            if isinstance(m, QuantModule):
                module_list += [m]
        if len(module_list) == 3: 
            module_list[-1].disable_act_quant = True
            module_list[-2].disable_act_quant = True
            module_list[-3].disable_act_quant = True # for the last 3 detection heads

    def get_memory_footprint(self):
            """Calculate the total memory footprint of the model's parameters and buffers."""
            total_size = 0
            for param in self.parameters():
                total_size += param.nelement() * param.element_size()
            for buffer in self.buffers():
                total_size += buffer.nelement() * buffer.element_size()

            total_size_MB = total_size / (1024 ** 2)  # Convert to MB
            return f"Model Memory Footprint: {total_size_MB:.2f} MB"