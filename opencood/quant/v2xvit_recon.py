
from typing import Union
import torch
import torch.nn.functional as F
from .quant_layer import QuantModule, lp_loss
from .quant_model import QuantModel
from .quant_block import BaseQuantBlock, specials_unquantized_names
from .adaptive_rounding import AdaRoundQuantizer
from .v2xvit_recon_utils import get_v2xvit_input, get_v2xvit_out

include = False
def find_unquantized_module(model: torch.nn.Module, module_list: list = [], name_list: list = []):
    """Store subsequent unquantized modules in a list"""
    global include
    for name, module in model.named_children():
        if isinstance(module, (QuantModule, BaseQuantBlock)):
            if not module.trained:
                include = True
                module.set_quant_state(False,False)
                name_list.append(name)
                module_list.append(module)
        elif include and type(module) in specials_unquantized_names:
            name_list.append(name)
            module_list.append(module)
        else:
            find_unquantized_module(module, module_list, name_list)
    return module_list[1:], name_list[1:]

def set_act_quantize_params(module: Union[QuantModel, QuantModule, BaseQuantBlock],
                            inp1: torch.Tensor, inp2: torch.Tensor, inp3: torch.Tensor):
    module.set_quant_state(True, True)

    for t in module.modules():
        if isinstance(t, (QuantModule, BaseQuantBlock)) and hasattr(t, 'act_quantizer'):
            t.act_quantizer.set_inited(False) # so that the quantizer will be re-inited

    '''set or init step size and zero point in the activation quantizer'''
    with torch.no_grad():
        for i in range(inp1.size(0)):
            module(inp1[i].cuda(), inp2[i].cuda(), inp3[i].cuda())
        
    torch.cuda.empty_cache()

    for t in module.modules():
        if isinstance(t, (QuantModule, BaseQuantBlock)) and \
                hasattr(t, 'act_quantizer'):
            t.act_quantizer.set_inited(True)

def v2xvit_reconstruction(qt_model: QuantModel, fp_model: QuantModel, qt_block: BaseQuantBlock, fp_block: BaseQuantBlock,
                        cali_data: list, iters: int = 20000, weight: float = 0.01, 
                        opt_mode: str = 'mse', b_range: tuple = (20, 2),
                        warmup: float = 0.0, p: float = 2.0, lr: float = 4e-5,
                        input_prob: float = 1.0, keep_gpu: bool = True, 
                        lamb_r: float = 0.2, T: float = 7.0, bn_lr: float = 1e-3, lamb_c=0.02):
    """
    Reconstruction to optimize the output from pyramid fusion

    :param model: QuantModel
    :param block: BaseQuantBlock that needs to be optimized
    :param cali_data: data for calibration, typically 1024 training images, as described in AdaRound
    :param batch_size: mini-batch size for reconstruction
    :param iters: optimization iterations for reconstruction,
    :param weight: the weight of rounding regularization term
    :param opt_mode: optimization mode
    :param asym: asymmetric optimization designed in AdaRound, use quant input to reconstruct fp output
    :param include_act_func: optimize the output after activation function
    :param b_range: temperature range
    :param warmup: proportion of iterations that no scheduling for temperature
    :param lr: learning rate for act delta learning
    :param p: L_p norm minimization
    :param lamb_r: hyper-parameter for regularization
    :param T: temperature coefficient for KL divergence
    :param bn_lr: learning rate for DC
    :param lamb_c: hyper-parameter for DC
    """

    '''
    PyramidFusion has 5 input in forward():
        1. heter_feature_2d: torch.Tensor
        2. record_len: torch.Tensor
        3. affine_matrix: torch.Tensor
    '''
    inp1, inp2, inp3 = get_v2xvit_input(qt_model, 
                                            qt_block, 
                                            cali_data, 
                                            input_prob=True, 
                                            keep_gpu=keep_gpu)
    
    cached_fused_feature, cached_output, cur_syms = get_v2xvit_out(fp_model, 
                                                            fp_block, 
                                                            cali_data, 
                                                            input_prob=True, 
                                                            keep_gpu=keep_gpu, 
                                                            bn_lr=bn_lr, 
                                                            lamb=lamb_c)
    
    set_act_quantize_params(qt_block, inp1, inp2, inp3)

    '''set state'''
    cur_weight, cur_act = True, True
    
    global include
    # module_list, name_list, include = [], [], False
    # module_list, name_list = find_unquantized_module(model, module_list, name_list) #TODO
    qt_block.set_quant_state(cur_weight, cur_act)
    for para in qt_model.parameters():
        para.requires_grad = False

    '''set quantizer'''
    round_mode = 'learned_hard_sigmoid'
    # Replace weight quantizer to AdaRoundQuantizer
    w_para, a_para = [], []
    w_opt, a_opt = None, None
    scheduler, a_scheduler = None, None

    for module in qt_block.modules():
        '''weight'''
        if isinstance(module, QuantModule):
            device = next(qt_model.parameters()).device  # Get the model's device
            module.weight_quantizer = AdaRoundQuantizer(uaq=module.weight_quantizer, round_mode=round_mode,
                                                        weight_tensor=module.org_weight.data.to(device))
            module.weight_quantizer.soft_targets = True
            w_para += [module.weight_quantizer.alpha]
        '''activation'''
        if isinstance(module, (QuantModule, BaseQuantBlock)):
            if not hasattr(module, 'act_quantizer'):
                continue # some BaseQuantBlock has no act_quantizer (e.g., QuantBaseBevBackbone)
            if module.act_quantizer.delta is not None:
                module.act_quantizer.delta = torch.nn.Parameter(torch.tensor(module.act_quantizer.delta))
                a_para += [module.act_quantizer.delta]
            '''set up drop'''
            module.act_quantizer.is_training = True

    if len(w_para) != 0:
        w_opt = torch.optim.Adam(w_para, lr=3e-3)
    if len(a_para) != 0:
        a_opt = torch.optim.Adam(a_para, lr=lr)
        a_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(a_opt, T_max=iters, eta_min=0.)

    loss_mode = 'relaxation'
    rec_loss = opt_mode # default is 'mse'
    loss_func = LossFunction(qt_block, round_loss=loss_mode, weight=weight, max_count=iters, rec_loss=rec_loss,
                             b_range=b_range, decay_start=0, warmup=warmup, p=p, lam=lamb_r, T=T)
    device = 'cuda'
    sz = inp1.size(0)
    for i in range(iters):
        idx = torch.randint(0, sz, ())
        cur_inp1 = inp1[idx].to(device)
        cur_inp2 = inp2[idx].to(device)
        cur_inp3 = inp3[idx].to(device)
        cur_sym = cur_syms[idx].to(device)
        # output_fp = cached_output[idx].to(device)
        cur_out = cached_fused_feature[idx].to(device)
        if input_prob < 1.0:
            drop_inp = torch.where(torch.rand_like(cur_inp1) < input_prob, cur_inp1, cur_sym)

        if w_opt:
            w_opt.zero_grad()
        if a_opt:
            a_opt.zero_grad()
        
        '''forward for prediction difference'''
        out_drop = qt_block(drop_inp, cur_inp2, cur_inp3) # noisy input
        # out_quant = qt_block(cur_inp) # clean input
        # output = out_quant
        # for num, module in enumerate(module_list):
        #     output = module(output) # TODO
        # qt output(noisy input), fp output, qt pred, fp pred
        err = loss_func(out_drop, cur_out, None, None)

        err.backward(retain_graph=False)
        if w_opt:
            w_opt.step()    
        if a_opt:
            a_opt.step()
        if scheduler:
            scheduler.step()
        if a_scheduler:
            a_scheduler.step()
    torch.cuda.empty_cache()

    for module in qt_block.modules():
        if isinstance(module, QuantModule):
            '''weight '''
            module.weight_quantizer.soft_targets = False
        '''activation'''
        if isinstance(module, (QuantModule, BaseQuantBlock)):
            module.trained = True
            if not hasattr(module, 'act_quantizer'):
                continue # some BaseQuantBlock has no act_quantizer (e.g., QuantBaseBevBackbone)
            module.act_quantizer.is_training = False
    for module in fp_block.modules():
        if isinstance(module, (QuantModule, BaseQuantBlock)):
            module.trained = True

class LossFunction:
    def __init__(self,
                 block: BaseQuantBlock,
                 round_loss: str = 'relaxation',
                 weight: float = 1.,
                 rec_loss: str = 'mse',
                 max_count: int = 2000,
                 b_range: tuple = (10, 2),
                 decay_start: float = 0.0,
                 warmup: float = 0.0,
                 p: float = 2.,
                 lam: float = 1.0,
                 T: float = 7.0):

        self.block = block
        self.round_loss = round_loss
        self.weight = weight
        self.rec_loss = rec_loss
        self.loss_start = max_count * warmup
        self.p = p
        self.lam = lam
        self.T = T

        self.temp_decay = LinearTempDecay(max_count, rel_start_decay=warmup + (1 - warmup) * decay_start,
                                          start_b=b_range[0], end_b=b_range[1])
        self.count = 0
        self.hetero_loss = torch.nn.KLDivLoss(reduction='batchmean')

    def __call__(self, pred, tgt, output, output_fp):
        """
        Compute the total loss for adaptive rounding:
        rec_loss is the quadratic output reconstruction loss, round_loss is
        a regularization term to optimize the rounding policy, pd_loss is the 
        prediction difference loss.

        :param pred: output from quantized model
        :param tgt: output from FP model
        :param output: prediction from quantized model
        :param output_fp: prediction from FP model
        :return: total loss function
        """
        self.count += 1
        if self.rec_loss == 'mse':
            rec_loss = lp_loss(pred, tgt, p=self.p)
        else:
            raise ValueError('Not supported reconstruction loss function: {}'.format(self.rec_loss))

        # hetero_loss = self.hetero_loss(F.log_softmax(output / self.T, dim=1), F.softmax(output_fp / self.T, dim=1)) / self.lam

        b = self.temp_decay(self.count)
        if self.count < self.loss_start or self.round_loss == 'none':
            b = round_loss = 0
        elif self.round_loss == 'relaxation':
            round_loss = 0
            # for name, module in self.block.named_modules():
            #     if isinstance(module, QuantModule):
            #         round_vals = module.weight_quantizer.get_soft_targets()
            #         round_loss += self.weight * (1 - ((round_vals - .5).abs() * 2).pow(b)).sum()
        else:
            raise NotImplementedError

        total_loss = rec_loss + round_loss
        # total_loss += hetero_loss
        if self.count % 50 == 0:
            print('Total loss:\t{:.5f} (rec:{:.5f}, round:{:.5f})\tb={:.5f}\tcount={}'.format(
                float(total_loss), float(rec_loss), float(round_loss), b, self.count))
        return total_loss


class LinearTempDecay:
    def __init__(self, t_max: int, rel_start_decay: float = 0.2, start_b: int = 10, end_b: int = 2):
        self.t_max = t_max
        self.start_decay = rel_start_decay * t_max
        self.start_b = start_b
        self.end_b = end_b

    def __call__(self, t):
        """
        Cosine annealing scheduler for temperature b.
        :param t: the current time step
        :return: scheduled temperature
        """
        if t < self.start_decay:
            return self.start_b
        else:
            rel_t = (t - self.start_decay) / (self.t_max - self.start_decay)
            # return self.end_b + 0.5 * (self.start_b - self.end_b) * (1 + np.cos(rel_t * np.pi))
            return self.end_b + (self.start_b - self.end_b) * max(0.0, (1 - rel_t))
