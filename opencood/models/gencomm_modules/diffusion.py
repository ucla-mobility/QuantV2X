import torch
from torch import nn, einsum
import torch.nn.functional as F
from opencood.models.gencomm_modules.network_modules import linear_beta_schedule, cosine_beta_schedule, quadratic_beta_schedule, sigmoid_beta_schedule
from opencood.models.gencomm_modules.network_modules import Unet
from typing import Tuple

def regroup(x, record_len):
    cum_sum_len = torch.cumsum(record_len, dim=0)
    split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
    return split_x

class DiffComm(nn.Module):
    def __init__(self, dim, channels, timesteps, ):
        super(DiffComm, self).__init__()
        self.denoise_model = Unet(dim=dim, channels = channels,)
        self.loss = nn.MSELoss()
        
        # diffusion 
        self.timesteps = timesteps
        self.betas = linear_beta_schedule(timesteps=timesteps)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        # diffusion calculations
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        
        # posterior calculations
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        
        
    def extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    # forward diffusion (using the nice property)
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape).to(x_start.device)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        ).to(x_start.device)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, denoise_model, x_start, t, labels,noise=None, loss_type="l1"):
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = denoise_model(x_noisy, t, labels = labels)

        if loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()

        return loss

    @torch.no_grad()
    def p_sample(self, model, x, t, t_index, labels=None):
        betas_t = self.extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = self.extract(self.sqrt_recip_alphas, t, x.shape)
        
        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, t, labels = labels) / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self.extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            # Algorithm 2 line 4:
            return model_mean + torch.sqrt(posterior_variance_t) * noise 

    # Algorithm 2 (including returning all images)
    @torch.no_grad()
    def p_sample_loop(self, model, shape, x_noise=None, conditions=None, labels=None):
        device = next(self.denoise_model.parameters()).device # get the device of the model

        b = shape[0]
        # start from pure noise (for each example in the batch)
        if x_noise is not None:
            img = x_noise
        else:
            img = torch.randn(shape, device=device)

        if conditions is not None:
            img_dim = img.shape[1]
            img = torch.cat([img, conditions], dim=1)
            for i in reversed(range(self.timesteps)):
                img = self.p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i, labels)
            return img[:, :img_dim, :, :]   # return the image without the conditions
        else:
            for i in reversed(range(self.timesteps)):
                img = self.p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i, labels)
            return img

    @torch.no_grad()
    def sample(self, model, shape: Tuple[int], x_noise = None, conditions=None, labels=None,):
        return self.p_sample_loop(model, shape=shape, x_noise=x_noise, conditions=conditions, labels=labels)
        
    def forward(self, x, record_len, conditions=None):
        x_split = regroup(x, record_len)
        conditions_split = regroup(conditions, record_len)
        x_out = []
        for i, _x in enumerate(x_split):
            if x_split[i].shape[0] == 1: # cav_num == 1 ego dont need to sample
                x_out.append(_x)
            else:
                ego = _x[0].unsqueeze(0) # ego dont need to sample
                _x = _x[1:] 
                x_noise = self.q_sample(x_start=_x, t=torch.full((_x.shape[0],), self.timesteps - 1)) # -1 : index 
                x_new = x_noise.clone()
                shape = x_noise.shape
                if conditions is not None:
                    if conditions[i].shape[0] > 1:
                        conditions_ = conditions_split[i][1:]
                    else:
                        conditions_ =  conditions_split[i]
                x_new = self.sample(self.denoise_model, shape, x_noise, conditions_) # sampling
                x_new = torch.cat([ego, x_new], dim=0)
                x_out.append(x_new)
        x_out = torch.cat(x_out, dim=0)
        loss = self.loss(x_out, x)
        return x_out, loss
    
if __name__ == '__main':
    input = torch.randn(5, 128, 100, 352)
    conditions = torch.randn(5, 2, 100, 352)
    record_len = torch.tensor([1, 2, 2])
    diffcomm = DiffComm(dim=130, channels=130, timesteps=3,)
    out, loss = diffcomm(input, record_len, conditions = conditions)
    print(out.shape, loss)