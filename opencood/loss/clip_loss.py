# implementation of CLIP loss here

import torch
import torch.nn.functional as F
import torch.nn as nn

class CLIPLoss(nn.Module):
    def __init__(self, args):
        super(CLIPLoss, self).__init__()
        self.logit_scale = nn.Parameter(torch.tensor(args['logit_scale'], dtype=torch.float32))
        self.loss_dict = {}

    def forward(self, embedding1, embedding2):
        """
        Forward pass for CLIP loss.
        Parameters:
            embedding1: Tensor, embeddings from modality 1 (e.g., backbone1 output).
            embedding2: Tensor, embeddings from modality 2 (e.g., backbone2 output).
        Returns:
            loss: CLIP loss value.
        """
        embedding1 = embedding1 / embedding1.norm(dim=1, keepdim=True)
        embedding2 = embedding2 / embedding2.norm(dim=1, keepdim=True)
        scale = self.logit_scale.exp()
        logits = scale * embedding1 @ embedding2.t()
        targets = torch.arange(logits.size(0), device=logits.device) # matching embeddings along the diagonal
        loss = F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets)
        loss = loss / 2.0
        self.loss_dict.update({
            'clip_loss': loss.item()
        })
        return loss

    def logging(self, epoch, batch_id, batch_len, writer=None):
        """
        Log the CLIP loss during training.
        """
        clip_loss = self.loss_dict.get('clip_loss', 0)

        print("[epoch %d][%d/%d] || CLIP Loss: %.4f" % (
            epoch, batch_id + 1, batch_len, clip_loss
        ))

        if writer is not None:
            writer.add_scalar('CLIP_loss', clip_loss, epoch * batch_len + batch_id)
