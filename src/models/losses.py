import torch
from torch import nn
import torch.nn.functional as F


class CAELoss(nn.Module):
    def __init__(self, alpha, lmd):
        super().__init__()
        self.alpha = alpha
        self.lmd = lmd

    def forward(self, model, pred_X, X, pred_clss, clss, pred_dil):
        loss_recon = F.mse_loss(pred_X, X)
        loss_class = F.cross_entropy(pred_clss, clss)
        loss_reg = _my_l1(model)
        loss = (1-self.alpha)*loss_recon + self.alpha*loss_class + self.lmd*loss_reg

        # Automatic penalization of the dilation: full1 has dimension (bottleneck_nn, k*M*length)
        # penalize_wrongdil = 0
        # weight_per_dil = model.length*model.M
        # for i in range(model.k):
        #     di = torch.sum(torch.abs(model.full1.weight[:,i*weight_per_dil:(i+1)*weight_per_dil]))
        #     penalize_wrongdil += torch.mean(pred_dil[:,i]*di)
        # loss += self.lmd/10*penalize_wrongdil

        return loss

# UTILS

def _my_l2(model):
    l2 = 0
    for param in model.parameters():
        l2 += torch.sum(torch.square(param))
    return l2

def _my_l1(model):
    l1 = 0
    for param in model.parameters():
        l1 += torch.sum(torch.abs(param))
    return l1
