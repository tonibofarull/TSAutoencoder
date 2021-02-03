import torch
from torch import nn
import torch.nn.functional as F


class CAELoss(nn.Module):
    def __init__(self, alpha, lmd):
        super().__init__()
        self.alpha = alpha
        self.lmd = lmd

    def forward(self, model, pred_X, X, pred_clss, clss):
        # Classification:
        loss_class = F.cross_entropy(pred_clss, clss)
        # Reconstruction:
        # loss_recon = F.mse_loss(pred_X, X)
        loss_recon = F.binary_cross_entropy(pred_X, X) # if output [0,1]
        # Regularization:
        loss_reg = _my_l2(model)

        loss = self.alpha*loss_class + (1-self.alpha)*loss_recon + self.lmd*loss_reg
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
