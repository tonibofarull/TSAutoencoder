import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


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
        # - Whole model:
        loss_reg = _my_l1(model)
        # - Filters:
        loss_col = _my_group_col(model)
        # - Neurons:
        #loss_row = _my_group_row(model)

        loss = self.alpha*loss_class + (1-self.alpha)*loss_recon + self.lmd*(loss_reg + loss_col)
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

def _my_group_col(model): # we should apply l1-reg as well
    loss = 0
    columns = model.k*model.M
    weights_per_column = model.length
    for i in range(columns):
        group = model.full_conv_bn.weight[:,weights_per_column*i:weights_per_column*(i+1)]
        loss += np.sqrt(len(group))*torch.sqrt(torch.sum(torch.square(group)))
    return loss

def _my_group_row(model): # we should apply l1-reg as well
    loss = 0
    for i in range(model.bottleneck_nn):
        group = model.full_conv_bn.weight[i]
        loss += np.sqrt(len(group))*torch.sqrt(torch.sum(torch.square(group)))
    return loss
