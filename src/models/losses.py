import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class CAELoss(nn.Module):
    def __init__(self, alpha, lmd):
        super().__init__()
        self.alpha = alpha
        self.lmd = lmd

    def forward(self, model, pred_X, X, pred_clss, clss, apply_reg=True):
        # Classification:
        loss_class = F.cross_entropy(pred_clss, clss)
        # Reconstruction (in case of arbitrary output, use F.mse_loss(pred_X, X)):
        loss_recon = F.binary_cross_entropy(pred_X, X)  # since output [0,1]

        loss = self.alpha*loss_class + (1-self.alpha)*loss_recon
        if apply_reg:
            # Regularization:
            # - Whole model:
            loss_reg = CAELoss._l1(model)
            # - Filters:
            loss_col = CAELoss._group_col(model)
            # - Neurons:
            loss_row = CAELoss._group_row(model)
            loss += self.lmd*(loss_reg + loss_col + loss_row)
        return loss

    # UTILS

    @staticmethod
    def _l2(model):
        l2 = 0
        for param in model.parameters():
            l2 += torch.sum(torch.square(param))
        return l2

    @staticmethod
    def _l1(model):
        l1 = 0
        for param in model.parameters():
            l1 += torch.sum(torch.abs(param))
        return l1

    @staticmethod
    def _group_col(model):
        loss = 0
        columns = model.k*model.M
        weights_per_column = model.length
        for i in range(columns):
            group = model.encoder.fc_conv_bn.weight[:, weights_per_column*i: weights_per_column*(i+1)]
            loss += np.sqrt(len(group)) * torch.sqrt(torch.sum(torch.square(group)))
        return loss

    @staticmethod
    def _group_row(model):
        loss = 0
        for i in range(model.bottleneck_nn):
            group = model.encoder.fc_conv_bn.weight[i]
            loss += np.sqrt(len(group)) * torch.sqrt(torch.sum(torch.square(group)))
        return loss
