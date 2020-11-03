import torch
import torch.nn.functional as F

def VCAE_loss(model, batch, lmd, regKL):
    pred, mean, logvar = model(batch, is_train=True)

    recon = F.mse_loss(pred,batch) # or binary_cross_entropy when [0,1]
    KL = 0.5 * torch.mean(torch.sum(torch.exp(logvar) + torch.square(mean) - 1.0 - logvar, dim=0))
    # KL(p(x|z) || N(0,I))

    return recon + regKL*KL + lmd*_my_l1(model)

def MSE_regularized(model, batch, lmd):
    pred = model(batch)
    return F.mse_loss(pred,batch) + lmd*_my_l1(model)

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
