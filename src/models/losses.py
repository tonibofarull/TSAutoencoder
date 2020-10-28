import torch
import torch.nn.functional as F

def VCAE_loss(model, batch, lmd=1):
    pred, mean, logvar = model(batch, is_train=True)

    recon = F.mse_loss(pred,batch) # or binary_cross_entropy when [0,1]
    KL = 0.5 * torch.mean(torch.sum(torch.exp(logvar) + torch.square(mean) - 1.0 - logvar, dim=0))
    # KL(p(x|z) || N(0,I))

    return recon + lmd*KL #+ my_l1(model)

def MSE_regularized(model, batch):
    pred = model(batch)
    return F.mse_loss(pred,batch) + _my_l1(model)

# UTILS

def _my_l2(model, decay=0.01):
    l2 = 0
    for param in model.parameters():
        l2 += torch.sum(torch.square(param))
    return decay * l2

def _my_l1(model, decay=0.001):
    l1 = 0
    for param in model.parameters():
        l1 += torch.sum(torch.abs(param))
    return decay * l1
