import torch

def VAE_loss(model, batch, lmd=0.1):
    pred, mean, logvar = model.forward(batch, is_train=True)

    recon = torch.mean((pred-batch)**2)
    KL = 0.5 * torch.mean(torch.sum(torch.exp(logvar) + torch.square(mean) - 1.0 - logvar, dim=0))

    return recon + lmd*KL #+ my_l1(model)

def MSE_regularized(model, batch):
    pred = model(batch)
    return _my_mse(pred,batch) + _my_l1(model)

# UTILS

def _my_mse(pred, real):
    N = torch.numel(pred) # pred.shape[0]
    return 1/N * torch.sum((pred-real)**2)

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
