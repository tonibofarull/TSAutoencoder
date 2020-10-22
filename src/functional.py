import torch

def my_mse(pred, real):
    N = torch.numel(pred) # pred.shape[0]
    return 1/N * torch.sum((pred-real)**2)

def my_l2(model, decay=0.01):
    l2 = 0
    for param in model.parameters():
        l2 += torch.sum(torch.square(param))
    return decay * l2

def my_l1(model, decay=0.001):
    l1 = 0
    for param in model.parameters():
        l1 += torch.sum(torch.abs(param))
    return decay * l1
