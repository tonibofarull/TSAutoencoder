from numpy.core.fromnumeric import std
import torch
from torch import nn
from functional import *

class VAutoEncoder(nn.Module):
    def __init__(self, length, Lf, M, bottleneck_nn):
        super().__init__()
        self.conv1 = nn.Conv1d(1, M, kernel_size=Lf,)
        self.conv2 = nn.Conv1d(1, M, kernel_size=Lf, dilation=2, padding=1)
        self.conv3 = nn.Conv1d(1, M, kernel_size=Lf, dilation=4, padding=3)
        self.conv4 = nn.Conv1d(1, M, kernel_size=Lf, dilation=8, padding=7)
        self.full1 = nn.Linear(4*M*(length-Lf+1), 2*bottleneck_nn) # 5 mean, 5 stds
        self.full2 = nn.Linear(bottleneck_nn, 4*M*(length-Lf+1))
        self.deco1 = nn.ConvTranspose1d(M, 1, kernel_size=Lf)
        self.deco2 = nn.ConvTranspose1d(M, 1, kernel_size=Lf, dilation=2, padding=1)
        self.deco3 = nn.ConvTranspose1d(M, 1, kernel_size=Lf, dilation=4, padding=3)
        self.deco4 = nn.ConvTranspose1d(M, 1, kernel_size=Lf, dilation=8, padding=7)

        self.M = M
        self.shape = (-1,4*M,length-Lf+1)

    def forward(self, X, get_bottleneck=False, is_train=False):
        z, mean, logvar = self.encode(X)
        if get_bottleneck:
            return z
        pred = self.decode(z)
        if is_train:
            return pred, mean, logvar
        return pred

    def reparameterizer(mean, logvar):
        eps = torch.normal(mean=0.0, std=0.1, size=(mean.shape))
        return mean + torch.exp(logvar)*eps

    def encode(self, X):
        X1 = self.conv1(X)
        X2 = self.conv2(X)
        X3 = self.conv3(X)
        X4 = self.conv4(X)
        X = torch.cat((X1,X2,X3,X4), dim=1)
        X = torch.flatten(X, start_dim=1)
        mean, logvar = torch.chunk(self.full1(X), chunks=2, dim=1)
        z = VAutoEncoder.reparameterizer(mean, logvar)
        return torch.sigmoid(z), mean, logvar

    def decode(self, bottleneck):
        X = self.full2(bottleneck)

        X = torch.reshape(X, self.shape)
        X1 = self.deco1(X[:,:self.M,:])
        X2 = self.deco2(X[:,self.M:2*self.M,:])
        X3 = self.deco3(X[:,2*self.M:3*self.M,:])
        X4 = self.deco4(X[:,3*self.M:,:])
        return X1+X2+X3+X4 
        # combine the transposed convolutions (element-wise addition)
