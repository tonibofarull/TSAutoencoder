import torch
import torch.nn.functional as F
from torch import nn
from models.base_model import BaseModel
from models.losses import VCAE_loss

class VCAE(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        M, Lf, bottleneck_nn, length = self.M, self.Lf, self.bottleneck_nn, self.length
        self.conv1 = nn.Conv1d(1, M, kernel_size=Lf,)
        self.conv2 = nn.Conv1d(1, M, kernel_size=Lf, dilation=2, padding=1)
        self.conv3 = nn.Conv1d(1, M, kernel_size=Lf, dilation=4, padding=3)
        self.conv4 = nn.Conv1d(1, M, kernel_size=Lf, dilation=8, padding=7)
        self.full1 = nn.Linear(4*M*(length-Lf+1), 2*bottleneck_nn) # mean and logvar
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
            return mean
        pred = self.decode(z)
        if is_train:
            return pred, mean, logvar
        return pred

    def encode(self, X):
        X1 = self.conv1(X)
        X2 = self.conv2(X)
        X3 = self.conv3(X)
        X4 = self.conv4(X)
        X = torch.cat((X1,X2,X3,X4), dim=1)
        X = F.relu(torch.flatten(X, start_dim=1)) # relu since logvar > 0
        mean, logvar = torch.chunk(self.full1(X), chunks=2, dim=1)
        z = VCAE._reparameterizer(mean, logvar)
        return z, mean, logvar # bottleneck, mean and logvar

    def decode(self, bottleneck):
        X = F.relu(self.full2(bottleneck))
        X = torch.reshape(X, self.shape)
        X1 = self.deco1(X[:,:self.M,:])
        X2 = self.deco2(X[:,self.M:2*self.M,:])
        X3 = self.deco3(X[:,2*self.M:3*self.M,:])
        X4 = self.deco4(X[:,3*self.M:,:])
        return X1+X2+X3+X4

    def _reparameterizer(mean, logvar):
        eps = torch.normal(mean=0.0, std=0.1, size=(mean.shape))
        return mean + torch.exp(logvar/2)*eps
    
    def loss(self, batch):
        return VCAE_loss(self, batch, self.reg, self.regKL)
