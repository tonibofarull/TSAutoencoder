import torch
from torch import nn
import torch.nn.functional as F
from models.base_model import BaseModel
from models.losses import MSE_regularized, Clus

class CAE(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        M, Lf, bottleneck_nn, length = self.M, self.Lf, self.bottleneck_nn, self.length
        # hidden_nn = self.hidden_nn
        self.conv1 = nn.Conv1d(1, M, kernel_size=Lf,)
        self.conv2 = nn.Conv1d(1, M, kernel_size=Lf, dilation=2, padding=1)
        self.conv3 = nn.Conv1d(1, M, kernel_size=Lf, dilation=4, padding=3)
        self.conv4 = nn.Conv1d(1, M, kernel_size=Lf, dilation=8, padding=7)
        self.full1 = nn.Linear(4*M*(length-Lf+1), bottleneck_nn)
        self.full2 = nn.Linear(bottleneck_nn, 4*M*(length-Lf+1))
        self.deco1 = nn.ConvTranspose1d(M, 1, kernel_size=Lf)
        self.deco2 = nn.ConvTranspose1d(M, 1, kernel_size=Lf, dilation=2, padding=1)
        self.deco3 = nn.ConvTranspose1d(M, 1, kernel_size=Lf, dilation=4, padding=3)
        self.deco4 = nn.ConvTranspose1d(M, 1, kernel_size=Lf, dilation=8, padding=7)

        self.full3 = nn.Linear(bottleneck_nn, 3)
        # self.full3 = nn.Linear(bottleneck_nn, hidden_nn)
        # self.full4 = nn.Linear(hidden_nn, 3)

        self.shape = (-1,4*M, length-Lf+1)

    def forward(self, X, get_bottleneck=False, get_training=False):
        X = X[:,:,:-1] # discard the cluster information
        bottleneck = self.encode(X)
        if get_bottleneck:
            return bottleneck
        if get_training:
            return self.decode(bottleneck), self.soft_clustering(bottleneck)
        return self.decode(bottleneck)

    def encode(self, X):
        X1 = self.conv1(X)
        X2 = self.conv2(X)
        X3 = self.conv3(X)
        X4 = self.conv4(X)
        X = torch.cat((X1,X2,X3,X4), dim=1)
        X = torch.flatten(X, start_dim=1)
        return torch.sigmoid(self.full1(X)) # bottleneck

    def decode(self, bottleneck):
        X = self.full2(bottleneck)

        X = torch.reshape(X, self.shape)
        X1 = self.deco1(X[:,:self.M,:])
        X2 = self.deco2(X[:,self.M:2*self.M,:])
        X3 = self.deco3(X[:,2*self.M:3*self.M,:])
        X4 = self.deco4(X[:,3*self.M:,:])
        return X1+X2+X3+X4

    def soft_clustering(self, bootleneck):
        return self.full3(bootleneck)
        # return self.full4(F.leaky_relu(self.full3(bootleneck)))

    def loss(self, batch):
        return Clus(self, batch, self.reg, self.alpha)