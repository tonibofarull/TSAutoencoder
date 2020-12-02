import torch
from torch import nn
import torch.nn.functional as F
from models.base_model import BaseModel
from models.losses import MSE_regularized, Clus

class CAE(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        M, Lf, bottleneck_nn, length = self.M, self.Lf, self.bottleneck_nn, self.length
        hidden_nn = self.hidden_nn
        self.deep_cluster = False
        num_classes = 3
        self.dilation = [2,3,5,7]

        self.conv = [nn.Conv1d(1, M, kernel_size=Lf, dilation=d, padding=d-1) for d in self.dilation]
        self.full1 = nn.Linear(4*M*(length-Lf+1), bottleneck_nn)
        self.full2 = nn.Linear(bottleneck_nn, 4*M*(length-Lf+1))
        self.deco = [nn.ConvTranspose1d(M, 1, kernel_size=Lf, dilation=d, padding=d-1) for d in self.dilation]

        if self.deep_cluster:
            self.full3 = nn.Linear(bottleneck_nn, hidden_nn)
            self.full4 = nn.Linear(hidden_nn, num_classes)
        else:
            self.full3 = nn.Linear(bottleneck_nn, num_classes)


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
        X = [conv(X) for conv in self.conv]
        X = torch.cat(X, dim=1)
        X = torch.flatten(X, start_dim=1)
        return F.leaky_relu(self.full1(X)) # bottleneck

    def decode(self, bottleneck):
        X = self.full2(bottleneck)

        X = torch.reshape(X, self.shape)
        X = [deco(X[:,i*self.M:(i+1)*self.M,:]) for i, deco in enumerate(self.deco)]
        X = torch.stack(X)
        X = torch.sum(X, dim=0)
        return X

    def soft_clustering(self, bootleneck):
        if self.deep_cluster:
            return self.full4(F.leaky_relu(self.full3(bootleneck)))
        return self.full3(bootleneck)

    def loss(self, batch):
        return Clus(self, batch, self.reg, self.alpha)
