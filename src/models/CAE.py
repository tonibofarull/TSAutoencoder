import torch
from torch import nn
import torch.nn.functional as F
from models.losses import CAELoss

class CAE(nn.Module):
    def __init__(self, cfg,
        dilation = [2, 3, 5, 7],
        num_classes = 3
    ):
        super().__init__()
        self.k = len(dilation)
        self.M = cfg.M
        self.Lf = cfg.Lf
        self.bottleneck_nn = cfg.bottleneck_nn
        self.length = cfg.length
        self.dilation = dilation
        self.num_classes = num_classes
        self.lossf = CAELoss(lmd=cfg.lmd, alpha=cfg.alpha)

        k, M, Lf, bottleneck_nn, length = self.k, self.M, self.Lf, self.bottleneck_nn, self.length

        # LAYER DEFINITION

        # By using ModuleList we the layers of the list are properly registered
        self.conv = nn.ModuleList([nn.Conv1d(1, M, kernel_size=Lf, dilation=d, padding=d*(Lf-1)//2) for d in dilation])
        self.full1 = nn.Linear(k*M*length, bottleneck_nn)
        self.full2 = nn.Linear(bottleneck_nn, k*M*length)
        self.deco = nn.ModuleList([nn.ConvTranspose1d(M, 1, kernel_size=Lf, dilation=d, padding=d*(Lf-1)//2) for d in dilation])

        self.full3 = nn.Linear(bottleneck_nn, num_classes)

        self.full4 = nn.Linear(num_classes, k)

    def forward(self, X):
        """
        X: (N, 1, length), where N is the number of observations in the batch with the corresponding length
        pred_X: same shape as X
        pred_class: (N, 1, num_classes)
        bottleneck: (N, bottleneck_nn)
        """
        bottleneck = self.encode(X)
        pred_X = self.decode(bottleneck)
        pred_class = self.soft_classification(bottleneck)
        return pred_X, pred_class, bottleneck

    def encode(self, X):
        X = [conv(X) for conv in self.conv]
        X = torch.cat(X, dim=1)
        X = torch.flatten(X, start_dim=1)
        bottleneck = F.leaky_relu(self.full1(X))
        return bottleneck

    def decode(self, bottleneck):
        X = self.full2(bottleneck)

        X = torch.reshape(X, (-1,self.k*self.M, self.length))
        X = [deco(X[:,i*self.M:(i+1)*self.M,:]) for i, deco in enumerate(self.deco)]
        X = torch.stack(X)
        X = torch.sum(X, dim=0)
        return X

    def soft_classification(self, bootleneck):
        """
        Apply torch.nn.functional.softmax(output, dim=1) to obtain probabilities
        """
        return self.full3(bootleneck)

    def loss(self, batch):
        X, clss = self.split_data(batch)
        clss = torch.flatten(clss).long()
        pred_X, pred_class, _ = self(X)

        pred_dil = torch.nn.functional.softmax(self.full4(pred_class), dim=1)

        return self.lossf(self, pred_X, X, pred_class, clss, pred_dil)

    # UTILS

    def split_data(self, batch):
        """
        batch: (N, 1, length+1) where the extra column of an observation is the class
        """
        X, y = batch[:,:,:-1], batch[:,:,-1]
        return X, y
