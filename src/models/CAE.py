import torch
from torch import nn
import torch.nn.functional as F
from models.losses import CAELoss

class CAE(nn.Module):
    def __init__(self, cfg,
        dilation = [1, 2, 4, 8],
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
        self.full_conv_bn = nn.Linear(k*M*length, bottleneck_nn)
        # Bottleneck
        # For reconstruction:
        self.full_bn_deco = nn.Linear(bottleneck_nn, k*M*length)
        self.deco = nn.ModuleList([nn.ConvTranspose1d(M, 1, kernel_size=Lf, dilation=d, padding=d*(Lf-1)//2) for d in dilation])
        # For classification:
        self.full_bn_class1 = nn.Linear(bottleneck_nn, 32)
        self.full_bn_class2 = nn.Linear(32, num_classes)

    def forward(self, X):
        """
        X: (N, 1, length), where N is the number of observations in the batch with the corresponding length
        pred_X: (N, 1, length)
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
        X = F.leaky_relu(X)
        bottleneck = F.leaky_relu(self.full_conv_bn(X))
        return bottleneck

    def decode(self, bottleneck):
        X = F.leaky_relu(self.full_bn_deco(bottleneck))

        X = torch.reshape(X, (-1,self.k*self.M, self.length))
        X = [deco(X[:,i*self.M:(i+1)*self.M,:]) for i, deco in enumerate(self.deco)]
        X = torch.stack(X)
        X = torch.sum(X, dim=0)
        return torch.sigmoid(X) # output between 0 and 1

    def soft_classification(self, bootleneck):
        """
        Apply torch.nn.functional.softmax(output, dim=1) to X to obtain probabilities
        """
        X = F.leaky_relu(self.full_bn_class1(bootleneck))
        X = self.full_bn_class2(X)
        return X

    def loss(self, batch):
        X, clss = self.split_data(batch)
        clss = torch.flatten(clss).long()
        pred_X, pred_class, _ = self(X)
        return self.lossf(self, pred_X, X, pred_class, clss)

    # UTILS

    def split_data(self, batch):
        """
        batch: (N, 1, length+1) where the extra column of an observation is the class
        """
        X, y = batch[:,:,:-1], batch[:,:,-1]
        return X, y
