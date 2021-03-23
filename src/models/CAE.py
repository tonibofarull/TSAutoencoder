import torch
from torch import nn
import torch.nn.functional as F
from models.losses import CAELoss

class Encoder(nn.Module):
    def __init__(self, k, M, Lf, dilation, length, bottleneck_nn):
        super().__init__()
        # By using ModuleList we ensure that the layers of the list are properly registered
        self.conv1 = nn.ModuleList([nn.Conv1d(1, M, kernel_size=Lf, dilation=d, padding=d*(Lf-1)//2) for d in dilation])
        self.act1 = nn.LeakyReLU()
        self.fc_conv_bn = nn.Linear(k*M*length, bottleneck_nn)
        self.last_act = nn.LeakyReLU()

    def forward(self, X, apply_noise): # (N, 1, length)
        if apply_noise:
            noise = torch.normal(mean=0, std=0.05, size=X.shape)
            X = torch.clip(X+noise, min=0, max=1)
        X = [conv(X) for conv in self.conv1] # [(N, M, length)]*k
        X = torch.cat(X, dim=1) # (N, k*M, length)
        X = self.act1(torch.flatten(X, start_dim=1)) # (N, k*M*length)
        X = self.last_act(self.fc_conv_bn(X)) # (N, bottleneck_nn)
        return X

class Decoder(nn.Module):
    def __init__(self, k, M, Lf, dilation, length, bottleneck_nn):
        super().__init__()
        self.k, self.M, self.length = k, M, length

        self.fc_bn_deco = nn.Linear(bottleneck_nn, k*M*length)
        self.act1 = nn.LeakyReLU()
        self.deco1 = nn.ModuleList([nn.ConvTranspose1d(M, 1, kernel_size=Lf, dilation=d, padding=d*(Lf-1)//2) for d in dilation])
        self.last_act = nn.Sigmoid()

    def forward(self, X): # (N, bottleneck_nn)
        X = self.act1(self.fc_bn_deco(X)) # (N, k*M*length)
        X = X.reshape(-1, self.k*self.M, self.length) # (N, k*M, length)
        X = torch.split(X, split_size_or_sections=self.M, dim=1) # [(N, M, length)]*k
        X = [deco(X[i]) for i, deco in enumerate(self.deco1)] # [(N, 1, length)]*k
        X = torch.stack(X) # (k, N, 1, length)
        X = self.last_act(torch.sum(X, dim=0)) # (N, 1, length), output values between 0 and 1
        return X

class Classifier(nn.Module):
    def __init__(self, bottleneck_nn, num_classes,
        hidden_nn=32
    ):
        super().__init__()
        self.fc1 = nn.Linear(bottleneck_nn, hidden_nn)
        self.act1 = nn.LeakyReLU()
        self.fc2 = nn.Linear(hidden_nn, num_classes)

    def forward(self, X):
        """
        Apply "get_probs" to X to obtain probabilities
        """
        X = self.act1(self.fc1(X))
        X = self.fc2(X)
        return X

    # UTILS

    @staticmethod
    def get_probs(X):
        return torch.nn.functional.softmax(X, dim=1)


class CAE(nn.Module):
    def __init__(self, cfg,
        dilation=[1, 2, 4, 8],
        num_classes=3
    ):
        super().__init__()
        self.k = len(dilation) # Number of dilations
        self.M = cfg.M # Number of filters per dilation
        self.Lf = cfg.Lf
        self.bottleneck_nn = cfg.bottleneck_nn
        self.length = cfg.length
        self.dilation = dilation
        self.num_classes = num_classes
        self.lossf = CAELoss(alpha=cfg.alpha, lmd=cfg.lmd)

        k, M, Lf, bottleneck_nn, length = self.k, self.M, self.Lf, self.bottleneck_nn, self.length
        # MODULES DEFINITION
        self.encoder = Encoder(k, M, Lf, dilation, length, bottleneck_nn)
        self.decoder = Decoder(k, M, Lf, dilation, length, bottleneck_nn)
        self.classifier = Classifier(bottleneck_nn, num_classes)

    def forward(self, X, apply_noise=True):
        """
        X: (N, 1, length), where N is the number of observations in the batch with the corresponding length
        pred_X: (N, 1, length)
        pred_class: (N, 1, num_classes)
        bottleneck: (N, bottleneck_nn)
        """
        bottleneck = self.encoder(X, apply_noise)
        pred_X = self.decoder(bottleneck)
        pred_class = self.classifier(bottleneck)
        return pred_X, pred_class, bottleneck

    def loss(self, batch, apply_reg=True):
        X, clss = CAE.split_data(batch)
        clss = torch.flatten(clss).long()
        pred_X, pred_class, _ = self(X)
        return self.lossf(self, pred_X, X, pred_class, clss, apply_reg)

    # UTILS

    @staticmethod
    def split_data(batch):
        """
        batch: (N, 1, length+1) where the extra column of an observation is the class
        """
        X, y = batch[:,:,:-1], batch[:,:,-1]
        return X, y

# class Encoder2(nn.Module):
#     def __init__(self, k, M, Lf, dilation, length, bottleneck_nn):
#         super().__init__()
#         # By using ModuleList we the layers of the list are properly registered
#         self.conv = nn.Linear(length, k*M*length)
#         self.first_act = nn.LeakyReLU()
#         self.full_conv_bn = nn.Linear(k*M*length, bottleneck_nn)
#         self.last_act = nn.LeakyReLU()

#     def forward(self, X, apply_noise): # (N, 1, length)
#         inp = X
#         if apply_noise:
#             inp = torch.clip(X+torch.normal(mean=0, std=0.05, size=X.shape), min=0, max=1)
#         inp = inp.reshape(-1,96)
#         X = self.first_act(self.conv(inp)) # (N, k*M*length)
#         X = self.last_act(self.full_conv_bn(X)) # (N, bottleneck_nn)
#         return X

# class Decoder2(nn.Module):
#     def __init__(self, k, M, Lf, dilation, length, bottleneck_nn):
#         super().__init__()
#         self.k, self.M, self.length = k, M, length

#         self.full_bn_deco = nn.Linear(bottleneck_nn, k*M*length)
#         self.first_act = nn.LeakyReLU()
#         self.deco = nn.Linear(k*M*length, length)
#         self.last_act = nn.Sigmoid()

#     def forward(self, X): # (N, bottleneck_nn)
#         X = self.first_act(self.full_bn_deco(X))
#         X = self.last_act(self.deco(X)) # (N, 1, length), output values between 0 and 1
#         X = X.reshape(-1,1,96)
#         return X