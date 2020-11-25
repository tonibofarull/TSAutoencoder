from torch import nn
from hydra.experimental import compose, initialize

class BaseModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.M = cfg.M
        self.Lf = cfg.Lf
        self.bottleneck_nn = cfg.bottleneck_nn
        self.length = cfg.length
        self.reg = cfg.reg
        self.regKL = cfg.regKL

        self.hidden_nn = cfg.hidden_nn
        self.alpha = cfg.alpha