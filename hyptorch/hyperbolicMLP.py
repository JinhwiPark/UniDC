from .nn import *
import torch.nn as nn
import torch.nn.functional as F
from time import time

class HypMLP(nn.Module): 
    def __init__(self, in_channels, out_channels, return_euc_feature, config):
        super(HypMLP, self).__init__()

        self.c = config['curvature']

        if config['non_linearity']:
            self.nonlin = nn.ReLU(inplace=True)
        else:
            self.nonlin = None
        if config['batchnorm']:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None

        self.hypMLP = HypLinear(in_channels, out_channels, self.c, self.nonlin, bias=True)
        self.e2p = ToPoincare(self.c, train_c=False, clip=config['clipping'], clip_param=config['clip_param'])

        if return_euc_feature:
            self.return_euc_feature=True
            self.p2e = FromPoincare(self.c, train_c=False)
        else: self.return_euc_feature=False

    def forward(self, x, c=None):
        """
        x -> [B, C, H, W]
        """
        x = x.permute(0, 2, 3, 1)
        # x=self.p2e(self.hypMLP(self.e2p(x)))
        x = self.e2p(x, c)
        # print(torch.norm(x, dim = -1).min())
        # print(torch.norm(x, dim = -1).mean())
        # print(torch.norm(x, dim = -1).max())
        x = self.hypMLP(x)
        if self.return_euc_feature:
            x = self.p2e(x)
        x = x.permute(0, 3, 1, 2)
        if self.bn:
            return self.bn(x)
        else:
            return x
        
import pmath
class HypDistance(nn.Module): 
    def __init__(self, in_channels, out_channels, config):
        super(HypDistance, self).__init__()

        self.c = config['curvature']
        # self.sigmoid = nn.Sigmoid()
           
        if config['non_linearity']:
            self.nonlin = nn.ReLU(inplace=True)
        else:
            self.nonlin = None
        if config['batchnorm']:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None

        # self.hypMLP = HypLinear(in_channels, out_channels, self.c, self.nonlin, bias=True)
        self.e2p = ToPoincare(self.c, train_c=False, clip=False, clip_param=1.0)



    def forward(self, x, y):
        """
        x -> [B, C, H, W]
        """
        x = x.permute(0, 2, 3, 1)
        y = y.permute(0, 2, 3, 1)
        # x = self.e2p(x)
        dist = pmath.dist(x=self.e2p(x), y=self.e2p(y), c=self.c).unsqueeze(3)
        dist = dist.permute(0, 3, 1, 2)
        if self.bn:
            return self.bn(dist)
        else:
            return dist

