from .nn import *
import torch.nn as nn
import torch.nn.functional as F
from time import time

class HypCNN(nn.Module): 
    def __init__(self, in_channels, out_channels, c, train_c=False, hyp_mode='naive', norm_layer=False, act=None):
        super(HypCNN, self).__init__()

        self.c = c
        self.sigmoid = nn.Sigmoid()
           
        if act is not None:
            self.nonlin = nn.ReLU(inplace=True)
        else:
            self.nonlin = None

        if norm_layer:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None

        self.hypConv = HALM(in_channels, out_channels, self.c, kernel=3, padding=1, strides=1, dilation=1, nonlin=self.nonlin, train_c=train_c, mode=hyp_mode)
        
    def forward(self, x):

        x_ = self.hypConv(x)  # torch.Size([1, 228, 304, 8])

        if self.bn:
            return self.bn(self.sigmoid(x_))
        else:
            return self.sigmoid(x_)
        

class HypCNN_curvature_generator(nn.Module):  # torch.Size([1, 64, 114, 152] ->  torch.Size([1, 8, 228, 304])
    def __init__(self, in_channels, out_channels, train_c=False, hyp_mode='naive', norm_layer=False, act=None):
        super(HypCNN_curvature_generator, self).__init__()

        self.sigmoid = nn.Sigmoid()
           
        if act is not None:
            self.nonlin = nn.ReLU(inplace=True)
        else:
            self.nonlin = None

        if norm_layer:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None

        self.hypConv = HALM_curvature_generator(in_channels, out_channels, kernel=3, padding=1, strides=1, dilation=1, nonlin=self.nonlin, train_c=train_c, mode=hyp_mode)
        
    def forward(self, x, c):

        x_ = self.hypConv(x, c)  # torch.Size([1, 228, 304, 8])

        if self.bn:
            return self.bn(self.sigmoid(x_))
        else:
            return self.sigmoid(x_)

