from imp import init_frozen
from numpy import outer
import torch
import torch.nn as nn
import torch.functional as F
from einops import rearrange
from basic_models import *

class ConvBlock(nn.Module):
    def __init__(self,in_channels, out_channels, kernel = 3, pad = 1,activation='None'):
        super(ConvBlock, self).__init__()

        m = [nn.Conv2d(in_channels, out_channels, kernel, stride=1, padding=pad)]
        #if activation != 'None':
        m.append(nn.LeakyReLU())
        self.models = nn.Sequential(*m)

    def forward(self, x):
        out = self.models(x)
        return out

class UpBlock(nn.Module):
    def __init__(self,filters, scale):
        super(UpBlock, self).__init__()
        
        m = []
        for _ in range(1, scale):
            m.append(DLKCB(filters, 4*filters, pad = 4))
            #m.append(nn.PixelShuffle(2))
            m.append(nn.ConvTranspose2d(4*filters, filters, 3, 2))

        self.model = nn.Sequential(*m)

    def forward(self, x):
        out = self.model(x)
        return out

class ResBlock(nn.Module):
    def __init__(self,filters, bottleneck):
        super(ResBlock, self).__init__()
        if filters == bottleneck:
            m=[ConvBlock(filters, filters,activation = 'Relu'), # with Relu
               ConvBlock(filters, filters)]
        else:
            m=[ConvBlock(filters, bottleneck,activation = 'Relu'),ConvBlock(bottleneck,bottleneck),ConvBlock(bottleneck,filters)]
        self.model = nn.Sequential(*m)
    def forward(self, x):

        res = x
        out = self.model(x)
        #out = torch.mul(out, 0.1)
        out = torch.add(out, res)

        return out

class SkippedBlock(nn.Module):
    def __init__(self, filters,bottleneck, n_resblock):
        super(SkippedBlock, self).__init__()

        m = []
        for _ in range(1, n_resblock):
            m.append(ResBlock(filters, bottleneck))
            m.append(nn.Dropout2d())
        self.resblocks = nn.Sequential(*m)
        self.conv = DLKCB(filters, filters,kernel=7, pad=24)

    def forward(self, x):
        res = x

        out = self.resblocks(x)
        out = self.conv(out)
        out = torch.add(out, res)
        return out

class SuperResolution(nn.Module):
    def __init__(self, filters, bottleneck, n_resblock, scale):
        super(SuperResolution,self).__init__()

        #self.conv = DLKCB(3, filters, pad=4)
        self.conv = ConvBlock(3,filters)
        self.skipped = SkippedBlock(filters,bottleneck,n_resblock)  # bottleneck is for if a bottleneck structure should be used and sets the filtersize of the bottleneck
        self.up = UpBlock(filters, scale)
        self.activ = nn.ReLU()
        self.conv3 = ConvBlock(filters,3, pad=0)

    def forward(self,x):
        out = self.conv(x)
        out = self.skipped(out)
        out = self.up(out)
        out = F.pad(out, (0,1,0,1))
        out = self.conv3(out)
        x = torch.div(x, x.max())
        return out

###############
#   https://arxiv.org/pdf/1707.02921v1.pdf EDSR

