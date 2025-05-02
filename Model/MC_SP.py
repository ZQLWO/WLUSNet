from torch import nn
BatchNorm2d = nn.BatchNorm2d
import torch
import torch.nn as nn
import torch.nn.functional as F
class MC_SP(nn.Module):
    def __init__(self, in_channel=256, out_channel=128, rate=1, bn_mom=0.1):
        super(MC_SP, self).__init__()
        self.conv_1 = nn.Sequential(nn.Conv2d(in_channel, in_channel, 1, 1),
                                    nn.BatchNorm2d(in_channel, momentum=bn_mom),nn.PReLU())
        self.conv_cat = nn.Sequential(
            nn.Conv2d(768, 128, 1, 1, bias=True),
            nn.BatchNorm2d(128, momentum=bn_mom),
            nn.PReLU())
        self.CAM=CAM(256)
        self.MixConv = MixConv(in_channel,in_channel,3,1,1,4)
    def forward(self, x0):
        mixconv = self.MixConv(x0)
        x1 = self.conv_1(x0)
        x5 = self.CAM(x0)
        out = torch.cat([mixconv, x1, x5], dim=1)
        result = self.conv_cat(out)
        return result