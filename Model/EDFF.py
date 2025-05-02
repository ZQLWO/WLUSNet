from torch import nn
BatchNorm2d = nn.BatchNorm2d
import torch
import torch.nn as nn
import torch.nn.functional as F
class EDFF(nn.Module):

    def __init__(self, in_channel, out_channel, upconv=False):
        super(EDFF, self).__init__()

        self.con00 = nn.Sequential(nn.Conv2d(in_channel,out_channel,1,1),
                                   nn.BatchNorm2d(out_channel),
                                   nn.ReLU(inplace=True),)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channel*2, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_prev, x):
        x = self.con00(x)
        x = F.interpolate(x, (x_prev.shape[2], x_prev.shape[2]), None, 'bilinear', True)

        x = torch.cat((x_prev, x), dim=1)
        x = self.conv(x)
        return x