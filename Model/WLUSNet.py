from torch import nn
BatchNorm2d = nn.BatchNorm2d
import torch
import torch.nn as nn
import torch.nn.functional as F
class WLUSNet(nn.Module):
    def __init__(self,classes, block_1, block_2,backbone,downsample_factor, pretrained):
        super(WLUSNet, self).__init__()
        self.MC_CFP = MC_CFP(classes, block_1, block_2).cuda()
        self.MC_SP = MC_SP(256,128,1).cuda()
        self.bn_prelu_4 =BNPReLU(160)
        self.EDFF_10 = EDFF(128,64)
        self.EDFF_11 = EDFF(64, 32)
        self.EDFF_12 = EDFF(32, 32)
        self.b40 = nn.Conv2d(256,128,1,1)
        self.b4 = nn.Conv2d(32,classes,1,1)
    def forward(self,img):
        output00, output01, output1, output2, out = self.MC_CFP(img)
        hight_level_Feature = self.MC_SP(out)
        hight_level_Feature = self.b40(torch.cat((hight_level_Feature,output2),dim=1))
        b0 = self.EDFF_10(output1,hight_level_Feature)
        b0 = self.EDFF_11(output01,b0 )
        b0 = self.EDFF_12( output00,b0)
        out = self.b4(b0)
        out = F.interpolate(out, (512, 512), None, 'bilinear', True)
        return out
