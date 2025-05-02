from torch import nn
BatchNorm2d = nn.BatchNorm2d
import torch
import torch.nn as nn
import torch.nn.functional as F
class MC_CFP(nn.Module):
    def __init__(self, classes=2, block_1=1, block_2=3):
        super(MC_CFP,self).__init__()

        self.init_conv_0 = DeConv(3, 32, 3, 2, padding=1, bn_acti=True,groups=1)

        self.init_conv_1 = nn.Sequential(MixConv(32, 192, 3, 2 ,padding=1,num_groups=4),
                                         DeConv(192, 32, 1, 1, padding=0, bn_acti=True, groups=1))
        self.down_1 = InputInjection(2)
        self.down_2 = InputInjection(3)
        self.down_3 = InputInjection(4) 

        self.bn_prelu_1 = BNPReLU(32 + 3)
        dilation_block_1 = [2]

        self.downsample_1 = MC(32 + 3, 64)
        self.CFP_Block_1 = nn.Sequential()
        for i in range(0, block_1):
            self.CFP_Block_1.add_module("CFP_Module_1_" + str(i), CFPModule(64, d=dilation_block_1[i]))

        self.bn_prelu_2 = BNPReLU(64 + 3)

        dilation_block_2 = [4, 4, 8]
        self.downsample_2 = MC(64 + 3, 128)
        self.CFP_Block_2 = nn.Sequential()
        for i in range(0, block_2):
            self.CFP_Block_2.add_module("CFP_Module_2_" + str(i),
                                        CFPModule(128, d=dilation_block_2[i]))
        self.bn_prelu_3 = BNPReLU(128 + 3)
        self.DSC_ = DSC(128+3, 256, 3, 1, padding=1, bn_acti=True)
        self.final_conv = nn.Sequential(DSC(256, 512, 3, 1, padding=1, bn_acti=True),
            DeConv(512, 256, 1, 1, padding=0, bn_acti=True, groups=1))

    def forward(self, input):
        output00 = self.init_conv_0(input)#32,256,256
        output01 = self.init_conv_1(output00)#32,128,128
        down_1 = self.down_1(input)
        down_2 = self.down_2(input)
        down_3 = self.down_3(input)
        output0_cat = self.bn_prelu_1(torch.cat([output01, down_1], 1))
        output1_0 = self.downsample_1(output0_cat)
        output1 = self.CFP_Block_1(output1_0)#64,64,64
        output1_cat = self.bn_prelu_2(torch.cat([output1, down_2], 1))#, output1_0
        output2_0 = self.downsample_2(output1_cat)
        output2 = self.CFP_Block_2(output2_0)#128,32,32
        output2_cat = self.bn_prelu_3(torch.cat([output2, down_3], 1))
        out = self.DSC_(output2_cat)
        out = self.final_conv(out)

        return output00, output01, output1, output2, out