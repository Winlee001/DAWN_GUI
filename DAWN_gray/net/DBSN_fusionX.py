import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from util.utils import init_weights, weights_init_kaiming
from functools import partial
from torchinfo import summary


 
class CentralMaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH//2, kH//2] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)
      

class DC_branch_head(nn.Module):
    def __init__(self, stride, in_ch):
        super().__init__()

        ly = []
        ly += [ CentralMaskedConv2d(in_ch, in_ch, kernel_size=2*stride-1, stride=1, padding=stride-1) ]
        # ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=2*stride-1, padding=stride-1)]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        self.head = nn.Sequential(*ly)  
        init_weights(self.head)
    def forward(self, x):
        return self.head(x)

class DC_branch_tail(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        ly = []
        ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ] 
        self.tail = nn.Sequential(*ly)
        init_weights(self.tail)
    def forward(self, x):
        return self.tail(x)



class Inception_block(nn.Module):
    def __init__(self, inplanes, kernel_size, dilation, bias, activate_fun):
        super(Inception_block, self).__init__()
        #
        if activate_fun == 'Relu':
            # self.relu = nn.ReLU(inplace=True)
            self.relu = partial(nn.ReLU, inplace=True)
        elif activate_fun == 'LeakyRelu':
            # self.relu = nn.LeakyReLU(0.1)
            self.relu = partial(nn.LeakyReLU, negative_slope=0.1)
        else:
            raise ValueError('activate_fun [%s] is not found.' % (activate_fun))
        #这个pad_size是为了使得每次卷积特征图一样
        pad_size = (kernel_size+(kernel_size-1)*(dilation-1)-1)//2
        # inception_br1 ----------------------------------------------
        lyr_br1=[]
        # 1x1 conv
        lyr_br1.append(nn.Conv2d(inplanes, inplanes, kernel_size=1, bias=bias))
        lyr_br1.append(self.relu())
        # # case1: two 3x3 dilated-conv
        # lyr_br1.append(nn.Conv2d(inplanes, inplanes, kernel_size, padding=pad_size, dilation=dilation, bias=bias))
        # lyr_br1.append(self.relu())
        # lyr_br1.append(nn.Conv2d(inplanes, inplanes, kernel_size, padding=pad_size, dilation=dilation, bias=bias))
        # lyr_br1.append(self.relu())
        # case2: one 5x5 dilated-conv
        tmp_kernel_size = 5
        tmp_pad_size = (tmp_kernel_size+(tmp_kernel_size-1)*(dilation-1)-1)//2
        lyr_br1.append(nn.Conv2d(inplanes, inplanes, kernel_size=tmp_kernel_size, padding=tmp_pad_size, dilation=dilation, bias=bias))
        lyr_br1.append(self.relu())
        self.inception_br1=nn.Sequential(*lyr_br1)
        init_weights(self.inception_br1)
        #
        # inception_br2 ----------------------------------------------
        lyr_br2=[]
        # 1x1 conv
        lyr_br2.append(nn.Conv2d(inplanes, inplanes, kernel_size=1, bias=bias))
        lyr_br2.append(self.relu())
        # 3x3 dilated-conv
        lyr_br2.append(nn.Conv2d(inplanes, inplanes, kernel_size, padding=pad_size, dilation=dilation, bias=bias))
        lyr_br2.append(self.relu())
        self.inception_br2=nn.Sequential(*lyr_br2)
        init_weights(self.inception_br2)
        #
        # inception_br3 ----------------------------------------------
        lyr_br3=[]
        # 1x1 conv
        lyr_br3.append(nn.Conv2d(inplanes, inplanes, kernel_size=1, bias=bias))
        lyr_br3.append(self.relu())
        self.inception_br3=nn.Sequential(*lyr_br3)
        init_weights(self.inception_br3)
        # Concat three inception branches
        self.concat = nn.Conv2d(inplanes*3,inplanes,kernel_size=1,bias=bias)
        self.concat.apply(weights_init_kaiming)
        # 1x1 convs
        lyr=[]
        lyr.append(nn.Conv2d(inplanes,inplanes,kernel_size=1,bias=bias))
        lyr.append(self.relu())
        lyr.append(nn.Conv2d(inplanes,inplanes,kernel_size=1,bias=bias))
        lyr.append(self.relu())
        self.middle_1x1_convs=nn.Sequential(*lyr)
        init_weights(self.middle_1x1_convs)
  

    def forward(self, x):
        residual = x
        x1 = self.inception_br1(x)
        x2 = self.inception_br2(x)
        x3 = self.inception_br3(x)
        out = torch.cat((x1, x2, x3), dim=1)
        out = self.concat(out)
        out = torch.relu_(out)
        out = out + residual
        out = self.middle_1x1_convs(out)
        return out



class DBSN_fusionX(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch, 
            blindspot_conv_type, blindspot_conv_bias,
            br1_blindspot_conv_ks, br1_block_num, 
            br2_blindspot_conv_ks, br2_block_num,
            activate_fun):
 
        super().__init__()

        assert mid_ch%2 == 0, "mid channel should be divided with 2"
        assert br1_block_num%2 == 0, "br1_block_num should be divided with 2"
        assert br2_block_num%2 == 0, "br2_block_num should be divided with 2"
        assert br1_block_num == br2_block_num, "br1_block_num should be equal to br2_block_num"
        br1_dilation_base=(br1_blindspot_conv_ks+1)//2
        br2_dilation_base=(br2_blindspot_conv_ks+1)//2

        self.Mod_num = br1_block_num
        # ly = []
        # ly += [ nn.Conv2d(in_ch, mid_ch, kernel_size=1) ]
        # ly += [ nn.ReLU(inplace=True) ]
        self.headx = nn.Sequential( nn.Conv2d(in_ch, mid_ch, kernel_size=1),
                                    nn.ReLU(inplace=True) )
        self.heady = nn.Sequential( nn.Conv2d(in_ch, mid_ch, kernel_size=1),
                                    nn.ReLU(inplace=True) )
        init_weights(self.headx)
        init_weights(self.heady)
        #TODO
        self.branch_head_x1 = DC_branch_head(br1_dilation_base, mid_ch)
        self.branch_head_x2 = DC_branch_head(br2_dilation_base, mid_ch)
        self.branch_head_y1 = DC_branch_head(br1_dilation_base, mid_ch)
        self.branch_head_y2 = DC_branch_head(br2_dilation_base, mid_ch)

        self.x1_DCL = nn.ModuleList(Inception_block(mid_ch,kernel_size=3,dilation=br1_dilation_base,bias=blindspot_conv_bias,activate_fun=activate_fun) for i in range(self.Mod_num))
        self.x2_DCL = nn.ModuleList(Inception_block(mid_ch,kernel_size=3,dilation=br2_dilation_base,bias=blindspot_conv_bias,activate_fun=activate_fun) for i in range(self.Mod_num))
        self.y1_DCL = nn.ModuleList(Inception_block(mid_ch,kernel_size=3,dilation=br1_dilation_base,bias=blindspot_conv_bias,activate_fun=activate_fun) for i in range(self.Mod_num))
        self.y2_DCL = nn.ModuleList(Inception_block(mid_ch,kernel_size=3,dilation=br2_dilation_base,bias=blindspot_conv_bias,activate_fun=activate_fun) for i in range(self.Mod_num))
        # self.conv1x1 = nn.Conv2d(mid_ch*2, mid_ch, kernel_size=1)
        # self.conv1x1.apply(weights_init_kaiming)
        self.xy1_conv1x1 = nn.ModuleList()
        self.xy2_conv1x1 = nn.ModuleList()
        for i in range(self.Mod_num//2):
            self.xy1_conv1x1.append(nn.Conv2d(mid_ch*2, mid_ch, kernel_size=1))
            self.xy1_conv1x1[i].apply(weights_init_kaiming)
            self.xy2_conv1x1.append(nn.Conv2d(mid_ch*2, mid_ch, kernel_size=1))
            self.xy2_conv1x1[i].apply(weights_init_kaiming)


        self.branch_tail_x1 = DC_branch_tail(mid_ch)
        self.branch_tail_x2 = DC_branch_tail(mid_ch)
        self.branch_tail_y1 = DC_branch_tail(mid_ch)
        self.branch_tail_y2 = DC_branch_tail(mid_ch)

        self.tailx = nn.Sequential(
            nn.Conv2d(mid_ch*2,  mid_ch,    kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch,    mid_ch//2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch//2, mid_ch//2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch//2, out_ch,    kernel_size=1)
            )
        self.taily = nn.Sequential(
            nn.Conv2d(mid_ch*2,  mid_ch,    kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch,    mid_ch//2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch//2, mid_ch//2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch//2, out_ch,    kernel_size=1)
        )
        init_weights(self.tailx)
        init_weights(self.taily)    

    def forward(self, x,y):
        x = self.headx(x)
        y = self.heady(y)

        x1 = self.branch_head_x1(x)
        x2 = self.branch_head_x2(x)
        y1 = self.branch_head_y1(y)
        y2 = self.branch_head_y2(y)

        xy1_concat = []
        xy2_concat = []
        #self.Mod_num must be even/encoder
        # print(self.Mod_num)
        for i in range(self.Mod_num//2):
            x1 = self.x1_DCL[i](x1)
            y1 = self.y1_DCL[i](y1)
            xy1_concat.append(torch.cat([x1, y1], dim=1))
            
            x2 = self.x2_DCL[i](x2)
            y2 = self.y2_DCL[i](y2)
            xy2_concat.append(torch.cat([x2, y2], dim=1))
        #1x1 conv for fusion and channel reduction
        for i in range(len(xy1_concat)):
            xy1_concat[i] = self.xy1_conv1x1[i](xy1_concat[i])
            xy2_concat[i] = self.xy2_conv1x1[i](xy2_concat[i])
        # decoder
        for j in range(self.Mod_num//2):
        #     print(j,len(xy1_concat),len(xy2_concat),len(self.x1_DCL),len(self.y1_DCL),len(self.x2_DCL),len(self.y2_DCL))
            x1 = self.x1_DCL[j+self.Mod_num//2](x1+xy1_concat[self.Mod_num//2-j-1])
            y1 = self.y1_DCL[j+self.Mod_num//2](y1+xy1_concat[self.Mod_num//2-j-1])
            x2 = self.x2_DCL[j+self.Mod_num//2](x2+xy2_concat[self.Mod_num//2-j-1])
            y2 = self.y2_DCL[j+self.Mod_num//2](y2+xy2_concat[self.Mod_num//2-j-1])
        #tail
        x = torch.cat([x1, x2], dim=1)
        y = torch.cat([y1, y2], dim=1)

        return self.tailx(x),self.taily(y)

    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)

if __name__ == '__main__':
    model = DBSN_fusionX(in_ch = 5,
                        out_ch = 1,
                        mid_ch = 96,
                        blindspot_conv_type = 'Mask',
                        blindspot_conv_bias = True,
                        br1_block_num = 8,
                        br1_blindspot_conv_ks =3,
                        br2_block_num = 8,
                        br2_blindspot_conv_ks = 5,
                        activate_fun = 'Relu',
                        ).cuda()
    # 注意传入两个输入形状的列表
    summary(model, input_size=[(8, 5, 96, 96), (8, 5, 96, 96)])
