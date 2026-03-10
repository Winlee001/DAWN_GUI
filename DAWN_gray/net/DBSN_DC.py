import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
# from .blind_spot_conv import BlindSpotConv
from net.blind_spot_conv import BlindSpotConv
from util.utils import init_weights, weights_init_kaiming
from functools import partial


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


class DBSN_branch(nn.Module):
    def __init__(self, in_ch,inplanes, bs_conv_type, bs_conv_bias, bs_conv_ks, block_num, activate_fun):
        super(DBSN_branch, self).__init__()
        # 
        if activate_fun == 'Relu':
            # self.relu = nn.ReLU(inplace=True)
            self.relu = partial(nn.ReLU, inplace=True)
        elif activate_fun == 'LeakyRelu':
            # self.relu = nn.LeakyReLU(0.1)
            self.relu = partial(nn.LeakyReLU, negative_slope=0.1)
        else:
            raise ValueError('activate_fun [%s] is not found.' % (activate_fun))
        self.bs_conv_type = bs_conv_type
        #
        dilation_base=(bs_conv_ks+1)//2
        # self.strd = stride
        p = dilation_base - 1

        if self.bs_conv_type != 'Mask3D':
            # self.central_mask_conv = CentralMaskedConv2d(inplanes, inplanes, kernel_size=bs_conv_ks, stride=1, padding=p)
            self.central_mask_conv = BlindSpotConv(inplanes, inplanes, bs_conv_ks, stride=1, dilation=1, bias=bs_conv_bias, conv_type=bs_conv_type)
            self.other_normal_conv       = nn.Conv2d(inplanes,inplanes , kernel_size=bs_conv_ks, stride=1, padding=p)
            self.mid_1conv = nn.Conv2d(1, inplanes, kernel_size=1)
            self.other_1conv = nn.Conv2d(in_ch-1, inplanes, kernel_size=1)
            init_weights(self.other_1conv)
            init_weights(self.central_mask_conv)
            init_weights(self.other_normal_conv)
        else:
            self.mid_1conv = nn.ModuleList([nn.Sequential(torch.nn.Conv2d(1,inplanes,kernel_size=1),
                                            self.relu()) for _ in range(in_ch)])
            self.central_mask_conv = BlindSpotConv(inplanes,inplanes,(in_ch,bs_conv_ks,bs_conv_ks), stride=1, dilation=(0,1,1), bias=bs_conv_bias, conv_type=bs_conv_type,padding=(0,p,p))
            
        #
        lyr=[]
        # lyr.append(BlindSpotConv(inplanes, inplanes, bs_conv_ks, stride=1, dilation=1, bias=bs_conv_bias, conv_type=bs_conv_type))
        lyr.append(self.relu())
        if self.bs_conv_type != 'Mask3D':
            lyr.append(nn.Conv2d(inplanes*2, inplanes, kernel_size=1, bias=bs_conv_bias))
        else:
            lyr.append(nn.Conv2d(inplanes, inplanes, kernel_size=1, bias=bs_conv_bias))
        lyr.append(self.relu())
        lyr.append(nn.Conv2d(inplanes, inplanes, kernel_size=1, bias=bs_conv_bias))
        lyr.append(self.relu())
        #
        for i in range(block_num):
            lyr.append(Inception_block(inplanes, kernel_size=3, dilation=dilation_base, bias=bs_conv_bias, activate_fun=activate_fun))
        #
        lyr.append(nn.Conv2d(inplanes, inplanes, kernel_size=1, bias=bs_conv_bias))
        self.branch=nn.Sequential(*lyr)
        init_weights(self.mid_1conv)
        init_weights(self.branch)

    def central_frame_blind(self, x):
        B, T, H, W = x.shape
        outs = []
        if self.bs_conv_type != 'Mask3D':
            other_frame = [x[:, [i], :, :] for i in range(T) if i != T//2]
            other_frame = torch.cat(other_frame, dim=1)
            other_frame = self.relu()(self.other_1conv(other_frame))
            other_frame = self.other_normal_conv(other_frame)
            mid_frame = x[:, [T//2], :, :]
            mid_frame = self.relu()(self.mid_1conv(mid_frame))
            mid_frame = self.central_mask_conv(mid_frame)
            out = torch.cat([mid_frame,other_frame], dim=1)
        else:
            #batch_size channel frame height width
            x = torch.cat([torch.unsqueeze(self.mid_1conv[i](x[:, [i], :, :]),dim=2) for i in range(T)], dim=2)
            out = self.central_mask_conv(x)
            out = torch.squeeze(out,2)
        return out

    def forward(self,x):
        return self.branch(self.central_frame_blind(x)) 

class DBSN_DC(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch, 
                blindspot_conv_type, blindspot_conv_bias,
                br1_blindspot_conv_ks, br1_block_num, 
                br2_blindspot_conv_ks, br2_block_num,
                activate_fun):
        super(DBSN_DC,self).__init__()
        #
        if activate_fun == 'Relu':
            # self.relu = nn.ReLU(inplace=True)
            self.relu = partial(nn.ReLU, inplace=True)
        elif activate_fun == 'LeakyRelu':
            # self.relu = nn.LeakyReLU(0.1)
            self.relu = partial(nn.LeakyReLU, negative_slope=0.1)
        else:
            raise ValueError('activate_fun [%s] is not found.' % (activate_fun))
        # # Head of DBSN
        # lyr = []
        # lyr.append(nn.Conv2d(in_ch, mid_ch, kernel_size=1, bias=blindspot_conv_bias))
        # lyr.append(self.relu())
        # self.dbsn_head = nn.Sequential(*lyr)
        # init_weights(self.dbsn_head)
        
        if blindspot_conv_type == 'MulMask':
            self.br1 = DBSN_branch(in_ch,mid_ch,'MulMask', blindspot_conv_bias, br1_blindspot_conv_ks, br1_block_num, activate_fun)
            self.br2 = DBSN_branch(in_ch,mid_ch,'Mask', blindspot_conv_bias, br2_blindspot_conv_ks, br2_block_num, activate_fun )
            self.br3 = DBSN_branch(in_ch,mid_ch,'MulMask', blindspot_conv_bias, br1_blindspot_conv_ks, br1_block_num, activate_fun )
            self.br4 = DBSN_branch(in_ch,mid_ch,'Mask', blindspot_conv_bias, br2_blindspot_conv_ks, br2_block_num, activate_fun )
        else:
            self.br1 = DBSN_branch(in_ch,mid_ch, blindspot_conv_type, blindspot_conv_bias, br1_blindspot_conv_ks, br1_block_num, activate_fun)
            self.br2 = DBSN_branch(in_ch,mid_ch, blindspot_conv_type, blindspot_conv_bias, br2_blindspot_conv_ks, br2_block_num, activate_fun)
            self.br3 = DBSN_branch(in_ch,mid_ch, blindspot_conv_type, blindspot_conv_bias, br1_blindspot_conv_ks, br1_block_num, activate_fun)
            self.br4 = DBSN_branch(in_ch,mid_ch, blindspot_conv_type, blindspot_conv_bias, br2_blindspot_conv_ks, br2_block_num, activate_fun)

        # Concat two branches
        self.concatx = nn.Conv2d(mid_ch*2,mid_ch,kernel_size=1,bias=blindspot_conv_bias)
        self.concatx.apply(weights_init_kaiming) #apply的操作是让模型的所有层递归的使用该初始化函数进行初始化
        self.concaty = nn.Conv2d(mid_ch*2,mid_ch,kernel_size=1,bias=blindspot_conv_bias)
        self.concaty.apply(weights_init_kaiming)
        # 1x1 convs
        lyr=[]
        lyr.append(nn.Conv2d(mid_ch,mid_ch,kernel_size=1,bias=blindspot_conv_bias))
        lyr.append(self.relu())
        lyr.append(nn.Conv2d(mid_ch,mid_ch,kernel_size=1,bias=blindspot_conv_bias))
        lyr.append(self.relu())
        lyr.append(nn.Conv2d(mid_ch,out_ch,kernel_size=1,bias=blindspot_conv_bias))
        self.dbsn_tailx=nn.Sequential(*lyr)
        init_weights(self.dbsn_tailx)
                # 1x1 convs
        lyt=[]
        lyt.append(nn.Conv2d(mid_ch,mid_ch,kernel_size=1,bias=blindspot_conv_bias))
        lyt.append(self.relu())
        lyt.append(nn.Conv2d(mid_ch,mid_ch,kernel_size=1,bias=blindspot_conv_bias))
        lyt.append(self.relu())
        lyt.append(nn.Conv2d(mid_ch,out_ch,kernel_size=1,bias=blindspot_conv_bias))
        self.dbsn_taily=nn.Sequential(*lyt)
        init_weights(self.dbsn_taily)

    def forward(self, x,y):
        # x = self.dbsn_head(x)
        x1 = self.br1(x)     
        x2 = self.br2(x)
        x_concat = torch.cat((x1,x2), dim=1)
        x = self.concatx(x_concat)
        y1 = self.br3(y)
        y2 = self.br4(y)
        y_concat = torch.cat((y1,y2), dim=1)
        y = self.concaty(y_concat)
        return self.dbsn_tailx(x),self.dbsn_taily(y)  
    
if __name__ == '__main__':
    model = DBSN_DC(in_ch = 3,
                        out_ch = 1,
                        mid_ch = 96,
                        blindspot_conv_type = 'Mask3D',
                        blindspot_conv_bias = True,
                        br1_block_num = 8,
                        br1_blindspot_conv_ks =3,
                        br2_block_num = 8,
                        br2_blindspot_conv_ks = 5,
                        activate_fun = 'Relu').cuda()
    # 注意传入两个输入形状的列表
    summary(model, input_size=[(8, 3, 96, 96), (8, 3, 96, 96)])
