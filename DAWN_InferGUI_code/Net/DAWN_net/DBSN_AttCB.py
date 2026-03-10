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
from net.backbone_net import Inception_block
from DAWN_net.utils import init_weights, weights_init_kaiming
from functools import partial

class ConvBlock(nn.Module):
    def __init__(self, out_ch,in_ch = 1,use_1x1_preconv=False):
        super().__init__()
        if use_1x1_preconv:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1),
                nn.ReLU(),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.ReLU(),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.ReLU()
            )
    def forward(self, x):
        return self.conv(x)

class ContextAttention(nn.Module):
    def __init__(self, embed_ch,in_ch=1,frames=3):
        super().__init__()
        self.encoder = ConvBlock(out_ch=embed_ch,in_ch=in_ch)
        self.query_conv = nn.Conv2d((frames-1)*embed_ch, embed_ch, 1)
        self.key_conv = nn.Conv2d((frames-1)*embed_ch, embed_ch, 1)
        self.value_conv = nn.Conv2d((frames-1)*embed_ch, embed_ch, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, neighbor_frames):
        # neighbor_frames: [B, 4, H, W] --> list of 4 tensors
        B, N, H, W = neighbor_frames.shape
        neighbors = neighbor_frames.view(B * N, H, W)
        neighbors = neighbors.unsqueeze(1)  # [B*N, 1, H, W]
        feat = self.encoder(neighbors)  # [B*N, embed_ch, H, W]
        feat = feat.view(B, N, -1, H, W)  # [B, N, C, H, W]

        # # Mean across neighbors to get a context representation
        # context = feat.mean(dim=1)  # [B, C, H, W]

        # Compute attention
        feat_cat = feat.permute(0, 2, 1, 3, 4).reshape(B, -1, H, W)  # [B, 4*C, H, W]
        Q = self.query_conv(feat_cat).flatten(2)  # [B, C, HW]
        K = self.key_conv(feat_cat).flatten(2)
        V = self.value_conv(feat_cat).flatten(2)


        attn = torch.bmm(Q.transpose(1, 2), K) / (Q.shape[1] ** 0.5)  # [B, HW, HW]
        attn = self.softmax(attn)
        context_feat = torch.bmm(attn, V.transpose(1, 2)).transpose(1, 2)  # [B, C, HW]
        context_feat = context_feat.view(B, -1, H, W)  # [B, C, H, W]

        return context_feat

class BlindSpotConv_set(nn.Module):
    def __init__(self, out_ch,in_ch = 1,use_1x1_preconv=False):
        super().__init__()
        if use_1x1_preconv:
            self.BSconv =nn.Sequential(nn.Conv2d(in_ch,out_ch, 1),
                        nn.ReLU(inplace=True),
                        BlindSpotConv(out_ch, out_ch, 3, stride=1,dilation=1, bias=True,conv_type='Mask'),
                        nn.ReLU(inplace=True)
                        )
        else:
            self.BSconv = nn.Sequential(
                BlindSpotConv(in_ch, out_ch, 3, stride=1, dilation=1, bias=True, conv_type='Mask'),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1,dilation=1),
                nn.ReLU(inplace=True)
            )
    def forward(self, x):
        x = self.BSconv(x)
        return x

class Fusion_AttBlindSpot(nn.Module):
    def __init__(self, in_ch=1, embed_ch=32,frames=3):
        super().__init__()
        self.context_attn = ContextAttention(embed_ch= embed_ch,frames=frames)
        self.blindspot = BlindSpotConv_set(out_ch=embed_ch, use_1x1_preconv=False)
        self.fusion = nn.Conv2d(embed_ch * 2, embed_ch, 1)

    def forward(self, x):
        # if len(x.shape) == 4:
        #     x = x.unsqueeze(2)  # Add a dummy dimension for single frame input
        # x: [B, 5, H, W]  -> assume 4-frame input
        B, T, H, W = x.shape
        center_frame = x[:, T//2]  # [B, H, W]
        center_frame = center_frame.unsqueeze(1)  # [B, 1, H, W]

        context = self.context_attn(x[:, [i for i in range(T) if i != T//2]])  # [B, C, H, W]
        blind_feat = self.blindspot(center_frame)     # [B, C, H, W]

        fused = torch.cat([context, blind_feat], dim=1)  # [B, 2C, H, W]
        fused = self.fusion(fused)                       # [B, C, H, W]

        return fused

class Inception_branch(nn.Module):
    def __init__(self, inplanes, bs_conv_type, bs_conv_bias, bs_conv_ks, block_num, activate_fun):
        super().__init__()
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
        #
        lyr=[]
        # lyr.append(BlindSpotConv(inplanes, inplanes, bs_conv_ks, stride=1, dilation=1, bias=bs_conv_bias, conv_type=bs_conv_type))
        #lyr.append(self.relu())
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
        init_weights(self.branch)


    def forward(self,x):
        return self.branch(x)     
   

class DBSN_AttCB(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch, 
                blindspot_conv_type, blindspot_conv_bias,
                br1_blindspot_conv_ks, br1_block_num, 
                br2_blindspot_conv_ks, br2_block_num,
                activate_fun):
        super(DBSN_AttCB,self).__init__()
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
        self.AttCB = Fusion_AttBlindSpot(embed_ch=mid_ch,frames=in_ch)
        self.br1 = Inception_branch(mid_ch, blindspot_conv_type, blindspot_conv_bias, br1_blindspot_conv_ks, br1_block_num, activate_fun)
        self.br2 = Inception_branch(mid_ch, blindspot_conv_type, blindspot_conv_bias, br2_blindspot_conv_ks, br2_block_num, activate_fun)

        # Concat two branches
        self.concat = nn.Conv2d(mid_ch*2,mid_ch,kernel_size=1,bias=blindspot_conv_bias)
        self.concat.apply(weights_init_kaiming) #apply的操作是让模型的所有层递归的使用该初始化函数进行初始化
        # 1x1 convs
        lyr=[]
        lyr.append(nn.Conv2d(mid_ch,mid_ch,kernel_size=1,bias=blindspot_conv_bias))
        lyr.append(self.relu())
        lyr.append(nn.Conv2d(mid_ch,mid_ch,kernel_size=1,bias=blindspot_conv_bias))
        lyr.append(self.relu())
        lyr.append(nn.Conv2d(mid_ch,out_ch,kernel_size=1,bias=blindspot_conv_bias))
        self.dbsn_tail=nn.Sequential(*lyr)
        init_weights(self.AttCB)
        init_weights(self.dbsn_tail)

    def forward(self, x):
        x = self.AttCB(x)
        x1 = self.br1(x)     
        x2 = self.br2(x)
        x_concat = torch.cat((x1,x2), dim=1)
        x = self.concat(x_concat)
        return self.dbsn_tail(x), x #前者是DBSN最终输出，即mu输出，后者是连接后的特征图向量

if __name__ == '__main__':
    model = DBSN_AttCB(in_ch = 3,
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
    summary(model, input_size=[(8, 3, 96, 96)])

