import os
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from net.blind_spot_conv import BlindSpotConv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from DAWN_net.utils import init_weights, weights_init_kaiming
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
      

# class DC_branch_head(nn.Module):
#     def __init__(self, stride, in_ch):
#         super().__init__()

#         ly = []
#         ly += [ CentralMaskedConv2d(in_ch, in_ch, kernel_size=2*stride-1, stride=1, padding=stride-1) ]
#         # ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=2*stride-1, padding=stride-1)]
#         ly += [ nn.ReLU(inplace=True) ]
#         ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=1) ]
#         ly += [ nn.ReLU(inplace=True) ]
#         ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=1) ]
#         ly += [ nn.ReLU(inplace=True) ]
#         self.head = nn.Sequential(*ly)  
#         init_weights(self.head)
#     def forward(self, x):
#         return self.head(x)

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
# class DC_branch_head(nn.Module):
#     def __init__(self, stride, in_ch,mid_ch,when_fusion='early'):
#         super().__init__()
#         self.relu = nn.ReLU(inplace=True)
#         self.when_fusion = when_fusion
#         # self.strd = stride
#         k = 2*stride - 1
#         p = stride - 1
#         self.central_mask_conv = CentralMaskedConv2d(mid_ch, mid_ch, kernel_size=k, stride=1, padding=p)
#         self.other_normal_conv       = nn.Conv2d(mid_ch,mid_ch , kernel_size=k, stride=1, padding=p)
#         self.mid_1conv = nn.Conv2d(1, mid_ch, kernel_size=1)
#         self.other_1conv = nn.Conv2d(in_ch-1, mid_ch, kernel_size=1)


#         ly = []
#         ly += [ nn.Conv2d(2*mid_ch, mid_ch, kernel_size=1) ]
#         ly += [ nn.ReLU(inplace=True) ]
#         ly += [ nn.Conv2d(mid_ch, mid_ch, kernel_size=1) ]
#         ly += [ nn.ReLU(inplace=True) ]
#         self.head = nn.Sequential(*ly)  
#         # 初始化所有子模块
#         init_weights(self.mid_1conv)
#         init_weights(self.central_mask_conv)
#         init_weights(self.other_normal_conv)
#         init_weights(self.other_1conv)
#         init_weights(self.head)

#     def central_frame_blind(self, x):
#         B, T, H, W = x.shape
#         outs = []
#         if self.when_fusion == 'late':
#             raise ValueError('late is not supported')
#             # for i in range(T):
#             #     xi = x[:, [i], :, :]      # shape (B,1,H,W)
#             #     xi = F.relu(self.x1_conv(xi))     # shape (B,mid_ch,H,W)
#             #     if i == T//2:
#             #         yi = self.central_mask_conv_x(xi)
#             #     else:
#             #         yi = self.normal_conv_x(xi)
#             #     outs.append(F.relu(yi))           # 收集所有帧处理后的输出

#             # 将 5 帧的结果重新拼成 (B,T*mid_ch,H,W)
#             out = torch.cat(outs, dim=1)
#         elif self.when_fusion == 'early':
#             other_frame = [ x[:, [i], :, :] for i in range(T) if i != T//2]
#             other_frame = torch.cat(other_frame, dim=1)
#             other_frame = self.relu(self.other_1conv(other_frame))
#             other_frame = self.other_normal_conv(other_frame)
#             mid_frame = x[:, [T//2], :, :]
#             mid_frame = self.relu(self.mid_1conv(mid_frame))
#             mid_frame = self.central_mask_conv(mid_frame)
#             out = torch.cat([mid_frame,other_frame], dim=1)
#         return out
#     def forward(self, x):
#         out1 = self.central_frame_blind(x)
#         out2 = self.head(out1)
#         return out2


class DC_branch_head(nn.Module):
    def __init__(self, stride, in_ch,mid_ch,when_fusion='early',bs_conv_type='Mask3D'):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.when_fusion = when_fusion
        self.bs_conv_type = bs_conv_type
        # self.strd = stride
        k = 2*stride - 1
        p = stride - 1
        if self.bs_conv_type != 'Mask3D':
            self.central_mask_conv = CentralMaskedConv2d(mid_ch, mid_ch, kernel_size=k, stride=1, padding=p)
            self.other_normal_conv       = nn.Conv2d(mid_ch,mid_ch , kernel_size=k, stride=1, padding=p)
            self.mid_1conv = nn.Conv2d(1, mid_ch, kernel_size=1)
            self.other_1conv = nn.Conv2d(in_ch-1, mid_ch, kernel_size=1)
            init_weights(self.central_mask_conv)
            init_weights(self.other_normal_conv)
            init_weights(self.other_1conv)
        else:
            self.mid_1conv = nn.ModuleList([nn.Sequential(torch.nn.Conv2d(1,mid_ch,kernel_size=1),
                                            self.relu) for _ in range(in_ch)])
            self.central_mask_conv = BlindSpotConv(mid_ch,mid_ch,(in_ch,k,k), stride=1, dilation=(0,1,1), conv_type=bs_conv_type,padding=(0,p,p))
            


        ly = []
        # ly.append(self.relu)
        if self.bs_conv_type != 'Mask3D':
            ly += [ nn.Conv2d(2*mid_ch, mid_ch, kernel_size=1) ]
        else:
            ly += [ nn.Conv2d(mid_ch, mid_ch, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(mid_ch, mid_ch, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        self.head = nn.Sequential(*ly)  
        # 初始化所有子模块
        init_weights(self.mid_1conv)
        init_weights(self.head)

    def central_frame_blind(self, x):
        B, T, H, W = x.shape
        outs = []
        if self.when_fusion == 'late':
            raise ValueError('late is not supported')
            # for i in range(T):
            #     xi = x[:, [i], :, :]      # shape (B,1,H,W)
            #     xi = F.relu(self.x1_conv(xi))     # shape (B,mid_ch,H,W)
            #     if i == T//2:
            #         yi = self.central_mask_conv_x(xi)
            #     else:
            #         yi = self.normal_conv_x(xi)
            #     outs.append(F.relu(yi))           # 收集所有帧处理后的输出

            # 将 5 帧的结果重新拼成 (B,T*mid_ch,H,W)
            out = torch.cat(outs, dim=1)
        elif self.when_fusion == 'early':
            if self.bs_conv_type != 'Mask3D':
                other_frame = [ x[:, [i], :, :] for i in range(T) if i != T//2]
                other_frame = torch.cat(other_frame, dim=1)
                other_frame = self.relu(self.other_1conv(other_frame))
                other_frame = self.other_normal_conv(other_frame)
                mid_frame = x[:, [T//2], :, :]
                mid_frame = self.relu(self.mid_1conv(mid_frame))
                mid_frame = self.central_mask_conv(mid_frame)
                out = torch.cat([mid_frame,other_frame], dim=1)
            else:
                #batch_size channel frame height width
                x = torch.cat([torch.unsqueeze(self.mid_1conv[i](x[:, [i], :, :]),dim=2) for i in range(T)], dim=2)
                out = self.central_mask_conv(x)
                out = torch.squeeze(out,2)
        return out
    def forward(self, x):
        out1 = self.central_frame_blind(x)
        out2 = self.head(out1)
        return out2


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



class DBSN_fusion_OnlyCB(nn.Module):
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

        #TODO
        self.branch_head_x1 = DC_branch_head(br1_dilation_base, in_ch,mid_ch)
        self.branch_head_x2 = DC_branch_head(br2_dilation_base, in_ch,mid_ch)
        self.branch_head_y1 = DC_branch_head(br1_dilation_base, in_ch,mid_ch)
        self.branch_head_y2 = DC_branch_head(br2_dilation_base, in_ch,mid_ch)

        self.x1_DCL = nn.ModuleList(Inception_block(mid_ch,kernel_size=3,dilation=br1_dilation_base,bias=blindspot_conv_bias,activate_fun=activate_fun) for i in range(self.Mod_num))
        self.x2_DCL = nn.ModuleList(Inception_block(mid_ch,kernel_size=3,dilation=br2_dilation_base,bias=blindspot_conv_bias,activate_fun=activate_fun) for i in range(self.Mod_num))
        self.y1_DCL = nn.ModuleList(Inception_block(mid_ch,kernel_size=3,dilation=br1_dilation_base,bias=blindspot_conv_bias,activate_fun=activate_fun) for i in range(self.Mod_num))
        self.y2_DCL = nn.ModuleList(Inception_block(mid_ch,kernel_size=3,dilation=br2_dilation_base,bias=blindspot_conv_bias,activate_fun=activate_fun) for i in range(self.Mod_num))
        self.conv1x1 = nn.Conv2d(mid_ch*2, mid_ch, kernel_size=1)
        self.conv1x1.apply(weights_init_kaiming)


        self.branch_tail_x1 = DC_branch_tail(mid_ch)
        self.branch_tail_x2 = DC_branch_tail(mid_ch)
        self.branch_tail_y1 = DC_branch_tail(mid_ch)
        self.branch_tail_y2 = DC_branch_tail(mid_ch)

        ly = []
        ly += [ nn.Conv2d(mid_ch*2,  mid_ch,    kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(mid_ch,    mid_ch//2, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(mid_ch//2, mid_ch//2, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(mid_ch//2, out_ch,     kernel_size=1) ]
        self.tailx = nn.Sequential(*ly)
        self.taily = nn.Sequential(*ly)
        init_weights(self.tailx)
        init_weights(self.taily)    

    def forward(self, x,y):
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
            xy1_concat[i] = self.conv1x1(xy1_concat[i])
            xy2_concat[i] = self.conv1x1(xy2_concat[i])
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
    model = DBSN_fusion_OnlyCB(in_ch = 3,
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
