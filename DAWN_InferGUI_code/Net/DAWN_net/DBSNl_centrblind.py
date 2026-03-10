import torch
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from net.blind_spot_conv import BlindSpotConv
from DAWN_net.utils import init_weights, weights_init_kaiming
from torchinfo import summary

class DBSNl_centrblind(nn.Module):
    '''
    Dilated Blind-Spot Network (cutomized light version)

    self-implemented version of the network from "Unpaired Learning of Deep Image Denoising (ECCV 2020)"
    and several modificaions are included. 
    see our supple for more details. 
    '''
    def __init__(self, in_ch=1, out_ch=3, base_ch=128, num_module=9,bs_conv_type='Mask3D',bs_conv_bias=True,head=True):
        '''
        Args:
            in_ch      : number of input channel
            out_ch     : number of output channel
            base_ch    : number of base channel
            num_module : number of modules in the network
            #TODO my modification
            bs_conv_type: type of blind spot convolution,same as in DBSN setting
            bs_conv_bias: if True, bias will be added to the blind spot convolution
            head: if head is True, the first conv layer used to change channels will be added
        '''
        super().__init__()
        
        assert base_ch%2 == 0, "base channel should be divided with 2"
        self.head = head
        if self.head:
            ly = []
            ly += [ nn.Conv2d(in_ch, base_ch, kernel_size=1) ]
            ly += [ nn.ReLU(inplace=True) ]
            self.head = nn.Sequential(*ly)

        # self.branch1 = DC_branchl(2, base_ch, num_module)
        # self.branch2 = DC_branchl(3, base_ch, num_module)
        self.branch1 = DC_branchl(2,in_ch,base_ch,bs_conv_type, bs_conv_bias,num_module)
        self.branch2 = DC_branchl(3,in_ch,base_ch,bs_conv_type, bs_conv_bias,num_module)

        ly = []
        ly += [ nn.Conv2d(base_ch*2,  base_ch,    kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(base_ch,    base_ch//2, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(base_ch//2, base_ch//2, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(base_ch//2, out_ch,     kernel_size=1) ]
        self.tail = nn.Sequential(*ly)

    def forward(self, x):
        if self.head:
            x = self.head(x)

        br1 = self.branch1(x)
        br2 = self.branch2(x)

        x = torch.cat([br1, br2], dim=1)

        return self.tail(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)


class DC_branchl(nn.Module):
    def __init__(self, stride, in_ch,mid_ch, bs_conv_type, bs_conv_bias,num_module):
        super().__init__()

        self.bs_conv_type = bs_conv_type
        self.bs_conv_bias = bs_conv_bias
        self.relu = nn.ReLU(inplace=True)


        if self.bs_conv_type != 'Mask3D':
            # self.central_mask_conv = CentralMaskedConv2d(mid_ch, mid_ch, kernel_size=bs_conv_ks, stride=1, padding=p)
            self.central_mask_conv = BlindSpotConv(mid_ch, mid_ch, 3, stride=1, dilation=1, bias=bs_conv_bias, conv_type=bs_conv_type)
            self.other_normal_conv       = nn.Conv2d(mid_ch,mid_ch , kernel_size=3, stride=1, padding=1)
            self.mid_1conv = nn.Conv2d(1, mid_ch, kernel_size=1)
            self.other_1conv = nn.Conv2d(in_ch-1, mid_ch, kernel_size=1)
            init_weights(self.other_1conv)
            init_weights(self.central_mask_conv)
            init_weights(self.other_normal_conv)
        else:
            self.mid_1conv = nn.ModuleList([nn.Sequential(torch.nn.Conv2d(1,mid_ch,kernel_size=1),
                                            nn.ReLU(inplace=True)) for _ in range(in_ch)])
            self.central_mask_conv = BlindSpotConv(mid_ch,mid_ch,(in_ch,3,3), stride=1, dilation=(0,1,1), bias=bs_conv_bias, conv_type=bs_conv_type,padding=(0,1,1))

        ly = []
        # ly += [ CentralMaskedConv2d(in_ch, in_ch, kernel_size=2*stride-1, stride=1, padding=stride-1) ]
        ly += [ nn.ReLU(inplace=True) ]
        if self.bs_conv_type != 'Mask3D':
            ly.append(nn.Conv2d(mid_ch*2, mid_ch, kernel_size=1, bias=bs_conv_bias))
        else:
            ly.append(nn.Conv2d(mid_ch, mid_ch, kernel_size=1, bias=bs_conv_bias))
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(mid_ch, mid_ch, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]

        ly += [ DCl(stride, mid_ch) for _ in range(num_module) ]

        ly += [ nn.Conv2d(mid_ch, mid_ch, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        
        self.body = nn.Sequential(*ly)


    def central_frame_blind(self, x):
        B, T, H, W = x.shape
        outs = []
        if self.bs_conv_type != 'Mask3D':
            other_frame = [x[:, [i], :, :] for i in range(T) if i != T//2]
            other_frame = torch.cat(other_frame, dim=1)
            # print(other_frame.shape,self.other_1conv(other_frame).shape)
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

    def forward(self,x):
        return self.body(self.central_frame_blind(x)) 

class DCl(nn.Module):
    def __init__(self, stride, in_ch):
        super().__init__()

        ly = []
        ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=stride, dilation=stride) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=1) ]
        self.body = nn.Sequential(*ly)

    def forward(self, x):
        return x + self.body(x)

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
    
if __name__ == '__main__':
    x = torch.randn(8, 3, 96, 96)  # 和你 summary 的一样
    # Example usage
    model = DBSNl_centrblind(in_ch=3, out_ch=1, base_ch=128, num_module=9, bs_conv_type='Mask', bs_conv_bias=True, head=False)
    # out = model(x)
    # Print model summary
    summary(model, input_size=(8, 3, 96, 96))
