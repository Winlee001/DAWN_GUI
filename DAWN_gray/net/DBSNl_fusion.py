import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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
    def forward(self, x):
        return self.head(x)

class DC_branch_tail(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        ly = []
        ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ] 
        self.tail = nn.Sequential(*ly)
    def forward(self, x):
        return self.tail(x)
    
class DCL(nn.Module):
    def __init__(self, stride, in_ch):
        super().__init__()

        ly = []
        ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=stride, dilation=stride) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=1) ]
        self.layer = ly
        self.body = nn.Sequential(*ly)

    def forward(self, x):
        return x + self.body(x)

class DBSNl_fusion(nn.Module):
    def __init__(self, in_ch=1, out_ch=3, base_ch=128, num_module=9):
        '''
        Args:
            in_ch      : number of input channel
            out_ch     : number of output channel
            base_ch    : number of base channel
            num_module : number of modules in the network
        '''
        super().__init__()

        assert base_ch%2 == 0, "base channel should be divided with 2"
        assert num_module%2 == 0, "num_module should be divided with 2"

        self.DCL_num = num_module
        ly = []
        ly += [ nn.Conv2d(in_ch, base_ch, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        self.headx = nn.Sequential(*ly)
        self.heady = nn.Sequential(*ly)

        self.branch_head_x1 = DC_branch_head(2, base_ch)
        self.branch_head_x2 = DC_branch_head(3, base_ch)
        self.branch_head_y1 = DC_branch_head(2, base_ch)
        self.branch_head_y2 = DC_branch_head(3, base_ch)

        # self.x1_DCL = [DCL(2, base_ch) for i in range(self.DCL_num)]
        # self.x2_DCL = [DCL(3, base_ch) for i in range(self.DCL_num)]
        # self.y1_DCL = [DCL(2, base_ch) for i in range(self.DCL_num)]
        # self.y2_DCL = [DCL(3, base_ch) for i in range(self.DCL_num)]
        self.x1_DCL = nn.ModuleList(DCL(2, base_ch) for i in range(self.DCL_num))
        self.x2_DCL = nn.ModuleList(DCL(3, base_ch) for i in range(self.DCL_num))
        self.y1_DCL = nn.ModuleList(DCL(2, base_ch) for i in range(self.DCL_num))
        self.y2_DCL = nn.ModuleList(DCL(3, base_ch) for i in range(self.DCL_num))
        self.conv1x1 = nn.Conv2d(base_ch*2, base_ch, kernel_size=1)


        self.branch_tail_x1 = DC_branch_tail(base_ch)
        self.branch_tail_x2 = DC_branch_tail(base_ch)
        self.branch_tail_y1 = DC_branch_tail(base_ch)
        self.branch_tail_y2 = DC_branch_tail(base_ch)

        ly = []
        ly += [ nn.Conv2d(base_ch*2,  base_ch,    kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(base_ch,    base_ch//2, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(base_ch//2, base_ch//2, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(base_ch//2, out_ch,     kernel_size=1) ]
        self.tailx = nn.Sequential(*ly)
        self.taily = nn.Sequential(*ly)

    def forward(self, x,y):
        x = self.headx(x)
        y = self.heady(y)

        x1 = self.branch_head_x1(x)
        x2 = self.branch_head_x2(x)
        y1 = self.branch_head_y1(y)
        y2 = self.branch_head_y2(y)

        xy1_concat = []
        xy2_concat = []
        #self.DCL_num must be even/encoder
        # print(self.DCL_num)
        for i in range(self.DCL_num//2):
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
        for j in range(self.DCL_num//2):
        #     print(j,len(xy1_concat),len(xy2_concat),len(self.x1_DCL),len(self.y1_DCL),len(self.x2_DCL),len(self.y2_DCL))
            x1 = self.x1_DCL[j+self.DCL_num//2](x1+xy1_concat[self.DCL_num//2-j-1])
            y1 = self.y1_DCL[j+self.DCL_num//2](y1+xy1_concat[self.DCL_num//2-j-1])
            x2 = self.x2_DCL[j+self.DCL_num//2](x2+xy2_concat[self.DCL_num//2-j-1])
            y2 = self.y2_DCL[j+self.DCL_num//2](y2+xy2_concat[self.DCL_num//2-j-1])
        #tail
        x = torch.cat([x1, x2], dim=1)
        y = torch.cat([y1, y2], dim=1)

        return self.tailx(x),self.taily(y)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)


if __name__ == '__main__':
    model = DBSNl_fusion(in_ch=5, out_ch=1, base_ch=128, num_module=10).cuda()
    # 注意传入两个输入形状的列表
    summary(model, input_size=[(1, 5, 512, 243), (1, 5, 512, 243)])

