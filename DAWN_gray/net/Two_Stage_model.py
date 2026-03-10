import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
# from .blind_spot_conv import BlindSpotConv
from util.utils import init_weights, weights_init_kaiming
from functools import partial
from net.Biconvlstm import BiConvLSTM
from net.DBSN_centrblind import DBSN_centrblind
from net.DBSN_AttCB import Inception_branch
from net.DBSNl import DBSNl
from net.blind_spot_conv import BlindSpotConv



# class BlindSpotConv_set(nn.Module):
#     def __init__(self, in_ch, out_ch,use_1x1_preconv=False):
#         super().__init__()
#         if use_1x1_preconv:
#             self.BSconv =nn.Sequential(nn.Conv2d(in_ch, out_ch, 1),
#                         nn.ReLU(inplace=True),
#                         BlindSpotConv(out_ch, out_ch, 3, stride=1,dilation=1, bias=True,conv_type='Mask'),
#                         nn.ReLU(inplace=True)
#                         )
#         else:
#             self.BSconv = nn.Sequential(
#                 BlindSpotConv(in_ch, out_ch, 3, stride=1, dilation=1, bias=True, conv_type='Mask'),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1,dilation=1),
#                 nn.ReLU(inplace=True)
#             )
#     def forward(self, x):
#         x = self.BSconv(x)
#         return x

#TODO:In present version,the code has bug if dec_in_ch != 1,in such case,the decoder layer DBSN has head,which is not compatible with the lstmout channel
class TwoStageModel(nn.Module):
    def __init__(self, dec_in_ch, dec_out_ch, dec_mid_ch, 
                blindspot_conv_type, blindspot_conv_bias,
                br1_blindspot_conv_ks, br1_block_num, 
                br2_blindspot_conv_ks, br2_block_num,
                frame_num = 7,
                activate_fun='Relu',
                lstm_layer=1 ,
                DBSN_choice = 'DBSNL'):
        super(TwoStageModel, self).__init__()
        self.lstm_layer = lstm_layer
        self.dec_in_ch = dec_in_ch
        self.dec_mid_ch = dec_mid_ch
        self.frame_num = frame_num
        self.DBSN_choice = DBSN_choice
        #Pre-convolution layers for each frame
        pre_convs = []
        #TODO here I use same kernel size、stride and padding as DBSN's branch 1
        for _ in range(self.frame_num):
            preconv = nn.ModuleList([torch.nn.Conv2d(1, self.dec_mid_ch,  kernel_size=br1_blindspot_conv_ks,  stride=1,padding=(br1_blindspot_conv_ks-1)//2, bias=blindspot_conv_bias),
                                     torch.nn.ReLU(inplace=True),
                                     torch.nn.Conv2d(self.dec_mid_ch, self.dec_mid_ch,kernel_size=br1_blindspot_conv_ks,  stride=1,padding=(br1_blindspot_conv_ks-1)//2, bias=blindspot_conv_bias),
                                     torch.nn.ReLU(inplace=True)])
            pre_convs.append(preconv)
        self.pre_convs = nn.ModuleList(pre_convs)
        
        self.Blindspot_mid = BlindSpotConv(self.dec_mid_ch,self.dec_mid_ch,kernel_size=br1_blindspot_conv_ks, stride=1, dilation=1, bias=blindspot_conv_bias, conv_type=blindspot_conv_type)
        self.biconvlstm = BiConvLSTM(input_dim=self.dec_mid_ch, hidden_dim=self.dec_mid_ch, kernel_size=(3, 3), num_layers=self.lstm_layer)
        self.lstm_out = nn.Sequential(torch.nn.Conv2d(self.frame_num*self.dec_mid_ch,self.dec_mid_ch,kernel_size=1, stride=1, padding=0),
                                        torch.nn.ReLU(inplace=True))

        if self.dec_in_ch == 1:
            if DBSN_choice == '2stage_DBSNL':
                self.DBSN = DBSNl(in_ch=dec_in_ch, out_ch=dec_out_ch, base_ch=dec_mid_ch, num_module=br1_block_num,head=False)
            elif DBSN_choice == '2stage_DBSN':
                self.DBSN = DBSN_Noblindspot(
                    in_ch=dec_in_ch, out_ch=dec_out_ch, mid_ch=dec_mid_ch,
                    blindspot_conv_type=blindspot_conv_type,
                    blindspot_conv_bias=blindspot_conv_bias,
                    br1_blindspot_conv_ks=br1_blindspot_conv_ks,
                    br1_block_num=br1_block_num,
                    br2_blindspot_conv_ks=br2_blindspot_conv_ks,
                    br2_block_num=br2_block_num,
                    activate_fun=activate_fun,
                )
        elif self.dec_in_ch != 1:
            self.DBSN =  DBSN_centrblind( 
                dec_in_ch=dec_in_ch, dec_out_ch=dec_out_ch, dec_mid_ch=dec_mid_ch,
                blindspot_conv_type=blindspot_conv_type,
                blindspot_conv_bias=blindspot_conv_bias,
                br1_blindspot_conv_ks=br1_blindspot_conv_ks,
                br1_block_num=br1_block_num,
                br2_blindspot_conv_ks=br2_blindspot_conv_ks,
                br2_block_num=br2_block_num,
                activate_fun=activate_fun
            )
        init_weights(self.Blindspot_mid)
        init_weights(self.pre_convs)
        init_weights(self.biconvlstm)
        init_weights(self.lstm_out)

    def forward(self, x):
        x = x.unsqueeze(2)
        CNN_seq = []
        half_num = (self.frame_num-1)//2

        for i in range(self.frame_num):
            pre_conv_this_frame = self.pre_convs[i][0](x[:, i])
            pre_conv_1_this_frame = self.pre_convs[i][1](pre_conv_this_frame)
            CNN_seq.append(pre_conv_1_this_frame) 
        # print(f'CNN_seq shape: {len(CNN_seq)}',CNN_seq[0].shape)  # Debugging line to check the shape of CNN_seq
        CNN_seq_other = [CNN_seq[i] for i in range(self.frame_num) if i != half_num] # Exclude the middle frame 
        CNN_seq_out_other = torch.stack(CNN_seq_other, dim=1)  # Shape: (Batchsize, Frame_num, Channel, Height, Width)
        CNN_seq_feature_maps = self.biconvlstm(CNN_seq_out_other)  # Shape: (Batchsize, Frame_num-1, Channel, Height, Width)
        CNN_seq_mid_BS = self.Blindspot_mid(CNN_seq[half_num])  # Apply BlindSpotConv to the middle frame
        # print(f'CNN_seq_mid_BS shape: {CNN_seq_mid_BS.shape}',f'CNN_seq_FM_shape:{CNN_seq_feature_maps.shape}')  # Debugging line to check the shape of CNN_seq_mid_BS
        CNN_seq_feature_maps = CNN_seq_feature_maps.view(CNN_seq_feature_maps.size(0), -1, CNN_seq_feature_maps.size(3), CNN_seq_feature_maps.size(4))  # Flatten the feature maps
        CNN_concat_input = torch.cat([CNN_seq_mid_BS, CNN_seq_feature_maps], dim=1)
        LSTM_out = self.lstm_out(CNN_concat_input)
        
        # if self.dec_in_ch == 1:
        #     dec_input = biconvlstm_out[:,half_num,:,:,:]  # Select the middle frame
        # elif self.dec_in_ch != 1:
        #     dec_input = biconvlstm_out
        if self.DBSN_choice == '2stage_DBSN':
            dec_output,_ = self.DBSN(LSTM_out)
        else:
            dec_output = self.DBSN(LSTM_out)
        return dec_output,LSTM_out

class DBSN_Noblindspot(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch, 
                blindspot_conv_type, blindspot_conv_bias,
                br1_blindspot_conv_ks, br1_block_num, 
                br2_blindspot_conv_ks, br2_block_num,
                activate_fun):
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
        # # Head of DBSN
        # lyr = []
        # lyr.append(nn.Conv2d(in_ch, mid_ch, kernel_size=1, bias=blindspot_conv_bias))
        # lyr.append(self.relu())
        # self.dbsn_head = nn.Sequential(*lyr)
        # init_weights(self.dbsn_head)
        self.br1 = Inception_branch(in_ch,mid_ch, blindspot_conv_type, blindspot_conv_bias, br1_blindspot_conv_ks, br1_block_num, activate_fun)
        self.br2 = Inception_branch(in_ch,mid_ch, blindspot_conv_type, blindspot_conv_bias, br2_blindspot_conv_ks, br2_block_num, activate_fun)

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
        init_weights(self.dbsn_tail)

    def forward(self, x):
        x1 = self.br1(x)     
        x2 = self.br2(x)
        x_concat = torch.cat((x1,x2), dim=1)
        x = self.concat(x_concat)
        return self.dbsn_tail(x), x #前者是DBSN最终输出，即mu输出，后者是连接后的特征图向量


if __name__ == "__main__":
    # Example usage
    model = TwoStageModel(
        dec_in_ch=1, 
        dec_out_ch=1, 
        dec_mid_ch=96, 
        blindspot_conv_type='Mask', 
        blindspot_conv_bias=True, 
        br1_blindspot_conv_ks=3, 
        br1_block_num=8, 
        br2_blindspot_conv_ks=5, 
        br2_block_num=8,
        frame_num=3,
        activate_fun='Relu',
        lstm_layer=1,
        DBSN_choice='2stage_DBSN'
    )
    print(summary(model, input_size=(8, 3, 96, 96)))  # Batch size of 2 and sequence length of 5