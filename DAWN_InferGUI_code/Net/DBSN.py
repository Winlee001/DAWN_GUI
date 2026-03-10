"""
DBSN model wrapper for DAWN
Supports both single channel (DBSN_centrblind) and dual channel (DBSN_fusion) models
"""
import sys
import os

# Add dbsn_gray to Python path

import torch
import torch.nn as nn
import numpy as np

from Net.DAWN_net.DBSN_centrblind import DBSN_centrblind
from Net.DAWN_net.DBSN_fusion import DBSN_fusion


class DBSN_SingleChannel(nn.Module):
    """Wrapper for single channel DBSN model (DBSN_centrblind)"""
    def __init__(self, opt, device="cuda"):
        super().__init__()
        self.opt = opt
        self.device = device
        
        # Get model parameters from opt
        in_ch = getattr(opt, 'Input_frame_num', 3)
        out_ch = getattr(opt, 'Output_channel', 1)
        mid_ch = getattr(opt, 'Middle_channel', 96)
        blindspot_conv_type = getattr(opt, 'Blindspot_conv_type', 'Mask3D')
        blindspot_conv_bias = getattr(opt, 'Blindspot_conv_bias', True)
        br1_block_num = getattr(opt, 'Br1_block_num', 8)
        br1_blindspot_conv_ks = getattr(opt, 'Br1_blindspot_conv_ks', 3)
        br2_block_num = getattr(opt, 'Br2_block_num', 8)
        br2_blindspot_conv_ks = getattr(opt, 'Br2_blindspot_conv_ks', 5)
        activate_fun = getattr(opt, 'Activate_fun', 'Relu')
        mask_shape = getattr(opt, 'Mask_shape', 'o')
        
        self.net = DBSN_centrblind(
            in_ch=in_ch,
            out_ch=out_ch,
            mid_ch=mid_ch,
            blindspot_conv_type=blindspot_conv_type,
            blindspot_conv_bias=blindspot_conv_bias,
            br1_block_num=br1_block_num,
            br1_blindspot_conv_ks=br1_blindspot_conv_ks,
            br2_block_num=br2_block_num,
            br2_blindspot_conv_ks=br2_blindspot_conv_ks,
            activate_fun=activate_fun,
            mask_shape=mask_shape
        )
        
        if device != "cpu":
            self.net = self.net.to(device)
    
    def forward(self, x):
        """
        Input: x shape (B, T, H, W) where T is frame number
        Output: (B, 1, H, W) - denoised middle frame
        """
        # DBSN_centrblind returns (output, mid_feature)
        output, _ = self.net(x)
        return output


class DBSN_DualChannel(nn.Module):
    """Wrapper for dual channel DBSN model (DBSN_fusion)"""
    def __init__(self, opt, device="cuda"):
        super().__init__()
        self.opt = opt
        self.device = device
        
        # Get model parameters from opt
        in_ch = getattr(opt, 'Input_frame_num', 3)
        out_ch = getattr(opt, 'Output_channel', 1)
        mid_ch = getattr(opt, 'Middle_channel', 96)
        blindspot_conv_type = getattr(opt, 'Blindspot_conv_type', 'Mask3D')
        blindspot_conv_bias = getattr(opt, 'Blindspot_conv_bias', True)
        br1_block_num = getattr(opt, 'Br1_block_num', 8)
        br1_blindspot_conv_ks = getattr(opt, 'Br1_blindspot_conv_ks', 3)
        br2_block_num = getattr(opt, 'Br2_block_num', 8)
        br2_blindspot_conv_ks = getattr(opt, 'Br2_blindspot_conv_ks', 5)
        activate_fun = getattr(opt, 'Activate_fun', 'Relu')
        unet_skip = getattr(opt, 'Unet_skip', False)
        
        self.net = DBSN_fusion(
            in_ch=in_ch,
            out_ch=out_ch,
            mid_ch=mid_ch,
            blindspot_conv_type=blindspot_conv_type,
            blindspot_conv_bias=blindspot_conv_bias,
            br1_block_num=br1_block_num,
            br1_blindspot_conv_ks=br1_blindspot_conv_ks,
            br2_block_num=br2_block_num,
            br2_blindspot_conv_ks=br2_blindspot_conv_ks,
            activate_fun=activate_fun,
            unet_skip=unet_skip
        )
        
        if device != "cpu":
            self.net = self.net.to(device)
    
    def forward(self, x, y):
        """
        Input: 
            x shape (B, T, H, W) - channel A
            y shape (B, T, H, W) - channel B
        Output: 
            (B, 1, H, W) - denoised middle frame for channel A
            (B, 1, H, W) - denoised middle frame for channel B
        """
        output_x, output_y = self.net(x, y)
        return output_x, output_y

