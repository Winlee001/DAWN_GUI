import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AreaCons_Loss(nn.Module):
    def __init__(self):
        super(AreaCons_Loss, self).__init__()
        # self.l1 = nn.L1Loss(reduction='mean')
        self.l2 = nn.MSELoss(reduction='mean')

    def forward(self, donor_noisy, acceptor_noisy, donor_denoised, acceptor_denoised):
        with torch.no_grad():
            # 亮度守恒目标（不参与梯度计算）
            noisy_diff = donor_noisy - acceptor_noisy
        denoised_diff = donor_denoised - acceptor_denoised  # 模型输出
        loss = self.l2(denoised_diff, noisy_diff)
        return loss


class L2_V_Loss(nn.Module):
    def __init__(self):
        super(L2_V_Loss, self).__init__()
        self.L2_V = nn.MSELoss(reduction='mean')

    def forward(self,target,mu,weight):
        loss = 0
        target = target.detach()
        #对每帧都赋予监督
        weight = weight
        total_loss = 0
        for i in range(target.shape[1]):
            #提取出对应帧
            target_frame = target[:,i,:,:]
            target_frame = target_frame.unsqueeze(1)
            loss = self.L2_V(target_frame,mu)
            total_loss += weight[i]*loss
        return total_loss

class L2_V_WI_Loss(nn.Module):
    def __init__(self):
        super(L2_V_WI_Loss, self).__init__()
        self.L2_VWI = nn.MSELoss(reduction='mean')

    def forward(self,target,mu,weight,gamma = 1.0):
        loss = 0
        target = target.detach()
        #对每帧都赋予监督
        total_loss = 0
        img_weight = 0
        for i in range(target.shape[1]):
            #提取出对应帧
            target_frame = target[:,[i],:,:]
            img_weight = target_frame** gamma
            loss = self.L2_VWI(target_frame*img_weight,mu*img_weight)
            total_loss += weight[i]*loss
        return total_loss

# L2Loss is mainly used to pre-train mu
class L2Loss(nn.Module):
    def __init__(self):
        super(L2Loss,self).__init__()
        self.L2=nn.MSELoss(reduction='mean')
        #定义了一个均方误差损失（MSE Loss），reduction='mean' 表示在计算损失时会对所有样本的损失求平均值。

    def forward(self, target, mu,P_frame):
        loss = 0
        target = target.detach()
        #target = target.detach() 将目标张量从计算图中分离出来，意味着它不会参与梯度计算。因为在训练过程中，target 通常是标签，
        # 而我们只希望计算预测值 mu 的梯度，所以将 target 分离出来。这是很重要的技巧
        loss = self.L2(target,mu)
        return loss

# To pre-train sigma_mu & sigma_n
class MAPLoss_Pretrain(nn.Module):
    def __init__(self):
        super(MAPLoss_Pretrain,self).__init__()
    def forward(self, target, mu, sigma_mu, sigma_n, sigma_y):
        loss = 0
        target = target.detach()
        mu = mu.detach()
        t1 = ((target - mu) ** 2) / sigma_y 
        t2 = sigma_y.log()
        loss = t1 + t2
        loss = loss.mean()
        return loss

# To finetune the framework
class MAPLoss(nn.Module):
    def __init__(self):
        super(MAPLoss,self).__init__()
    def forward(self, target, mu, sigma_mu, sigma_n, sigma_y):
        loss = 0
        target = target.detach()
        t1 = ((target - mu) ** 2) / sigma_y #一定为正数
        t2 = sigma_y.log()
        # t3 = 0.1*sigma_n.sqrt() 
        loss = t1 + t2 # - t3 # t3 for AWGN only
        loss = loss.mean()
        if t1.max() > 1e+8:
            loss.data.zero_()
        return loss

# To pre-train sigma_mu & sigma_n
class DBSNLoss_Pretrain(nn.Module):
    def __init__(self):
        super(DBSNLoss_Pretrain,self).__init__()
    def forward(self, target, mu, sigma_mu, sigma_n, sigma_y):
        loss = 0
        eps = 1e-6
        target = target.detach()
        mu = mu.detach()
        t1 = ((target - mu) ** 2) / sigma_y
        t2 = (sigma_n.clamp(eps)).log()
        t3 = sigma_mu / (sigma_n).clamp(eps)
        loss = t1 + t2 + t3
        loss = loss.mean()
        if t1.max() > 1e+8 or t3.max()> 1e+8:
            loss.data.zero_()
        return loss

# To finetune the framework
class DBSNLoss(nn.Module):
    def __init__(self):
        super(DBSNLoss,self).__init__()
    def forward(self, target, mu, sigma_mu, sigma_n, sigma_y):
        loss = 0
        eps = 1e-6
        target = target.detach()
        t1 = ((target - mu) ** 2) / sigma_y
        t2 = (sigma_n.clamp(eps)).log()
        t3 = sigma_mu / (sigma_n).clamp(eps)
        loss = t1 + t2 + t3
        loss = loss.mean()
        if t1.max() > 1e+8 or t3.max()> 1e+8:
            loss.data.zero_()
        return loss
    
#for Mulframes-input
class DBSN_V_Loss(nn.Module):
    def __init__(self):
        super(DBSN_V_Loss,self).__init__()
    def forward(self, target, mu, sigma_mu, sigma_n, sigma_y,weight):
        loss = 0
        eps = 1e-6
        target = target.detach()
        total_loss = 0
        for i in range(target.shape[1]):
            #Frame extraction
            target_frame = target[:,i,:,:]
            target_frame = target_frame.unsqueeze(1)
            t1 = ((target - mu) ** 2) / sigma_y
            t2 = (sigma_n.clamp(eps)).log()
            t3 = sigma_mu / (sigma_n).clamp(eps)
            loss = t1 + t2 + t3
            loss = loss.mean()
            if t1.max() > 1e+8 or t3.max()> 1e+8:
                loss.data.zero_()
            total_loss += weight[i]*loss
        return total_loss
    

