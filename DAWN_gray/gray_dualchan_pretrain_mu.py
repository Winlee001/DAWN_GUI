import os
import random
import datetime
import time
from pathlib import Path
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from data import create_dataset

import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

from gray_options import opt
from net.backbone_net import DBSN_Model
from net.sigma_net import Sigma_mu_Net, Sigma_n_Net
from net.losses import L2Loss,L2_V_Loss,AreaCons_Loss,L2_V_WI_Loss
from setproctitle import setproctitle
import wandb
#TODO import DBSN light version and fusion version
from net.DBSNl_fusion import DBSNl_fusion
from net.DBSNl import DBSNl
from net.DBSN_fusion import DBSN_fusion
from net.DBSN_fusionX import DBSN_fusionX
from net.DBSN_DC import DBSN_DC
from data.image_dataset import IterImageDataset
#TODO 自定义进程名
# Proc_name = opt.noise_type+'_'+str(opt.train_noiseL)
Proc_name = str(opt.Run_description)
# 设置自定义进程名称
setproctitle(Proc_name)

seed=0
random.seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
if seed == 0:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def main(args):
    
    # 获取当前时间，格式为 "MMDD_HHMM"（如 0319_1530）
    current_time = datetime.datetime.now().strftime("%m%d_%H%M")
    # # config=dict(learing_rate=0.1,batch_size=2,epoch=50)
    WANDB = wandb.init(project=args.Run_description,dir=r"C:\Users\Admin\Desktop\MUFFLE\Cursor_implement\DAWN_gray\wandb",name=f"{args.Run_description}_{current_time}",settings=wandb.Settings(init_timeout=500),mode="offline")

    # Init loggers
    os.makedirs(args.log_dir, exist_ok=True)
    if args.noise_type == 'Noisy_data' :
        noise_level = args.Run_description

    else:
        raise ValueError('Noise_type [%s] is not found.' % (args.noise_type))
    checkpoint_dir = args.save_prefix + '_' + noise_level
    ckpt_save_path = os.path.join(args.log_dir, checkpoint_dir)
    os.makedirs(ckpt_save_path, exist_ok=True)
    logger_fname = os.path.join(args.log_dir, checkpoint_dir+'_log.txt')


    with open(logger_fname, "w") as log_file:
        now = time.strftime("%c")
        log_file.write('================ Log output (%s) ================\n' % now)
        log_file.write('Parameters \n')
        for key in opt.__dict__:
            p = key+': '+str(opt.__dict__[key])
            log_file.write('%s\n' % p)

    # net architecture
    if args.bsn_ver == 'dbsn':
        dbsn_net = DBSN_Model(in_ch = args.input_channel,
                                out_ch = args.output_channel,
                                mid_ch = args.middle_channel,
                                blindspot_conv_type = args.blindspot_conv_type,
                                blindspot_conv_bias = args.blindspot_conv_bias,
                                br1_block_num = args.br1_block_num,
                                br1_blindspot_conv_ks =args.br1_blindspot_conv_ks,
                                br2_block_num = args.br2_block_num,
                                br2_blindspot_conv_ks = args.br2_blindspot_conv_ks,
                                activate_fun = args.activate_fun)
    elif args.bsn_ver == 'dbsn_light':
        dbsn_net = DBSNl(in_ch = args.input_channel,
                        out_ch = args.output_channel,
                        base_ch = args.middle_channel,
                        num_module = args.br1_block_num+1)
    elif args.bsn_ver == 'dbsnl_fuse':
        dbsn_net = DBSNl_fusion(in_ch = args.input_channel,
                                out_ch = args.output_channel,
                                base_ch = args.middle_channel,
                                num_module = args.br1_block_num)
    elif args.bsn_ver == 'dbsn_fuse':
        dbsn_net = DBSN_fusion(in_ch = args.input_channel,
                        out_ch = args.output_channel,
                        mid_ch = args.middle_channel,
                        blindspot_conv_type = args.blindspot_conv_type,
                        blindspot_conv_bias = args.blindspot_conv_bias,
                        br1_block_num = args.br1_block_num,
                        br1_blindspot_conv_ks =args.br1_blindspot_conv_ks,
                        br2_block_num = args.br2_block_num,
                        br2_blindspot_conv_ks = args.br2_blindspot_conv_ks,
                        activate_fun = args.activate_fun,
                        unet_skip = args.Unet_skip
                        )
    elif args.bsn_ver == 'dbsn_fuseX':
        dbsn_net = DBSN_fusionX(in_ch = args.input_channel,
                        out_ch = args.output_channel,
                        mid_ch = args.middle_channel,
                        blindspot_conv_type = args.blindspot_conv_type,
                        blindspot_conv_bias = args.blindspot_conv_bias,
                        br1_block_num = args.br1_block_num,
                        br1_blindspot_conv_ks =args.br1_blindspot_conv_ks,
                        br2_block_num = args.br2_block_num,
                        br2_blindspot_conv_ks = args.br2_blindspot_conv_ks,
                        activate_fun = args.activate_fun
                        )  
    elif args.bsn_ver == 'dbsn_DC':
        dbsn_net = DBSN_DC(in_ch = args.input_channel,
                        out_ch = args.output_channel,
                        mid_ch = args.middle_channel,
                        blindspot_conv_type = args.blindspot_conv_type,
                        blindspot_conv_bias = args.blindspot_conv_bias,
                        br1_block_num = args.br1_block_num,
                        br1_blindspot_conv_ks =args.br1_blindspot_conv_ks,
                        br2_block_num = args.br2_block_num,
                        br2_blindspot_conv_ks = args.br2_blindspot_conv_ks,
                        activate_fun = args.activate_fun
                        )
                         
    # loss function
    if args.Loss_choice == 'L2_video' or args.Loss_choice == 'L2_video_AC':
        criterion = L2_V_Loss().cuda()
        if args.Loss_choice == 'L2_video_AC':
            AC_Loss = AreaCons_Loss().cuda()
    elif args.Loss_choice == 'L2_video_WeiImage':
        criterion = L2_V_WI_Loss().cuda()

        
    else:
        criterion = L2Loss().cuda()

    # Move to GPU
    dbsn_model = nn.DataParallel(dbsn_net, args.device_ids).cuda()

    # Optimizer
    training_params = None
    optimizer_dbsn = None
    if args.resume == "continue":
        tmp_ckpt=torch.load(args.last_ckpt,map_location=torch.device('cuda', args.device_ids[0]))
        training_params = tmp_ckpt['training_params']
        start_epoch = training_params['start_epoch']
        # Initialize dbsn_model
        pretrained_dict=tmp_ckpt['state_dict_dbsn']
        model_dict=dbsn_model.state_dict()
        pretrained_dict_update = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        assert(len(pretrained_dict)==len(pretrained_dict_update))
        assert(len(pretrained_dict_update)==len(model_dict))
        model_dict.update(pretrained_dict_update)
        dbsn_model.load_state_dict(model_dict)
        optimizer_dbsn = optim.Adam(dbsn_model.parameters(), lr=args.lr_dbsn)
        optimizer_dbsn.load_state_dict(tmp_ckpt['optimizer_state_dbsn'])
        # no_update 强制重置学习率为初始化时的 lr=args.lr_dbsn
        if args.update_opt == 1:
            for param_group in optimizer_dbsn.param_groups:
                param_group['lr'] = args.lr_dbsn
            print([param_group['lr'] for param_group in optimizer_dbsn.param_groups])
    
        schedule_dbsn = torch.optim.lr_scheduler.MultiStepLR(optimizer_dbsn, milestones=args.steps, gamma=args.decay_rate) #衰减器，每到了milestones轮后，衰减到gamma*lr的学习率
        schedule_dbsn.load_state_dict(tmp_ckpt['scheduler_state_dbsn'])
    elif args.resume == "new":
        training_params = {}
        training_params['step'] = 0
        start_epoch = 0
        # Initialize dbsn
        optimizer_dbsn = optim.Adam(dbsn_model.parameters(), lr=args.lr_dbsn)
        schedule_dbsn = torch.optim.lr_scheduler.MultiStepLR(optimizer_dbsn, milestones=args.steps, gamma=args.decay_rate)
    #TODO
    # if args.Real_dataset_mode:
    if not args.dynamic_load:
        # set training set
        train_setname = args.trainset
        train_dataset = create_dataset(train_setname, 'train_CT', args).load_data()
        dataset_num = len(train_dataset)
        # set val set
        val_setname = args.valset
        dataset_val = create_dataset(val_setname, 'val_CT', args).load_data()
        # logging
        with open(logger_fname, "a") as log_file:
            log_file.write('training/val Noisy_dataset created\n')
            log_file.write('number of training examples {0} \n'.format(dataset_num))
    else:
        # set training set
        train_setname = args.trainset
        train_dataset = IterImageDataset(args, 'train_CT', train_setname)
        dataset_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=int(args.load_thread),
            drop_last=False)
   
        dataset_num = train_dataset.length()
        # set val set
        val_setname = args.valset
        dataset_val = create_dataset(val_setname, 'val_CT', args).load_data()
        # logging
        with open(logger_fname, "a") as log_file:
            log_file.write('training/val Noisy_dataset created\n')
            log_file.write('number of training examples {0} \n'.format(dataset_num))

    #定义一个每轮的平均loss
    Avg_loss = {}
    # --------------------------------------------
    # Checkpoint
    # --------------------------------------------
    if args.resume == 'continue':
        # evaluating the loaded model first ...
        print("Starting from epoch: %d "%(start_epoch))
        # logging
        with open(logger_fname, "a") as log_file:
            log_file.write('checkpoint evaluation on epoch {0} ... \n'.format(start_epoch))
        dbsn_model.eval()
        with torch.no_grad():
            # psnr_val = 0
            for count, data in enumerate(dataset_val):
                #
                img_val = data['clean'].cuda()
                img_noise_val = data['noisy'].cuda()
                _,C,H,W = img_noise_val.shape
                # 切片操作，取第 1 到第 args.input_channel 个通道（索引为 0 到 args.input_channel-1）
                img_val = img_val[:, 0:args.input_channel, :, :]  
                img_noise_val = img_noise_val[:, 0:args.input_channel, :, :]  


                # forward
                if args.bsn_ver == 'dbsn':
                    mu_out, _ = dbsn_model(img_noise_val)
                elif args.bsn_ver == 'dbsn_light':
                    mu_out = dbsn_model(img_noise_val)
                elif args.bsn_ver == 'dbsnl_fuse':
                    mux_out,muy_out = dbsn_model(img_val,img_noise_val)
                elif args.bsn_ver == 'dbsn_fuse':
                    mux_out,muy_out = dbsn_model(img_val,img_noise_val)
                elif args.bsn_ver == 'dbsn_fuseX':
                    mux_out,muy_out = dbsn_model(img_val,img_noise_val)
                elif args.bsn_ver == 'dbsn_DC':
                    mux_out,muy_out = dbsn_model(img_val,img_noise_val)
                # logging
                with open(logger_fname, "a") as log_file:
                    log_file.write('Image {0} \n'
                        .format(count))
            # TODO
            Min_Avg_loss = args.best_loss

            # logging
            with open(logger_fname, "a") as log_file:
                log_file.write('Current Best_loss:{0}\n'.format(Min_Avg_loss))

            idx_epoch = start_epoch
    else:
        Min_Avg_loss = 1e10  # Initialize to a large value

    # logging
    with open(logger_fname, "a") as log_file:
        log_file.write('started training\n')
    for epoch in range(start_epoch, args.epoch):
        epoch_start_time = time.time()
        #
        print('lr: %f' % (optimizer_dbsn.state_dict()["param_groups"][0]["lr"]))
        #
        train_dataset.epoch = epoch
        dbsn_model.train()
        # train
        for i, data in enumerate(dataset_loader):
            # load training data
            img_train = data['clean'].cuda()
            img_noise = data['noisy'].cuda()
            
            # print(data['clean_name'],'\t',data['noisy_name'])
            # print(img_train.shape,img_train)
            # print(img_noise,img_noise.shape)
         
            N,C,H,W = img_noise.shape
            #
            optimizer_dbsn.zero_grad()
            # forward
            if args.bsn_ver == 'dbsn':
                mu_out, _ = dbsn_model(img_noise)
            elif args.bsn_ver == 'dbsn_light':
                mu_out = dbsn_model(img_noise)
            elif args.bsn_ver == 'dbsnl_fuse':
                mux_out,muy_out = dbsn_model(img_train,img_noise)
            elif args.bsn_ver == 'dbsn_fuse':
                mux_out,muy_out = dbsn_model(img_train,img_noise)
            elif args.bsn_ver == 'dbsn_fuseX':
                mux_out,muy_out = dbsn_model(img_train,img_noise)
            elif args.bsn_ver == 'dbsn_DC':
                mux_out,muy_out = dbsn_model(img_train,img_noise)
            
            loss_x = criterion(img_train, mux_out,args.weight)
            loss_y = criterion(img_noise, muy_out,args.weight)
            loss = loss_x + loss_y
            #Brightness conversative loss
            if args.Loss_choice == 'L2_video_AC':
                AC_Loss = AreaCons_Loss().cuda()
                target_donor = img_train[:,img_train.shape[1]//2,:,:].unsqueeze(1)
                target_acceptor = img_noise[:,img_noise.shape[1]//2,:,:].unsqueeze(1)
                AC_Loss = AC_Loss(target_donor, target_acceptor, mux_out, muy_out)
                AC_Loss_value = AC_Loss.item()
                loss += 0.5*AC_Loss
            else:
                AC_Loss_value = 0
            loss = loss / (2*args.batch_size)
            loss_value,loss_x_value,loss_y_value = loss.item(),loss_x.item(),loss_y.item()
            WANDB.log({'loss':loss_value, 'loss_x':loss_x_value,'loss_y':loss_y_value,'loss_BC':AC_Loss_value,'lr':optimizer_dbsn.state_dict()["param_groups"][0]["lr"],'epoch':epoch}, step=training_params['step'])
            loss.backward()
            optimizer_dbsn.step()
            # Results
            training_params['step'] += args.batch_size
            # logging
            if epoch not in Avg_loss:
                Avg_loss[epoch] = 0
            Avg_loss[epoch] += loss_value

        schedule_dbsn.step()
        # taking time for each epoch
        tr_take_time = time.time() - epoch_start_time

        # --------------------------------------------
        # validation
        # --------------------------------------------
        # print
        # print("Evaluating on "+str(val_setname[0]))
        # logging
        with open(logger_fname, "a") as log_file:
             log_file.write('Evaluation on %s \n' % (str(val_setname[0])) )
          #
        dbsn_model.eval()
        val_log_dict = {}
        with torch.no_grad():
            # psnr_val = 0
            for count, data in enumerate(dataset_val):
                #
                img_val = data['clean'].cuda()
                img_noise_val = data['noisy'].cuda()
                _,C,H,W = img_noise_val.shape
                
                img_val = img_val[:, 0:args.input_channel, :, :]  
                img_noise_val = img_noise_val[:, 0:args.input_channel, :, :]  
                # forward
                if args.bsn_ver == 'dbsn':
                    mu_out, _ = dbsn_model(img_noise_val)
                elif args.bsn_ver == 'dbsn_light':
                    mu_out = dbsn_model(img_noise_val)
                elif args.bsn_ver == 'dbsnl_fuse':
                    mux_out,muy_out = dbsn_model(img_val,img_noise_val)
                elif args.bsn_ver == 'dbsn_fuse':
                    mux_out,muy_out = dbsn_model(img_val,img_noise_val)
                elif args.bsn_ver == 'dbsn_fuseX':
                    mux_out,muy_out = dbsn_model(img_val,img_noise_val)
                elif args.bsn_ver == 'dbsn_DC':
                    mux_out,muy_out = dbsn_model(img_val,img_noise_val)

                # log 每张图像，用 count 区分
                val_log_dict[f'val/noisyx_img_{count}'] = wandb.Image(
                    img_val[0][args.input_channel // 2].float().cpu().numpy(),
                    caption=f'noisyx_val_ep{epoch}_#{count}'
                )
                val_log_dict[f'val/mux_out_{count}'] = wandb.Image(
                    mux_out[0].detach().cpu().numpy(),
                    caption=f'mux_val_ep{epoch}_#{count}'
                )
                val_log_dict[f'val/noisyy_img_{count}'] = wandb.Image(
                    img_noise_val[0][args.input_channel // 2].float().cpu().numpy(),
                    caption=f'noisyy_val_ep{epoch}_#{count}'
                )
                val_log_dict[f'val/muy_out_{count}'] = wandb.Image(
                    muy_out[0].detach().cpu().numpy(),
                    caption=f'muy_val_ep{epoch}_#{count}'
                )

            # 所有图像一次性 log，统一使用 training_params['step']
            WANDB.log(val_log_dict, step=training_params['step'])   
        torch.cuda.synchronize()

        # average loss
        Avg_loss[epoch] = Avg_loss[epoch] / dataset_num
        WANDB.log({'Avg_loss_epoch':Avg_loss[epoch]}, step=training_params['step'])
        if epoch == 0:
            # TODO 先将首轮loss记为最小
            Min_Avg_loss = Avg_loss[0]

        if Avg_loss[epoch] <= Min_Avg_loss or (epoch + 1) % args.save_model_freq == 0:
            training_params['start_epoch'] = epoch
            save_dict = {'state_dict_dbsn': dbsn_model.state_dict(),
                         'optimizer_state_dbsn': optimizer_dbsn.state_dict(),
                         'schedule_state_dbsn': schedule_dbsn.state_dict(),
                         'training_params': training_params,
                         'args': args,
                         }

            torch.save(save_dict, os.path.join(ckpt_save_path, args.save_prefix + '_ckpt_e{}.pth'.format(epoch)))
            if Avg_loss[epoch] <= Min_Avg_loss:
                # TODO 动态地为args新增属性
                args.best_loss = Min_Avg_loss
                Min_Avg_loss = Avg_loss[epoch]
                idx_epoch = epoch
                torch.save(dbsn_model.state_dict(),
                           os.path.join(ckpt_save_path, 'dbsn_net_best_e{}.pth'.format(epoch)))
                del save_dict

        # logging
        with open(logger_fname, "a") as log_file:
            # [0/1/2]: 0-current epoch, 1-total epoch, 2-best epoch
            log_file.write('Epoch: [{0}/{1}/{2}] \t'
                            'Train_time:{tr_take_time:.1f} sec \t'
                            'Current_epoch_avg_loss:{Now_epoch_avg_loss:.8f} \t'
                            'Best_avg_loss:{Best_epoch_avg_loss:.8f} \n'
                                                                .format(epoch, args.epoch, idx_epoch,
                                                                                 tr_take_time=tr_take_time,
                                                                                 Now_epoch_avg_loss=Avg_loss[epoch],
                                                                                 Best_epoch_avg_loss=Min_Avg_loss))
if __name__ == "__main__":
#lker 
    main(opt)

    exit(0)
