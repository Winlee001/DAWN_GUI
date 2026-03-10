#
import os
import random
import datetime
import time
from pathlib import Path
import tifffile as tiff
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from data import create_dataset
from setproctitle import setproctitle
from data.image_dataset import IterImageDataset

from gray_options import opt
from net.backbone_net import DBSN_Model
from net.DTB_DBSN import DTB_DBSN
from net.sigma_net import Sigma_mu_Net, Sigma_n_Net
from net.DBSN_AttCB import DBSN_AttCB
from util.utils import batch_psnr
from PIL import Image
from net.DBSNl import DBSNl
from net.DBSN_centrblind import DBSN_centrblind
from net.DBSNl_centrblind import DBSNl_centrblind
seed = 0
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

Proc_name = str(opt.Run_description+'_test')
# 设置自定义进程名称
setproctitle(Proc_name)

def Numpy_PNG_TIF(args, numpy, count, save_path,Perfect=False):
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    # print(numpy.shape)
    numpy = numpy[:,0,:,:]
    # print(numpy.shape)
    # numpy = numpy.squeeze(0).squeeze(0).detach().cpu().numpy(
    if not Perfect:
        numpy = (numpy * 65535).astype(np.uint16)
    if args.imlib == 'tiff':
        tiff.imwrite(os.path.join(save_path, f'{count}.tif'), numpy)
    else:
        image = Image.fromarray(numpy.squeeze(), mode='I;16')

        image.save(os.path.join(save_path, f'{count}.png'))
def main(args):
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
                                activate_fun = args.activate_fun,
                                )
    elif args.bsn_ver == 'dbsn_light':
        dbsn_net = DBSNl(in_ch = args.input_channel,
                        out_ch = args.output_channel,
                        base_ch = args.middle_channel,
                        num_module = args.br1_block_num+1)
    elif args.bsn_ver == 'dbsn_centrblind':
        dbsn_net = DBSN_centrblind(in_ch = args.input_channel,
                                out_ch = args.output_channel,
                                mid_ch = args.middle_channel,
                                blindspot_conv_type = args.blindspot_conv_type,
                                blindspot_conv_bias = args.blindspot_conv_bias,
                                br1_block_num = args.br1_block_num,
                                br1_blindspot_conv_ks =args.br1_blindspot_conv_ks,
                                br2_block_num = args.br2_block_num,
                                br2_blindspot_conv_ks = args.br2_blindspot_conv_ks,
                                mask_shape=args.mask_shape,
                                activate_fun = args.activate_fun)
    elif args.bsn_ver == 'DTB_DBSN':
        dbsn_net = DTB_DBSN(in_ch = args.input_channel,
                                out_ch = args.output_channel,
                                mid_ch = args.middle_channel,
                                blindspot_conv_type = args.blindspot_conv_type,
                                blindspot_conv_bias = args.blindspot_conv_bias,
                                br1_block_num = args.br1_block_num,
                                br1_blindspot_conv_ks =args.br1_blindspot_conv_ks,
                                br2_block_num = args.br2_block_num,
                                br2_blindspot_conv_ks = args.br2_blindspot_conv_ks,
                                activate_fun = args.activate_fun)
    elif args.bsn_ver == 'DBSN_AttCB':
        dbsn_net = DBSN_AttCB(in_ch = args.input_channel,
                                out_ch = args.output_channel,
                                mid_ch = args.middle_channel,
                                blindspot_conv_type = args.blindspot_conv_type,
                                blindspot_conv_bias = args.blindspot_conv_bias,
                                br1_block_num = args.br1_block_num,
                                br1_blindspot_conv_ks =args.br1_blindspot_conv_ks,
                                br2_block_num = args.br2_block_num,
                                br2_blindspot_conv_ks = args.br2_blindspot_conv_ks,
                                activate_fun = args.activate_fun,
                                frames=args.frames)
    elif args.bsn_ver == 'dbsnl_centrblind':
        dbsn_net = DBSNl_centrblind(in_ch = args.input_channel,
                                out_ch = args.output_channel,
                                base_ch = args.middle_channel,
                                bs_conv_type = args.blindspot_conv_type,
                                bs_conv_bias = args.blindspot_conv_bias,
                                num_module= 9,
                                head=False)
  # Move to GPU
    dbsn_model = nn.DataParallel(dbsn_net, args.device_ids).cuda()

    tmp_ckpt = torch.load(args.last_ckpt, map_location=torch.device('cuda', args.device_ids[0]))
    
    #TODO 加入训练轮数的相关参数
    training_params = tmp_ckpt['training_params']
    start_epoch = training_params['start_epoch']
    if args.Run_description == None:  
        Run_description = tmp_ckpt['args'].Run_description
    else:
        Run_description = args.Run_description
    
    # Initialize dbsn_model
    pretrained_dict = tmp_ckpt['state_dict_dbsn']
    model_dict = dbsn_model.state_dict()
    pretrained_dict_update = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    assert (len(pretrained_dict) == len(pretrained_dict_update))
    assert (len(pretrained_dict_update) == len(model_dict))
    model_dict.update(pretrained_dict_update)
    dbsn_model.load_state_dict(model_dict)

    # 根据两种不同数据集形式设置验证集
    if not args.Real_dataset_mode:
        # set val set
        val_setname = args.valset
        dataset_val = create_dataset(val_setname, 'val', args).load_data()
    else:
        # set val set
        val_setname = args.valset
        # dataset_val = create_dataset(val_setname, 'val_CT', args).load_data()
        val_dataset = IterImageDataset(args, 'val_CT', val_setname)
        dataset_val = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=1,
            num_workers=int(args.load_thread),
            drop_last= False,
            )
    # --------------------------------------------
    # Evaluation
    # --------------------------------------------
    print("Evaluation on : %s " % (val_setname))
    dbsn_model.eval()
    with torch.no_grad():  # 推理过程不需要计算梯度，这是因为在验证阶段不需要更新模型参数，只需进行前向传播，因此可以节省内存和加速计算。
        for count, data in enumerate(dataset_val):
            # load input 从干净图像和噪声图像取出图片后并放于显卡上
            # img_val = data['clean'].cuda()
            img_noise_val = data['noisy'].cuda()
            
            #TODO count=name of nosiy image
            count = data['noisy_name'][0].split('/')[-1].split('.')[0]
            
            _, C, H, W = img_noise_val.shape  # 获取带噪声图像的形状信息：C 是通道数，H 是高度，W 是宽度。_ 表示批量大小，但在此不需要使用，故用下划线代替。
            # print(C,H,W)

            #slide windows
            frames = len(args.weight)
            Real_output = []
            start_time = time.time()  # <<< 记录开始时间
            for i in range(C-frames+1):
                
                # print(i_n_v.shape,img_noise_val.shape)
                window = img_noise_val[:,i:i+frames,:,:]      
                # print(window.shape)
                # forward backbone
                if args.bsn_ver == 'dbsn':
                    mu_out_val, mid_out_val = dbsn_model(window)
                elif args.bsn_ver == 'dbsn_light':
                    mu_out_val = dbsn_model(window)
                elif args.bsn_ver == 'dbsn_centrblind':
                    mu_out_val, mid_out_val = dbsn_model(window)
                elif args.bsn_ver == 'DTB_DBSN':
                    mu_out_val, mid_out_val = dbsn_model(window)
                elif args.bsn_ver == 'DBSN_AttCB':
                    mu_out_val, mid_out_val = dbsn_model(window)
                elif args.bsn_ver == 'dbsnl_centrblind':
                    mu_out_val = dbsn_model(window)
                #
                mu_out_val = mu_out_val.clamp(min=0,max=1)
                
                m_o_v = mu_out_val.detach().cpu().numpy()

                # print(window.shape)
                Real_output.append(m_o_v)
            end_time = time.time()  # <<< 记录结束时间
            print(f"Processing time for {count}: {end_time - start_time:.4f} seconds")
            Real_output = np.stack(Real_output, axis=0)
            Real_output = Real_output.reshape(Real_output.shape[0], Real_output.shape[2], Real_output.shape[3], Real_output.shape[4])
            Numpy_PNG_TIF(args, Real_output, count,f'/hdd/0/lsy/DBSN_results/Pic_Results/mu_slidewindow_test_{Run_description}_{start_epoch}' )  
        # print
        print('done')
        

if __name__ == "__main__":
    main(opt)
    #lst = np.random((8,5,160,180))
    #print(lst,3)
    exit(0)

