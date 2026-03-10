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
from net.sigma_net import Sigma_mu_Net, Sigma_n_Net
from util.utils import batch_psnr
from PIL import Image
from net.DBSNl import DBSNl
from net.DBSNl_fusion import DBSNl_fusion
from net.DBSN_fusion import DBSN_fusion
from net.DBSN_fusionX import DBSN_fusionX
from net.DBSN_DC import DBSN_DC   
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
    # 由于tensor数据类型一般为（N,C,H,W），所以简单的去除前两维度，用squeeze，并一次性转变为Numpy数组

    numpy = numpy[:,0,:,:]
    # print(numpy.shape)
    # numpy = numpy.squeeze(0).squeeze(0).detach().cpu().numpy(
    # 将 Tensor 从 (C, H, W) 转换为 (H, W, C) 格式，并转换为 NumPy 数组
    # numpy = numpy.permute(1, 2, 0).cpu().numpy().由于Numpy只能处理cpu上的张量，所以需要转移到cpu上
    # detach操作是针对训练过程正在使用梯度的分离出来转化为Numpy，验证过程中使用也不会有问题

    # 如果 Tensor 的值是 [0, 1] 范围，需要转换为 [0, 65535]，并将类型转换为 uint16
    if not Perfect:
        numpy = (numpy * 65535).astype(np.uint16)
    if args.imlib == 'tiff':
        tiff.imwrite(os.path.join(save_path, f'{count}.tif'), numpy)
    else:
        # 将 NumPy 数组转换为 PIL Image
        image = Image.fromarray(numpy.squeeze(), mode='I;16')

        # 保存为指定格式的图片，比如 PNG
        image.save(os.path.join(save_path, f'{count}.png'))


def main(args):
    # net architecture
    if args.bsn_ver == 'dbsnl_fuse':
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
                        unet_skip=args.Unet_skip)
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
                        activate_fun = args.activate_fun)
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
                        activate_fun = args.activate_fun)
        
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
            img_val = data['clean'].cuda()
            img_noise_val = data['noisy'].cuda()
            
            #TODO count=name of image,based on the dir of valset
            countx = data['clean_name'][0].split('/')[-1].split('.')[0]
            county = data['noisy_name'][0].split('/')[-1].split('.')[0]
            
            _, C, H, W = img_noise_val.shape  # 获取带噪声图像的形状信息：C 是通道数，H 是高度，W 是宽度。_ 表示批量大小，但在此不需要使用，故用下划线代替。
            # print(C,H,W)

            #slide windows
            frames = len(args.weight)
            Real_outputx = []
            Real_outputy = []
            start_time = time.time()  # <<< 记录开始时间
            for i in range(C-frames+1):
                
                n_window = img_noise_val[:,i:i+frames,:,:]
                window = img_val[:,i:i+frames,:,:]
                # forward backbone
                # if args.bsn_ver == 'dbsn':
                #     mu_out_val, mid_out_val = dbsn_model(window)
                # elif args.bsn_ver == 'dbsn_light':
                #     mu_out_val = dbsn_model(window)
                if args.bsn_ver == 'dbsnl_fuse':
                    mux_out_val,muy_out_val = dbsn_model(window,n_window)
                elif args.bsn_ver == 'dbsn_fuse':
                    mux_out_val,muy_out_val = dbsn_model(window,n_window)
                elif args.bsn_ver == 'dbsn_fuseX':
                    mux_out_val,muy_out_val = dbsn_model(window,n_window)
                elif args.bsn_ver == 'dbsn_DC':
                    mux_out_val,muy_out_val = dbsn_model(window,n_window)
                #
                mux_out_val = mux_out_val.clamp(min=0,max=1)
                muy_out_val = muy_out_val.clamp(min=0,max=1)
                
                mx_o_v = mux_out_val.detach().cpu().numpy()
                my_o_v = muy_out_val.detach().cpu().numpy()
                # print(window.shape)
                Real_outputx.append(mx_o_v)
                Real_outputy.append(my_o_v)
            end_time = time.time()  # <<< 记录结束时间
            print(f"Processing time for {countx}&&{county}: {end_time - start_time:.4f} seconds")

            Real_outputx = np.stack(Real_outputx, axis=0)
            Real_outputy = np.stack(Real_outputy, axis=0)

            Real_outputx = Real_outputx.reshape(Real_outputx.shape[0], Real_outputx.shape[2], Real_outputx.shape[3], Real_outputx.shape[4])
            Real_outputy = Real_outputy.reshape(Real_outputy.shape[0], Real_outputy.shape[2], Real_outputy.shape[3], Real_outputy.shape[4])

            Numpy_PNG_TIF(args, Real_outputx, countx,f'/hdd/0/lsy/DBSN_results/Fusion_Results/slidwin_{Run_description}_{start_epoch}/dark_h' )
            Numpy_PNG_TIF(args, Real_outputy, county,f'/hdd/0/lsy/DBSN_results/Fusion_Results/slidwin_{Run_description}_{start_epoch}/dark')
        # print
        print('done')
        

if __name__ == "__main__":
    main(opt)
    #lst = np.random((8,5,160,180))
    #print(lst,3)
    exit(0)

