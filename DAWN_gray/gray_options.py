import os
import argparse
import datetime
import torch
from pathlib import Path
import numpy as np

def str2bool(s):
    """ Can use type=str2bool in the parser.add_argument function """
    return s.lower() in ('t', 'true', 'y', 'yes', '1', 'sure')

parser = argparse.ArgumentParser(description="DAWN_gray")
parser.add_argument("--log_name", type=str, default="DAWN_gray", help="file name for save")
parser.add_argument("--noise_type", type=str, default="Noisy_data", help="set the noise type")
# data
parser.add_argument("--dataroot", type=str, default="", help="path to set")
#parser.add_argument("--trainset", type=str, default="set12,bsd68,imagenet_val", help="training set name")
parser.add_argument("--trainset", type=str, default="imagenet_val", help="training set name")
parser.add_argument("--valset", type=str, default="bsd68", help="validation set name")
#
#
parser.add_argument("--patch_size", type=int, default=96, help="the patch size of input")
parser.add_argument("--batch_size", type=int, default=8, help="training batch size")
parser.add_argument("--load_thread", type=int, default=4, help="thread for data loader")    # Ming: maybe 0 is not a good choice
# DTCN
parser.add_argument("--input_channel",type=int,default=1,help="the input channel")
parser.add_argument("--output_channel",type=int,default=1,help="the output channel")
parser.add_argument("--middle_channel",type=int,default=96,help="the middle channel")
parser.add_argument("--blindspot_conv_type",type=str,default='Mask',choices=['Trimmed','Mask','MulMask','Mask3D'], help="type of conv(Trimmed | Mask | Othermasks)")
parser.add_argument("--blindspot_conv_bias",type=str2bool,default=True,help="if blindspot conv need bias")
# branch1
parser.add_argument("--br1_block_num",type=int,default=8,help="the number of dilated conv for branch1")
parser.add_argument("--br1_blindspot_conv_ks",type=int,default=3,help="the basic kernel size of dilated conv")
# branch2
parser.add_argument("--br2_block_num",type=int,default=8,help="the number of dilated conv for branch2")
parser.add_argument("--br2_blindspot_conv_ks",type=int,default=5,help="the basic kernel size of dilated conv")
# net_mu
parser.add_argument("--activate_fun", type=str, default='Relu', choices=['LeakyRelu','Relu'],
                    help='type of activate funcition(LeakyRelu | Relu)')
# net_sigma_mu
parser.add_argument("--sigma_mu_input_channel",type=int,default=0,help="the input channel of net_sigma_mu, NO USE!")
parser.add_argument("--sigma_mu_output_channel",type=int,default=1,help="the output channel of net_sigma_mu")
parser.add_argument("--sigma_mu_middle_channel",type=int,default=32,help="the middle channel of net_sigma_mu")
parser.add_argument("--sigma_mu_layers",type=int,default=3,help="the number of conv in net_sigma_mu")
parser.add_argument("--sigma_mu_kernel_size",type=int,default=1,help="the kernel size of conv in net_sigma_mu")
parser.add_argument("--sigma_mu_bias",type=str2bool,default=True,help="if conv in net_sigma_mu need bias")
# net_sigma_n
parser.add_argument("--sigma_n_input_channel",type=int,default=1,help="the input channel of net_sigma_n")
parser.add_argument("--sigma_n_output_channel",type=int,default=1,help="the output channel of net_sigma_n")
parser.add_argument("--sigma_n_middle_channel",type=int,default=32,help="the middle channel of net_sigma_n")
parser.add_argument("--sigma_n_layers",type=int,default=5,help="the number of conv in net_sigma_n")
parser.add_argument("--sigma_n_kernel_size",type=int,default=1,help="the kernel size of conv in net_sigma_n")
parser.add_argument("--sigma_n_bias",type=str2bool,default=True,help="if conv in net_sigma_n need bias")
# save
parser.add_argument("--init_ckpt",type=str,default="None",help="for pre-training the sub-nets: mu, sigma_mu, sigma_n")
parser.add_argument("--last_ckpt",type=str,default="None",help="the ckpt of last net for checkpoint")
parser.add_argument("--resume", type=str, choices=("continue", "new"), default="new",help="continue to train model")
#parser.add_argument("--log_dir", type=str, default='./ckpts/', help='path of log files')
parser.add_argument("--log_dir", type=str, default='/ssd/0/lsy/DBSN_0327bbncz/dual_ckpts', help='path of log files')
#default=50,but no use for now
parser.add_argument("--display_freq", type=int, default=35, help="frequency of showing training results on screen")
parser.add_argument("--save_model_freq", type=int, default=5, help="Number of training epchs to save state")
# Training parameters
parser.add_argument("--optimizer_type", type=str, default='Adam', help="the default optimizer")
parser.add_argument("--lr_policy", type=str, default='step', help='learning rate policy. [linear | step | plateau | cosine]')
parser.add_argument("--lr_dbsn", type=float, default=3e-4, help="the initial learning rate of backbone network")
parser.add_argument("--lr_sigma_mu", type=float, default=3e-4, help="the initial learning rate of net_sigma_mu")
parser.add_argument("--lr_sigma_n", type=float, default=3e-4, help="the initial learning rate of net_sigma_n")
parser.add_argument("--decay_rate", type=float, default=0.1, help="the decay rate of lr rate")
parser.add_argument("--epoch", type=int, default=90, help="number of epochs the model needs to run")
parser.add_argument("--steps", type=str, default="30,60,80", help="schedule steps,use comma(,) between numbers")
# additional
parser.add_argument("--gamma",type=float,default=1,help="additional parameter for MAP Inference, can be set in [0.9,1] (test only)")
# data processing when loaded
'''
*args：这是位置参数，用来指定位置参数或可选参数的名称。
**kwargs：这是关键字参数，允许传递参数的其他选项，如默认值、类型、帮助信息等。
'''
#store_true是一种bool开关，其实现功能就是将no_flip变成一种开关，不给定no_flip即为false，给定（但不用给no_flip的值）即为true
parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data argumentation')
parser.add_argument('--shuffle', type=str2bool, default=True, help='if true shuffle the data')
parser.add_argument('--max_dataset_size', type=int, default=float("inf"),
                    help='Maximum number of samples allowed per dataset. If the dataset directory contains '
                         'more than max_dataset_size, only a subset is loaded.') #float("inf")在python中代表无穷大的
parser.add_argument('--isTrain', type=str2bool, default=True, help='training flag')
parser.add_argument('--preload', type=str2bool, default=True)
parser.add_argument('--multi_imreader', type=str2bool, default=True)
#TODO My module parameters
parser.add_argument('--Real_dataset_mode',type=bool, default=True,help='if true use real dataset')
parser.add_argument('--Run_description',type=str, default=None,help='choose reasonable description')
parser.add_argument('--DBSN_warmup_by_mu',type=bool,default=True,help='if true use DBSN warmup')
parser.add_argument('--Loss_choice',type=str,choices=('DBSNL','MAPL','L2','L2_video','DBSN_video','L2_video_AC','L2_video_WeiImage'),default='L2_video',help='choice to determine loss type,former two used in gray_train,others in gray_pretrain_mu')
parser.add_argument('--weight', type=str,  default='One', choices=('One', 'all'), help='weight for each frame')
parser.add_argument('--best_loss',type=float,default=1,help='Set to compare each epoch and initialize')
parser.add_argument('--update_opt',type=int,default=0,help='Set to choose whether update the DBSN optimizer lr')
parser.add_argument('--use_tracker',type=bool,default=False,help='if True,use Wandb')
parser.add_argument('--bsn_ver',type=str,default='dbsn',choices=('dbsn','dbsn_light','dbsnl_fuse','dbsn_fuse','dbsn_fuseX','dbsn_centrblind','2stage_DBSN','2stage_DBSNL','DTB_DBSN','DBSN_AttCB','dbsnl_centrblind','dbsn_DC'),help='dbsn_net version,and dbsnl_fuse is only available in gray_dualchan_pretrain_mu')
parser.add_argument('--data_rate',type=float,default=1.0,help='Set the use data rate of the dataset')
parser.add_argument('--stratified_sample',type=str2bool,default=True,help='if True,use stratified sample')
parser.add_argument('--repeat',type=int,default=1,help='Set the repeat times of the dataset')
parser.add_argument('--Frame_shuffle',type=str2bool,default=False,help='if True,use Frame shuffle,only applied in SW_train')
parser.add_argument('--mask_shape',type=str,default='o',choices=('o','x','+'),help='Set the mask shape')
#CT tif mode的选择：L_16 tiff
# parser.add_argument('--frames',type=int,default=3,help='Now only support in DBSN_AttCB, set the frames of input.In other case,the input channel is frames of input')
parser.add_argument('--mode', type=str, default='L_16', choices=['L', 'RGB','L_16'])
parser.add_argument('--imlib', type=str, default='tiff', choices=['cv2', 'pillow', 'h5','tiff'])
parser.add_argument('--dynamic_load', type=str2bool, default=True,help='if True,use dynamic load')
parser.add_argument('--Unet_skip',type=str2bool,default=True,help='if True,use Unet skip connection')

# GPU
parser.add_argument('--device_ids', type=str, default='all', help="integers seperated by comma for selected GPUs, -1 for CPU mode.")
# Option parsing
opt = parser.parse_args()

# save_prefix
opt.save_prefix = opt.log_name + '_' + opt.noise_type

# parse steps
steps = opt.steps
steps = steps.split(',')
opt.steps = [int(eval(step)) for step in steps]
# parse trainset
trainsets = opt.trainset
if trainsets.find(',') == -1: #当find找不到逗号的时候将返回-1
    opt.trainset = [trainsets]
else:
    trainsets = trainsets.split(',') #此处已经将字符串分割为列表
    opt.trainset = [str(trainset).replace(' ','') for trainset in trainsets] #去掉列表中可能存在的空格
# parse trainset
valsets = opt.valset
if valsets.find(',') == -1:
    opt.valset = [valsets]
else:
    valsets = valsets.split(',')
    opt.valset = [str(valset).replace(' ','') for valset in valsets]


# set weight
assert opt.input_channel % 2 != 0, 'input channel should be odd number'
target_frame = opt.input_channel//2
if opt.weight == 'One':  
    opt.weight = [0 for i in range(opt.input_channel)]
    opt.weight[target_frame] = 1
elif opt.weight == 'all':
    # use standard normalized gaussian distribution
    x = np.arange(-target_frame, target_frame+1)
    opt.weight = np.exp(-x**2/2)
    opt.weight = opt.weight/opt.weight.sum()
    # 保留两位小数
    opt.weight = np.round(opt.weight, 3)
else:
    raise ValueError('weight should be One or all')

print(f'The weight of each frame is:{opt.weight}')



# set gpu ids
cuda_device_count = torch.cuda.device_count()
if opt.device_ids == 'all':
    # GT 710 (3.5), GT 610 (2.1)
    device_ids = [i for i in range(cuda_device_count)]
else:
    device_ids = [int(i) for i in opt.device_ids.split(',') if int(i) >= 0]
opt.device_ids = [i for i in device_ids if torch.cuda.get_device_capability(i) >= (4,0)]
if len(opt.device_ids) == 0 and len(device_ids) > 0:
    opt.device_ids = device_ids
    print('You\'re using GPUs with computing capability < 4')
elif len(opt.device_ids) != len(device_ids):
    print('GPUs with computing capability < 4 have been disabled')

if len(opt.device_ids) > 0:
    assert torch.cuda.is_available(), 'No cuda available !!!'
    torch.cuda.set_device(opt.device_ids[0])
    print('The GPUs you are using:')
    for gpu_id in opt.device_ids:
        print(' %2d *%s* with capability %d.%d' % (gpu_id,
                torch.cuda.get_device_name(gpu_id),
                *torch.cuda.get_device_capability(gpu_id)))
else:
    print('You are using CPU mode')

# print('\tParameteres list:')
# for key in opt.__dict__:
#     print('\t'+key+': '+str(opt.__dict__[key]))
