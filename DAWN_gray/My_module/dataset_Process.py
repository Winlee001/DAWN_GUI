import glob
import os
import shutil
import numpy as np
import tifffile as tiff
from PIL import Image
from imageio.plugins import gdal
from torch import nn
from torchvision import transforms
import torch
import cv2

img_ext = ('png', 'PNG', 'jpg', 'JPG', 'jpeg', 'JPEG', 'bmp', 'BMP', 'tif', 'TIF')

def is_image(fname):
    return any(fname.endswith(i) for i in img_ext)
def scan(base_path, path=''): # No '/' or '\' at the end of base_path
    images = []
    cur_base_path = base_path if path == '' else os.path.join(base_path, path)
    for d in os.listdir(cur_base_path):
        tmp_path = os.path.join(cur_base_path, d)
        if os.path.isdir(tmp_path):
            images.extend(scan(base_path, os.path.join(path, d)))
        elif is_image(tmp_path):
            images.append(tmp_path)
        else:
            print('Scanning [%s], [%s] is skipped.' % (base_path, tmp_path))
    return images


def Pixel_small_away(file_path):
     lst = os.listdir(file_path)
     num = 0
     for file in lst:
        if file.endswith('.JPEG'):
            img = cv2.imread(os.path.join(file_path, file))
            #if img.shape[0]<256 and img.shape[1]<256:
            if img.shape[0]<97 or img.shape[1]<97:
                num+=1
                os.remove(os.path.join(file_path, file))
                print(os.path.join(file_path, file))
     print(num)
def MAX_MIN_PIXEL(file_path):
    lst = os.listdir(file_path)
    sp0 = [cv2.imread(os.path.join(file_path,file)).shape[0] for file in lst]
    sp1 = [cv2.imread(os.path.join(file_path,file)).shape[1] for file in lst]
    print(max(sp0), max(sp1),min(sp0), min(sp1))

#移除文件名中的'0'
def Name_0_removal(file_name):
    Start = 0
    for i in range(len(file_name)):
        if file_name[i] !='0':
            Start = i
            break
    return file_name[Start:]

def Tif_Tiff(file_path,frame:int):
    #处理的tiff数量
    Num = len(file_path)//frame
    #合并图片的列表
    Stack_lst = []
    for i in range(Num):
        list = []
        print('c')
        for j in range(frame):
            list.append(tiff.imread(file_path[i*frame+j]))
            print(file_path[i*frame+j])
            # 检查图像大小是否一致
            if not all(img.shape == list[0].shape for img in list):
                raise ValueError("所有图像的尺寸必须一致")
        print('end')
        #CHW:frame,height,width
        Stack_img = np.stack(list,axis=0)
        Stack_lst.append(Stack_img)
    return Stack_lst



def read_trans_Mulframe_tiff(file_pathname,build_prefix,frame_num):
    # 遍历该目录下的所有图片文件，也可以用glob模块
    for folder_name in os.listdir(file_pathname):
        for subfolder in os.listdir(os.path.join(file_pathname, folder_name)):
            if (file_pathname[-4:] == 'dark' and subfolder == 'chal') or (
                    file_pathname[-6:] == 'bright' and subfolder == 'chah'):
                # print(subfolder)
                build_name = os.path.join(build_prefix, file_pathname.split('/')[-1], folder_name, subfolder)
                if not os.path.exists(build_name):
                    os.makedirs(build_name)
                print('da')
                file_names = sorted(
                    [f for f in os.listdir(os.path.join(file_pathname, folder_name, subfolder)) if f.endswith('.tif')])
                # 两种file_names计算方法
                # file_names = sorted(glob.glob(os.path.join(file_pathname,folder_name,subfolder,'*.tif')))

                # 得出每个图片的最终路径
                File_path = [os.path.join(file_pathname, folder_name, subfolder, file_name) for file_name in file_names]

                # # 保存tif图片
                # Stack_lst = Tif_Tiff(File_path, frame_num)
                # for i in range(len(Stack_lst)):
                #     tiff.imwrite(build_name +'/'+ f'{i}.tif', Stack_lst[i])

#将帧数合并为整个视频，并且将整个视频帧数进行系统调整
def Merge_Allframe_Bri_adjust(file_pathname,bulid_prefix):
    
    if not os.path.exists(bulid_prefix):
        os.makedirs(bulid_prefix)

    imgs_path = sorted(scan(file_pathname),key=lambda x:int(x.split('/')[-1].split('.')[0]))
    img = Tif_Tiff(imgs_path,len(imgs_path))

    img = np.array(img)
    print(img.max())
    #Bright_adjust
    factor = (65535*0.95)/img.max()
    Adjust_img = (img*factor).astype(np.uint16)
    print(Adjust_img.max())
    #reshape(1,160,5,180,160)->(800,180,160)
    N1,N2,C,H,W = Adjust_img.shape
    Adjust_img = Adjust_img.reshape(N2*C,H,W)
    print(Adjust_img.shape)
    tiff.imwrite(os.path.join(bulid_prefix,'0.tif'),Adjust_img)        
        



#read_trans_Mulframe_tiff('/home/lsy/dark','/home/lsy/Migrate_data',5)
# print('/home/lsy/dark'.split()[-1])
# a = [1,3]
# a = np.array(a)
# print(a.ndim)
#read_trans_Mulframe_tiff('/hhd/2/cyx/Unpaired_dataset_0320_new/train/dark','/home/lsy/Migrate_data',5)
#Pixel_small_away('/home/lsy/anaconda3/pycharm_proj1/DBSN-master/dbsn_gray/datasets/ILSVRC2012_img_val')
#MAX_MIN_PIXEL('/home/lsy/anaconda3/pycharm_proj1/DBSN-master/dbsn_gray/datasets/ILSVRC2012_img_val')
# img1 = cv2.imread(r'/home/lsy/anaconda3/pycharm_proj1/bright_trans_val/96799.png', cv2.IMREAD_GRAYSCALE)
# img1_1 = cv2.imread(r'/home/lsy/anaconda3/pycharm_proj1/bright_trans_val/96799.png', cv2.IMREAD_UNCHANGED)
# print(img1.max(), img1.min(),img1.mean(),img1_1.max(),img1_1.min(),img1_1.mean())
# print(cv2.imread('/home/lsy/anaconda3/pycharm_proj1/bright_trans_Train/96800.png', cv2.IMREAD_GRAYSCALE))
# print(cv2.imread(r'/home/lsy/anaconda3/pycharm_proj1/dark_trans_Test_relight/110691.png', cv2.IMREAD_GRAYSCALE))
# print(cv2.imread('/home/lsy/anaconda3/pycharm_proj1/dark_trans_Train_relight/110689.png', cv2.IMREAD_GRAYSCALE))
#shutil.copytree('/home/lsy/Migrate_data/Bright_strd','/home/lsy/Migrate_data/Bright_strd_Train',dirs_exist_ok=True)
# shutil.rmtree('/home/lsy/anaconda3/pycharm_proj1/DBSN-master/dbsn_gray/Pic_Results')

#手动代码
#shutil.copytree('/home/lsy/Migrate_data/Bright_strd_Test','/home/lsy/Migrate_data/Bright_strd_Test_video')
# s = Tif_Tiff(scan('/home/lsy/Migrate_data/Bright_strd_Test_video'),len(scan('/home/lsy/Migrate_data/Bright_strd_Test_video')))
# for i in range(len(s)):
#     tiff.imwrite('/home/lsy/Migrate_data/Bright_strd_Test_video/20230220_Dai_Final_S9401_0050_S9401_0050.tif/chah'+'/'+f'{i}.tif', s[i])
# shutil.rmtree('/home/lsy/Migrate_data/Bright_strd_Test_video/20230220_Dai_Final_S9401_0050_S9401_0050.tif/chah')
# tif = tiff.imread('/home/lsy/Migrate_data/Dark_strd_Test_video/20230220_Dai_Final_S9401_0049_S9401_0049.tif/chal/0.tif')
# tif = tif.reshape(tif.shape[0]*tif.shape[1], tif.shape[2], tif.shape[3])
# tiff.imwrite('/home/lsy/Migrate_data/Dark_strd_Test_video/20230220_Dai_Final_S9401_0049_S9401_0049.tif/chal/0.tif',tif)

#手动代码
# shutil.copytree('/home/lsy/Migrate_data/Dark_strd_Test','/home/lsy/Migrate_data/Dark_strd_Test_video')
# s = Tif_Tiff(scan('/home/lsy/Migrate_data/Dark_strd_Test_video'),len(scan('/home/lsy/Migrate_data/Dark_strd_Test_video')))
# for i in range(len(s)):
#     tiff.imwrite('/home/lsy/Migrate_data/Dark_strd_Test_video/20230220_Dai_Final_S9401_0049_S9401_0049.tif/chall'+'/'+f'{i}.tif', s[i])
# shutil.rmtree('/home/lsy/Migrate_data/Dark_strd_Test_video/20230220_Dai_Final_S9401_0049_S9401_0049.tif/chal')
# tif = tiff.imread('/home/lsy/Migrate_data/Bright_strd_Test_video/20230220_Dai_Final_S9401_0050_S9401_0050.tif/chah/0.tif')
# tif = tif.reshape(tif.shape[0]*tif.shape[1], tif.shape[2], tif.shape[3])
# tiff.imwrite('/home/lsy/Migrate_data/Bright_strd_Test_video/20230220_Dai_Final_S9401_0050_S9401_0050.tif/chah/0.tif',tif)


# tif = tiff.imread('/home/lsy/Migrate_data/Dark_strd_Test_video/20230220_Dai_Final_S9401_0049_S9401_0049.tif/chal/0.tif')
# print(tif.shape)
if __name__ == '__main__':
    Merge_Allframe_Bri_adjust('/home/lsy/Migrate_data/bright/20230220_Dai_Final_S9401_0050_S9401_0050.tif/chah','/home/lsy/Migrate_data/Bright_strd_Test_video/20230220_Dai_Final_S9401_0050_S9401_0050.tif/chah')
    Merge_Allframe_Bri_adjust('/home/lsy/Migrate_data/dark/20230220_Dai_Final_S9401_0049_S9401_0049.tif/chal','/home/lsy/Migrate_data/Dark_strd_Test_video/20230220_Dai_Final_S9401_0049_S9401_0049.tif/chal')