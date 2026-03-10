#!/home/lsy/anaconda3/envs/pytorch201/bin/python
import os
import re
import shutil
import cv2
import numpy as np
import tifffile as tiff
from imageio.plugins import gdal
import matplotlib.pyplot as plt
import glob
from PIL import Image
#from osgeo import gdal
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
img_ext = ('png', 'PNG', 'jpg', 'JPG', 'jpeg', 'JPEG', 'bmp', 'BMP', 'tif', 'TIF')

def is_image(fname):
    return any(fname.endswith(i) for i in img_ext)

def scan(base_path, path=''): # No '/' or '\' at the end of base_path
    images = []
    cur_base_path = base_path if path == '' else os.path.join(base_path, path)
    if not os.path.isdir(cur_base_path) and is_image(cur_base_path):
        images.append(cur_base_path)
    else:
        for d in os.listdir(cur_base_path):
            tmp_path = os.path.join(cur_base_path, d)
            if os.path.isdir(tmp_path):
                images.extend(scan(base_path, os.path.join(path, d)))
            elif is_image(tmp_path):
                images.append(tmp_path)
            else:
                print('Scanning [%s], [%s] is skipped.' % (base_path, tmp_path))
    return images

def Split_data(path_name,spilt_point,Newpath_name):
    Num=0
    for file in os.listdir(path_name):
        if file.endswith(".xml"):
            os.remove(os.path.join(path_name,file))
        elif file.endswith(".png") or file.endswith('.jpg') or file.endswith('.tif'):
            Num=Num+1
            if Num>spilt_point:
                os.renames(os.path.join(path_name, file), os.path.join(Newpath_name, file))
#split and copy data
def Split_copy_data(path_name,split_point,Newpath_name):
    if not os.path.exists(Newpath_name):
        os.makedirs(Newpath_name,exist_ok=True)
    Num=0
    for file in os.listdir(path_name):
        if file.endswith(".xml"):
            os.remove(os.path.join(path_name,file))
        elif file.endswith(".png") or file.endswith('.jpg'):
            if Num<split_point:
                shutil.copy(os.path.join(path_name, file), os.path.join(Newpath_name, file))
            Num=Num+1

def Rename_files(filepath):
    num = 0
    for img_name in os.listdir(filepath):
        if img_name.endswith(".png") or img_name.endswith(".JPEG") or img_name.endswith(".jpg") or img_name.endswith(
            ".jpeg"):
            os.rename(os.path.join(filepath,img_name),os.path.join(filepath, f'{num}.png'))
            num=num+1

# /ssd/0/lsy/SingleMF_Sec_data/s1309_0000/s1309_0000.tif
#split all tif flies into x frames and split into two parts, choose left/right part in selected folder to save.
def Split_tif(filepath,savepath,s_point = 243,frame_num=5,Need_file = 5,choice = 'left'):
    if not os.path.exists(savepath):
                os.makedirs(savepath)
    file_num = 0
    for file in scan(filepath):
        file_num = file_num + 1
        if file_num > Need_file:
            break
        num = 0
        img = tiff.imread(file)
        print(img.shape)
        # frame_num = img.shape[0]
        Dirc = os.path.join(savepath,file.split('/')[-1])
        if not os.path.exists(Dirc):
                os.makedirs(Dirc, exist_ok=True)
        for i in range(img.shape[0]):
            if (i+1) % frame_num == 0:
                num = num + 1
                if choice == 'left':
                    # if s_point is None:No split
                    if s_point is None:
                        tiff.imsave(os.path.join(Dirc, file.split('/')[-1].split('.')[0] + f'_{num}.tif'), img[i - frame_num + 1:i + 1, :, :])
                    else:
                        tiff.imsave(os.path.join(Dirc, file.split('/')[-1].split('.')[0] + f'_{num}.tif'), img[i-frame_num+1:i+1,: , :s_point])      
                elif choice == 'right':
                    if s_point is None:
                        tiff.imsave(os.path.join(Dirc, file.split('/')[-1].split('.')[0] + f'_{num}.tif'), img[i - frame_num + 1:i + 1, :, :])
                    else:
                        tiff.imsave(os.path.join(Dirc, file.split('/')[-1].split('.')[0] + f'_{num}.tif'), img[i-frame_num+1:i+1,: , s_point:])    

def Bright_baseMax_adjust_group(adj_file_path1, save_prefix):
    # 如果目标文件夹不存在则创建
    if not os.path.exists(save_prefix):
        os.makedirs(save_prefix)

    for img_name in scan(adj_file_path1):
        if adj_file_path1 != save_prefix:
            # 根据原始路径相对于输入文件夹的相对路径构造目标路径
            rel = os.path.relpath(img_name, adj_file_path1)
            dst = os.path.join(save_prefix, rel)
            # 如果目标文件夹（可能包含多级子目录）不存在则创建
            dst_dir = os.path.dirname(dst)
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir)
        else:
            dst = img_name
        # 处理png图片
        if img_name.lower().endswith(".png"):
            img = Image.open(img_name)
            img_array = np.array(img, dtype=np.uint16)
            # 对16位深度的图像进行系统调整
            factor = (65535 * 0.95) / img_array.max()
            Adjust_img_array = img_array * factor
            Adjust_img = Image.fromarray(Adjust_img_array.astype(np.uint16))
            Adjust_img.save(dst)

        # 处理tif图片
        elif img_name.lower().endswith('.tif'):
            img = tiff.imread(img_name)
            factor = (65535 * 0.95) / img.max()
            Adjust_img = (img * factor).astype(np.uint16)
            tiff.imwrite(dst, Adjust_img)
            print(f"Saved adjusted image: {dst}")


def extract_from_first_number(path):
    # 使用正则表达式匹配第一个出现的数字及其后续内容
    match = re.search(r'\d.*', path)
    if match:
        return match.group()
    return None


# 展示前 n 个最大元素及其位置
def top_n_elements(arr, n):
    # 将数组展平
    flat = arr.flatten()
    # 获取前 n 个最大元素的索引，类似于快速排序的思想
    indices = np.argpartition(flat, -n)[-n:]
    # print(indices)
    # 获取前 n 个最大元素
    top_n_values = flat[indices]
    # 获取这些元素在原数组中的位置,urnravel_index()函数首先根据indices将一维索引转换为多维索引，并返回对应元素的多维位置
    top_n_indices = np.unravel_index(indices, arr.shape)
    return top_n_values, top_n_indices


#兼具异常输出功能，n为要展示的最大元素个数
def read_tif_print_bright(path,n=5):
    for img_name in scan(path):
        img = tiff.imread(img_name)
        #TODO:异常检测
        # if img.max() != 62258:
        # print(img_name,top_n_elements(img,n),img.max(),img.min(),img.mean())
        if img.max() > 10000:
            print(img_name,top_n_elements(img,n),img.max(),img.min(),img.mean())
        
def Del_abnorm_frame(path,save_path=''):
    for img_name in scan(path):
        img = tiff.imread(img_name)
        filt_img = []
        for frame in range(img.shape[0]):
            if img[frame].max() > 10000:
                print('Abnormal image{0} frame{1}'.format(img_name,frame))
                continue
            filt_img.append(img[frame])
        filt_img = np.array(filt_img)     
        print(img.shape,filt_img.shape)
        tiff.imsave(img_name,filt_img)

# by aligning the name of each file and move them to the corresponding folder
def auto_sort(path,build_name=['s1209','s1205','s1305','s1309']):
    # for img_name in scan(path):
    #     for name in build_name:
    #         if name in img_name:
    #             shutil.move(img_name,os.path.join(path,name,img_name.split('/')[-1]))
    for name in build_name:
        if not os.path.exists(os.path.join(path,name)):
            os.makedirs(os.path.join(path,name))
        for img_name in os.listdir(path):
            if not os.path.isdir(os.path.join(path,img_name)):  
                print(img_name)       
                if name in img_name:
                    shutil.move(os.path.join(path,img_name),os.path.join(path,name,img_name.split('/')[-1]))

from pathlib import Path
def get_files_by_extension(directory, extension):
    # 创建 Path 对象
    directory_path = Path(directory)
    # 递归的筛选出指定文件，rglob
    specified_files = list(directory_path.rglob(f'*{extension}'))
    return [file.resolve() for file in specified_files]

# fuse a specific number of frames into one file,alert!!when use it to fuse mulframe to more mulframe,such as 9_1500，it
#will add one dimension, so be cautious when use the function many times
def fuse_frames(path, save_path, frame_num=5):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    # 对每个最低级文件夹（包含 tif 文件的文件夹）进行处理
    for dir_name in set(os.path.dirname(file) for file in scan(path)):
        # 计算该目录相对于 path 的相对路径，并在 save_path 下构造相同的目录结构/relpath的应用
        rel_path = os.path.relpath(dir_name, path)
        save_dir = os.path.join(save_path, rel_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # 获取该目录下所有 tif 文件，并根据文件名中的数字排序/splitext的应用,将文件名分为主文件名和扩展名
        files = [f for f in os.listdir(dir_name) if f.endswith(".tif")]
        if frame_num == 'auto':
            frame_num = len(files)
        #TODO 处理命名格式为 ：'frame_num+帧起始点序号.tif'||或者'帧起始点序号.tif'
        # Proc = lambda x: int(os.path.splitext(x)[0].split('+')[1])
        # files_sorted = sorted(files, key=Proc)
        files_sorted = sorted(files, key=lambda x: int(os.path.splitext(x)[0]))
        frames = [os.path.join(dir_name, f) for f in files_sorted]
        
        # 每 frame_num 个帧堆叠在一起（沿着新增加的通道维度）
        for i in range(0, len(frames) - frame_num + 1, frame_num):
            group_files = frames[i:i+frame_num]
            # 读取每个灰度图（2D数组）
            imgs = [tiff.imread(fp) for fp in group_files]
            # 使用 np.stack 在新轴（最后一轴）上堆叠，形成 (height, width, frame_num)
            stacked_img = np.stack(imgs, axis= 0)
            
            # 命名格式： 'frame_num+帧起始点序号.tif'
            fused_name = f"{frame_num}+{i}.tif"
            fused_path = os.path.join(save_dir, fused_name)
            
            # 保存多通道 tif 文件
            tiff.imwrite(fused_path, stacked_img)
            print(f"Saved concatenated frame: {fused_path}")

# by aligning the name of each folder with data's type(S/S+N|3000/S+N|9000) and move them to the corresponding folder
def auto_sort_v2(path,mode='dark'):
    if mode == 'dark':
        build_dic = {'S':['s1205','s1209','s1305','s1309'],'S+N|3000':['s1206','s1210','s1306','s1310'],'S+N|9000':['s1207','s1211','s1307','s1311']}
    elif mode == 'bright':
        build_dic = {'S':['s1105','s1109'],'S+N|3000':['s1106','s1110'],'S+N|9000':['s1107','s1111']}
    
    # preprocess the name of each folder to match the key of build_dic
    get_prefix = lambda x: x.strip().split('_')[0]

    for name in build_dic.keys():
        if not os.path.exists(os.path.join(path,name)):
            os.makedirs(os.path.join(path,name))
    for file in os.listdir(path):
        prefix = get_prefix(file)

        if prefix in build_dic['S']:
            shutil.move(os.path.join(path,file),os.path.join(path,'S',file))
        if prefix in build_dic['S+N|3000']:
            shutil.move(os.path.join(path,file),os.path.join(path,'S+N|3000',file))
        if prefix in build_dic['S+N|9000']:
            shutil.move(os.path.join(path,file),os.path.join(path,'S+N|9000',file))

#TODO BUG by aligning the name of each folder in image alignment file and move them to the corresponding folder
def auto_sort_v3(path,save_path,build_name=['0001','0011','0021','0031','0041']):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    Sec_Dir_path = list(set(os.path.dirname(os.path.dirname(x)) for x in scan(path)))
    # all_path = list(set(os.path.dirname(x) for x in scan(path)))
    print(Sec_Dir_path)
    save_paths = []
    for dir_name in Sec_Dir_path:
        rel_path = os.path.relpath(dir_name, path)
        save_dir = os.path.join(save_path,rel_path)
        save_paths.append(save_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    proc = lambda x: x.split('/')[-1].split('.')[0].split('_')[-1]
    # proc = lambda x: x.split('/')[-2].split('.')[0].split('_')[-1]
    
    proc_paths = [proc(x) for x in Sec_Dir_path]
    print(proc_paths)
    for i,ph in enumerate(proc_paths):
        if ph in build_name:
            # if not os.path.exists(save_paths[i]):
            #     os.makedirs(save_paths[i])
            for file in scan(Sec_Dir_path[i]):
                shutil.move(file,save_paths[i])

def Crop_tif(path,crop_shape = (10,260)):
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    for img_name in scan(path):
        img = tiff.imread(img_name)
        tiff.imsave(img_name,img[:,crop_shape[0]:crop_shape[1]])
        print('Cropped image:',img_name)

# remove empty directories
def remove_empty_dirs(path):
    for root, dirs, files in os.walk(path, topdown=False):
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            if not os.listdir(dir_path):  # 检查文件夹是否为空
                os.rmdir(dir_path)
                print(f"Removed empty directory: {dir_path}")

# 示例调用
# remove_empty_dirs('/path/to/directory')
#删除重复目录
def move_files(base_path):
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(".tif"):
                # 获取文件的完整路径
                file_path = os.path.join(root, file)
                # 分割路径，去掉中间的 chal/chal 目录
                parts = file_path.split(os.sep)
                if 'chal' in parts:
                    chal_index = parts.index('chal')
                    new_parts = parts[:chal_index] + parts[chal_index + 2:]
                    new_path = os.path.join(os.sep, *new_parts)  # 确保路径以根目录开始
                    new_dir = os.path.dirname(new_path)
                    # 创建新的目录结构
                    if not os.path.exists(new_dir):
                        os.makedirs(new_dir)
                    # 移动文件
                    shutil.move(file_path, new_path)
                    print(f"Moved: {file_path} -> {new_path}")

#use to split the valset from source data
def auto_sort_v4(path,save_path,build_name=['0001','0011','0021','0031','0041']):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for file in scan(path):
        # proc = lambda x: x.split('/')[-1].split('.')[0].split('_')[-1]
        proc = lambda x: x.split('/')[-3].split('.')[0].split('_')[-1]
        f_name = proc(file)
        if f_name in build_name:
            rel_path = os.path.relpath(os.path.dirname(file),path)
            save_dir = os.path.join(save_path,rel_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            # shutil.move(file,save_dir)
            shutil.copy(file,save_dir)
            print(f"Copyd: {file} -> {save_dir}")

def auto_sort_v5(path,build_name={'low_salt':['s1205','s1206','s1207','s1305','s1306','s1307'],'high_salt':['s1209','s1210','s1211','s1309','s1310','s1311']}):
# by aligning the name of each file and move them to the corresponding folder
    path = path.strip()
    for name in build_name.keys():
        if not os.path.exists(os.path.join(path,name)):
            os.makedirs(os.path.join(path,name))
        for folder_name in os.listdir(path):
            # if not os.path.isdir(os.path.join(path,folder_name)):  
            #     print(fo_name)
                # img_prefix = folder_name.split('.')[0].split('_')[0]  
                img_prefix = folder_name[3:8]     
                if img_prefix in build_name[name]:
                    shutil.move(os.path.join(path,folder_name),os.path.join(path,name))




def Histogram_draw(path):
    for file in scan(path):
        img = tiff.imread(file)
        plt.hist(img.flatten(), bins=1000)
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.show()
#TODO hard to find proper percent
def Percentile_Clipping(path):
    for file in scan(path):
        img = tiff.imread(file)
        # a suitable percent is key,here is 99.9999 for 1500*512*512
        percentile_value = np.percentile(img, 99.9995)
        print(percentile_value)
        scale_factor = (0.95 * 65535) / percentile_value
        # 
        adjusted_img = np.clip(img * scale_factor, 0, 65535).astype(np.uint16)
        tiff.imwrite(file, adjusted_img)


def after_align_adjust(l_path, r_path, save_path):
    for root, Dirs, files in os.walk(l_path):
        if os.path.basename(root) == 'chal':
            frame_name = os.path.basename(os.path.dirname(root))  # 更系统兼容
            Conc_tiff = []
            files = sorted(files, key=lambda x: int(x.split('.')[0]))
            l_files = [os.path.join(root, f) for f in files if f.endswith('.tif')]
            rel_path = os.path.relpath(root, l_path)
            save_dir = os.path.join(save_path, rel_path)
            os.makedirs(save_dir, exist_ok=True)  # 确保输出目录存在
            r_root = os.path.join(r_path, rel_path)
            r_files = [os.path.join(r_root, f) for f in files if f.endswith('.tif')]

            for l_file, r_file in zip(l_files, r_files):
                l_img = tiff.imread(l_file)
                r_img = tiff.imread(r_file)

                if l_img.shape != r_img.shape:
                    raise ValueError(f"Shape mismatch: {l_file} vs {r_file}")
                
                Conc_img = np.concatenate((l_img, r_img), axis=1)
                Conc_tiff.append(Conc_img)

            stacked_img = np.stack(Conc_tiff, axis=0)
            # a suitable percent is key,here is 99.9999 for 600*256*512,which left 79points
            percentile_value = np.percentile(stacked_img, 99.9999)
            scale_factor = (0.95 * 65535) / percentile_value

            if scale_factor < 1:
                raise ValueError(f"Scale factor < 1 for frame: {frame_name}. Check image intensity.")

            adjusted_img = np.clip(stacked_img * scale_factor, 0, 65535).astype(np.uint16)
            fused_path = os.path.join(save_dir, f"Adj{frame_name}")  # 
            tiff.imwrite(fused_path, adjusted_img)
            print(f"Saved concatenated frame: {fused_path}")




    
if __name__ == '__main__':


    #TODO Different attenuation auto_sort
    # auto_sort_v5('/hdd/0/lsy/SMF_dataset/SMF_full_9_cali/train/dark/S/high_salt',
    #              build_name={'10%':['s1205','s1206','s1207','s1209','s1210','s1211'],'25%':['s1305','s1306','s1307','s1309','s1310','s1311']})
    # auto_sort_v5('/hdd/0/lsy/SMF_dataset/SMF_full_9_cali/train/dark/S+N|3000/high_salt',
    #              build_name={'10%':['s1205','s1206','s1207','s1209','s1210','s1211'],'25%':['s1305','s1306','s1307','s1309','s1310','s1311']})
    # auto_sort_v5('/hdd/0/lsy/SMF_dataset/SMF_full_9_cali/train/dark/S+N|9000/high_salt',
    #                 build_name={'10%':['s1205','s1206','s1207','s1209','s1210','s1211'],'25%':['s1305','s1306','s1307','s1309','s1310','s1311']})
    # auto_sort_v5('/hdd/0/lsy/SMF_dataset/SMF_full_9_cali/train/dark/S/low_salt',
    #                 build_name={'10%':['s1205','s1206','s1207','s1209','s1210','s1211'],'25%':['s1305','s1306','s1307','s1309','s1310','s1311']})
    # auto_sort_v5('/hdd/0/lsy/SMF_dataset/SMF_full_9_cali/train/dark/S+N|3000/low_salt',
    #                 build_name={'10%':['s1205','s1206','s1207','s1209','s1210','s1211'],'25%':['s1305','s1306','s1307','s1309','s1310','s1311']})
    # auto_sort_v5('/hdd/0/lsy/SMF_dataset/SMF_full_9_cali/train/dark/S+N|9000/low_salt',
    #                 build_name={'10%':['s1205','s1206','s1207','s1209','s1210','s1211'],'25%':['s1305','s1306','s1307','s1309','s1310','s1311']})
    
    # auto_sort_v5('/hdd/0/lsy/SMF_dataset/SMF_full_9_cali/train/dark_h/S/high_salt',
    #                 build_name={'10%':['s1205','s1206','s1207','s1209','s1210','s1211'],'25%':['s1305','s1306','s1307','s1309','s1310','s1311']})
    # auto_sort_v5('/hdd/0/lsy/SMF_dataset/SMF_full_9_cali/train/dark_h/S+N|3000/high_salt',
    #                 build_name={'10%':['s1205','s1206','s1207','s1209','s1210','s1211'],'25%':['s1305','s1306','s1307','s1309','s1310','s1311']})
    # auto_sort_v5('/hdd/0/lsy/SMF_dataset/SMF_full_9_cali/train/dark_h/S+N|9000/high_salt',
    #                 build_name={'10%':['s1205','s1206','s1207','s1209','s1210','s1211'],'25%':['s1305','s1306','s1307','s1309','s1310','s1311']})
    # auto_sort_v5('/hdd/0/lsy/SMF_dataset/SMF_full_9_cali/train/dark_h/S/low_salt',
    #                 build_name={'10%':['s1205','s1206','s1207','s1209','s1210','s1211'],'25%':['s1305','s1306','s1307','s1309','s1310','s1311']})
    # auto_sort_v5('/hdd/0/lsy/SMF_dataset/SMF_full_9_cali/train/dark_h/S+N|3000/low_salt',
    #                 build_name={'10%':['s1205','s1206','s1207','s1209','s1210','s1211'],'25%':['s1305','s1306','s1307','s1309','s1310','s1311']})
    # auto_sort_v5('/hdd/0/lsy/SMF_dataset/SMF_full_9_cali/train/dark_h/S+N|9000/low_salt',
    #                 build_name={'10%':['s1205','s1206','s1207','s1209','s1210','s1211'],'25%':['s1305','s1306','s1307','s1309','s1310','s1311']})
      
    
    # auto_sort_v5('/hdd/2/lsy/Cali_9_val/full_dark/S/high_salt',
    #             build_name={'10%':['s1205','s1206','s1207','s1209','s1210','s1211'], '25%':['s1305','s1306','s1307','s1309','s1310','s1311']})
    # auto_sort_v5('/hdd/2/lsy/Cali_9_val/full_dark/S+N|3000/high_salt',
    #             build_name={'10%':['s1205','s1206','s1207','s1209','s1210','s1211'], '25%':['s1305','s1306','s1307','s1309','s1310','s1311']})
    # auto_sort_v5('/hdd/2/lsy/Cali_9_val/full_dark/S+N|9000/high_salt',
    #             build_name={'10%':['s1205','s1206','s1207','s1209','s1210','s1211'], '25%':['s1305','s1306','s1307','s1309','s1310','s1311']})
    # auto_sort_v5('/hdd/2/lsy/Cali_9_val/full_dark/S/low_salt',
    #             build_name={'10%':['s1205','s1206','s1207','s1209','s1210','s1211'], '25%':['s1305','s1306','s1307','s1309','s1310','s1311']})
    # auto_sort_v5('/hdd/2/lsy/Cali_9_val/full_dark/S+N|3000/low_salt',
    #             build_name={'10%':['s1205','s1206','s1207','s1209','s1210','s1211'], '25%':['s1305','s1306','s1307','s1309','s1310','s1311']})
    # auto_sort_v5('/hdd/2/lsy/Cali_9_val/full_dark/S+N|9000/low_salt',
    #             build_name={'10%':['s1205','s1206','s1207','s1209','s1210','s1211'], '25%':['s1305','s1306','s1307','s1309','s1310','s1311']})

    # auto_sort_v5('/hdd/2/lsy/Cali_9_val/full_dark_h/S/high_salt',
    #             build_name={'10%':['s1205','s1206','s1207','s1209','s1210','s1211'], '25%':['s1305','s1306','s1307','s1309','s1310','s1311']})
    # auto_sort_v5('/hdd/2/lsy/Cali_9_val/full_dark_h/S+N|3000/high_salt',
    #             build_name={'10%':['s1205','s1206','s1207','s1209','s1210','s1211'], '25%':['s1305','s1306','s1307','s1309','s1310','s1311']})
    # auto_sort_v5('/hdd/2/lsy/Cali_9_val/full_dark_h/S+N|9000/high_salt',
    #             build_name={'10%':['s1205','s1206','s1207','s1209','s1210','s1211'], '25%':['s1305','s1306','s1307','s1309','s1310','s1311']})
    # auto_sort_v5('/hdd/2/lsy/Cali_9_val/full_dark_h/S/low_salt',
    #             build_name={'10%':['s1205','s1206','s1207','s1209','s1210','s1211'], '25%':['s1305','s1306','s1307','s1309','s1310','s1311']})
    # auto_sort_v5('/hdd/2/lsy/Cali_9_val/full_dark_h/S+N|3000/low_salt',
    #             build_name={'10%':['s1205','s1206','s1207','s1209','s1210','s1211'], '25%':['s1305','s1306','s1307','s1309','s1310','s1311']})
    # auto_sort_v5('/hdd/2/lsy/Cali_9_val/full_dark_h/S+N|9000/low_salt',
    #             build_name={'10%':['s1205','s1206','s1207','s1209','s1210','s1211'], '25%':['s1305','s1306','s1307','s1309','s1310','s1311']})
    
    # auto_sort_v5('/ssd/0/lsy/SMF_BSN/SMF_tiny_test/full_tiny_val/full_dark/S/high_salt',build_name={'10%':['s1205','s1206','s1207','s1209','s1210','s1211'],
    #                                                                                                 '25%':['s1305','s1306','s1307','s1309','s1310','s1311']})
    # auto_sort_v5('/ssd/0/lsy/SMF_BSN/SMF_tiny_test/full_tiny_val/full_dark/S+N|3000/high_salt',build_name={'10%':['s1205','s1206','s1207','s1209','s1210','s1211'],
    #                                                                                                 '25%':['s1305','s1306','s1307','s1309','s1310','s1311']})
    # auto_sort_v5('/ssd/0/lsy/SMF_BSN/SMF_tiny_test/full_tiny_val/full_dark/S+N|9000/high_salt',build_name={'10%':['s1205','s1206','s1207','s1209','s1210','s1211'],
    #                                                                                                 '25%':['s1305','s1306','s1307','s1309','s1310','s1311']})
    # auto_sort_v5('/ssd/0/lsy/SMF_BSN/SMF_tiny_test/full_tiny_val/full_dark/S/low_salt',build_name={'10%':['s1205','s1206','s1207','s1209','s1210','s1211'],
    #                                                                                                 '25%':['s1305','s1306','s1307','s1309','s1310','s1311']})
    # auto_sort_v5('/ssd/0/lsy/SMF_BSN/SMF_tiny_test/full_tiny_val/full_dark/S+N|3000/low_salt',build_name={'10%':['s1205','s1206','s1207','s1209','s1210','s1211'],
    #                                                                                                 '25%':['s1305','s1306','s1307','s1309','s1310','s1311']})
    # auto_sort_v5('/ssd/0/lsy/SMF_BSN/SMF_tiny_test/full_tiny_val/full_dark/S+N|9000/low_salt',build_name={'10%':['s1205','s1206','s1207','s1209','s1210','s1211'],
    #                                                                                                 '25%':['s1305','s1306','s1307','s1309','s1310','s1311']})
    
    # auto_sort_v5('/ssd/0/lsy/SMF_BSN/SMF_tiny_test/full_tiny_val/full_dark_h/S/high_salt',build_name={'10%':['s1205','s1206','s1207','s1209','s1210','s1211'],
    #                                                                                                 '25%':['s1305','s1306','s1307','s1309','s1310','s1311']})
    # auto_sort_v5('/ssd/0/lsy/SMF_BSN/SMF_tiny_test/full_tiny_val/full_dark_h/S+N|3000/high_salt',build_name={'10%':['s1205','s1206','s1207','s1209','s1210','s1211'],
    #                                                                                                 '25%':['s1305','s1306','s1307','s1309','s1310','s1311']})
    # auto_sort_v5('/ssd/0/lsy/SMF_BSN/SMF_tiny_test/full_tiny_val/full_dark_h/S+N|9000/high_salt',build_name={'10%':['s1205','s1206','s1207','s1209','s1210','s1211'],
    #                                                                                                 '25%':['s1305','s1306','s1307','s1309','s1310','s1311']})
    # auto_sort_v5('/ssd/0/lsy/SMF_BSN/SMF_tiny_test/full_tiny_val/full_dark_h/S/low_salt',build_name={'10%':['s1205','s1206','s1207','s1209','s1210','s1211'],
    #                                                                                                 '25%':['s1305','s1306','s1307','s1309','s1310','s1311']})
    # auto_sort_v5('/ssd/0/lsy/SMF_BSN/SMF_tiny_test/full_tiny_val/full_dark_h/S+N|3000/low_salt',build_name={'10%':['s1205','s1206','s1207','s1209','s1210','s1211'],
    #                                                                                                 '25%':['s1305','s1306','s1307','s1309','s1310','s1311']})
    # auto_sort_v5('/ssd/0/lsy/SMF_BSN/SMF_tiny_test/full_tiny_val/full_dark_h/S+N|9000/low_salt',build_name={'10%':['s1205','s1206','s1207','s1209','s1210','s1211'],
    #                                                                                                 '25%':['s1305','s1306','s1307','s1309','s1310','s1311']})
    # after_align_adjust('/hdd/0/lsy/SMF_0401_dataset_test/test/dark','/hdd/0/lsy/SMF_0401_dataset_test/test/dark_h','/hdd/0/lsy/SMF_0401_dataset_test/test/combine')
    # after_align_adjust('/hdd/0/lsy/SMF_0401_3H260/train/dark','/hdd/0/lsy/SMF_0401_3H260/train/dark_h','/hdd/0/lsy/SMF_0401_3H260/combine')
    # print(os.path.basename('/hdd/0/lsy/SMF_dataset/Cali_9_val/train/full_dark/S/high_salt/s1205_0000.tif'))
    # path = '/hdd/0/lsy/SMF_0401_dataset_test/test/combine'
    # save_lpath = '/hdd/0/lsy/SMF_0401_Dealeddata/test/dark'
    # save_rpath = '/hdd/0/lsy/SMF_0401_Dealeddata/test/dark_h'
    # for file in scan(path):
    #     name = os.path.basename(file)
    #     img = tiff.imread(file)
    #     imgl = img[:,:,:img.shape[2]//2]
    #     imgr = img[:,:,img.shape[2]//2:]
    #     tiff.imwrite(os.path.join(save_lpath,name),imgl)
    #     print(f'Save:{save_lpath}{name}')
    #     tiff.imwrite(os.path.join(save_rpath,name),imgr)
    #     print(f'Save:{save_rpath}{name}')
    # Crop_tif('/hdd/0/lsy/SMF_0401_3H260/combine')
    # path = '/hdd/0/lsy/SMF_0401_3H260/combine'
    # save_lpath = '/hdd/0/lsy/SMF_0401_Dealeddata/train_3H260/dark'
    # save_rpath = '/hdd/0/lsy/SMF_0401_Dealeddata/train_3H260/dark_h'
    # Split_tif(path,save_lpath,s_point=252,frame_num=9,Need_file=1e10,choice='left')
    # Split_tif(path,save_rpath,s_point=252,frame_num=9,Need_file=1e10,choice='right')

    build_name1 = {
        '40%':['s1205','s1206','s1207','s1209','s1210','s1211'],
        '25%':['s1305','s1306','s1307','s1309','s1310','s1311'],
        '15%':['s1405','s1406','s1407','s1409','s1410','s1411'],
        '10%':['s1505','s1506','s1507','s1509','s1510','s1511'],
    }
    build_name2 = {'low_salt':['s1205','s1206','s1207','s1305','s1306','s1307','s1405','s1406','s1407','s1505','s1506','s1507'],
                    'high_salt':['s1209','s1210','s1211','s1309','s1310','s1311','s1409','s1410','s1411','s1509','s1510','s1511']}
    build_name3 = {'S':['s1205','s1209','s1305','s1309','s1405','s1409','s1505','s1509'],
                  'S+N|3000':['s1206','s1210','s1306','s1310', 's1406','s1410','s1506','s1510'],
                  'S+N|9000':['s1207','s1211','s1307','s1311','s1407','s1411','s1507','s1511']}
    
    def classify_file(file_name, base_path):
    # 获取文件名中的关键部分
        prefix = os.path.basename(file_name)[3:8]  # 's1205'类似的前缀
        
        # 进行分类
        for build1, files1 in build_name1.items():
            if prefix in files1:
                for build2, files2 in build_name2.items():
                    if prefix in files2:
                        for build3, files3 in build_name3.items():
                            if prefix in files3:
                                # 创建对应目录
                                build1_path = os.path.join(base_path, build1)
                                build2_path = os.path.join(build1_path, build2)
                                build3_path = os.path.join(build2_path, build3)
                                
                                # 如果目录不存在就创建
                                os.makedirs(build3_path, exist_ok=True)
                                print(f'Creating directory: {build3_path}')
                                print(f'Moving file: {file_name} to {build3_path}')
                                # 移动文件到对应目录,中断情况重新传输
                                try:
                                    shutil.move(file_name, build3_path)
                                except shutil.Error as e:
                                    print(f'File already exists : {e}')
                                return

    def organize_files(folder_path):
        # 获取文件夹下所有文件绝对路径
        files = scan(folder_path)
        # 遍历文件并进行分类/对于训练集采取文件夹分类
        # for file in files:
        for file in set([os.path.dirname(file) for file in files]):
            # 如果是tif文件就进行分类
            # if os.path.isfile(file) and file.endswith('.tif'):
            if file.endswith('.tif'):
                classify_file(file, folder_path)
        
 
    a= np.arange(0, 5, 1)
    print(type(a),len(a),a[3])
  
     
       
    


    


