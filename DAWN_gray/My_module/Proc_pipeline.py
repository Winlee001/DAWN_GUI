from concurrent.futures import ProcessPoolExecutor,as_completed
import glob
import os
import shutil
import cv2
import numpy as np
import tifffile as tiff
from PIL import Image
from imageio.plugins import gdal
from torch import nn
from torchvision import transforms
import torch
from tqdm import tqdm 
from dataset_Process import scan,is_image
'''
Sift alignment
'''
#data_test_pro.py

'''
fuse and bright adjustment
'''
def after_align_adjust(l_path, r_path, save_path,ratio=99.9999,scale = 0.95):
    for root, Dirs, files in os.walk(l_path):
        if os.path.basename(root) == 'chal' or os.path.basename(root) == 'chah':
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
            percentile_value = np.percentile(stacked_img, ratio)
            scale_factor = (scale * 65535) / percentile_value

            if scale_factor < 1:
                raise ValueError(f"Scale factor < 1 for frame: {frame_name}. Check image intensity.")

            adjusted_img = np.clip(stacked_img * scale_factor, 0, 65535).astype(np.uint16)
            fused_path = os.path.join(save_dir, f"Adj{frame_name}")  # 
            tiff.imwrite(fused_path, adjusted_img)
            print(f"Saved concatenated frame: {fused_path}")

#TODO bright adjust without alignment,attention organize files !!
def Bright_baseMax_adjust_group_Perc(adj_file_path1, save_prefix,ratio=99.99995,scale = 0.90):
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
            percentile_value = np.percentile(img, ratio)
            scale_factor = (scale * 65535) / percentile_value
            if scale_factor < 1:
                raise ValueError(f"Scale factor < 1 for image: {img_name}. Check image intensity.")
            Adjust_img = np.clip(img * scale_factor, 0, 65535).astype(np.uint16)
            tiff.imwrite(dst, Adjust_img)
            print(f"Saved adjusted image: {dst}")


#Paralel processing
def adjust_image(img_path, input_root, output_root, ratio=99.99995, scale=0.90):
    try:
        # 构造目标路径
        rel_path = os.path.relpath(img_path, input_root)
        dst_path = os.path.join(output_root, rel_path)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

        if img_path.lower().endswith(".png"):
            img = Image.open(img_path)
            img_array = np.array(img, dtype=np.uint16)
            factor = (65535 * 0.95) / img_array.max()
            Adjust_img_array = img_array * factor
            Adjust_img = Image.fromarray(Adjust_img_array.astype(np.uint16))
            Adjust_img.save(dst_path)

        elif img_path.lower().endswith(".tif"):
            img = tiff.imread(img_path)
            # print(img.shape)

            percentile_value = np.percentile(img, ratio)
            scale_factor = (scale * 65535) / percentile_value

            if scale_factor < 1:
                print(f"[WARN] Skip (scale factor < 1): {img_path}")
                return
            # if (img*scale_factor).max() > 65535:
            #     print(f"[WARN] Clipping values for {img_path} to fit in uint16 range.")
            Adjust_img = np.clip(img * scale_factor, 0, 65535).astype(np.uint16)
            tiff.imwrite(dst_path, Adjust_img)
            print(f"Processing {img_path}: scale factor = {scale_factor} percentile value = {percentile_value} \
                  mean/max value = {img.mean()}||{img.max()} adj mean/max value = {Adjust_img.mean()}||{Adjust_img.max()}")
            print(f"[INFO] Saved adjusted image: {dst_path}")
    except Exception as e:
        print(f"[ERROR] Failed processing {img_path}: {e}")
def Bright_baseMax_adjust_group_Perc_parallel(adj_file_path1, save_prefix, ratio=99.99995, scale=0.90, max_workers=8):
    os.makedirs(save_prefix, exist_ok=True)
    img_list = scan(adj_file_path1)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(adjust_image, img_path, adj_file_path1, save_prefix, ratio, scale): img_path
                   for img_path in img_list}

        # tqdm进度条显示，跟踪每个完成的future
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing images"):
            try:
                future.result()
            except Exception as e:
                img_path = futures[future]
                print(f"[ERROR] Failed on {img_path}: {e}")
'''
split the image into 2 parts and specified frame number for train
only for train,because the test set is full video
'''
#split all tif flies into x frames and split into two parts, choose left/right part in selected folder to save.
def Split_tif(filepath,savepath,s_point = 243,frame_num=5,Need_file = 5,choice = 'left_x',mode = 'EvenSplit'):
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
        if choice.split('_')[1] == 'all':
            print(img.shape)
            frame_num = img.shape[0]
        rel_path = os.path.relpath(file,filepath)
        file_save_path = os.path.join(savepath, os.path.dirname(rel_path))
        if not os.path.exists(file_save_path):
                os.makedirs(file_save_path, exist_ok=True)
        print(f'Splitting {file} into {file_save_path} with frame_num {frame_num}, s_point {s_point}, choice {choice}')

        for i in range(img.shape[0]):
            if mode == 'EvenSplit':
                if (i+1) % frame_num == 0:
                    num = num + 1
                    if choice.split('_')[0] == 'left':
                        # if s_point is None:No split
                        if s_point is None:
                            tiff.imwrite(os.path.join(file_save_path, file.split('/')[-1].split('.')[0] + f'_{num}.tif'), img[i - frame_num + 1:i + 1, :, :])
                        else:
                            tiff.imwrite(os.path.join(file_save_path, file.split('/')[-1].split('.')[0] + f'_{num}.tif'), img[i-frame_num+1:i+1,: , :s_point])      
                    elif choice.split('_')[0] == 'right':
                        if s_point is None:
                            tiff.imwrite(os.path.join(file_save_path, file.split('/')[-1].split('.')[0] + f'_{num}.tif'), img[i - frame_num + 1:i + 1, :, :])
                        else:
                            tiff.imwrite(os.path.join(file_save_path, file.split('/')[-1].split('.')[0] + f'_{num}.tif'), img[i-frame_num+1:i+1,: , s_point:])
            elif mode == 'SlideSplit':
                if i > img.shape[0]-frame_num:
                    break
                num = num + 1
                if choice.split('_')[0] == 'left':
                    if s_point is None:
                        tiff.imwrite(os.path.join(file_save_path, file.split('/')[-1].split('.')[0] + f'_{num}.tif'), img[i: i+frame_num, :,:])
                    else:
                        tiff.imwrite(os.path.join(file_save_path, file.split('/')[-1].split('.')[0] + f'_{num}.tif'), img[i: i+frame_num, :, :s_point]) 
                elif choice.split('_')[0] == 'right':
                    if s_point is None:
                        tiff.imwrite(os.path.join(file_save_path, file.split('/')[-1].split('.')[0] + f'_{num}.tif'), img[i: i+frame_num, :,:])
                    else:
                        tiff.imwrite(os.path.join(file_save_path, file.split('/')[-1].split('.')[0] + f'_{num}.tif'), img[i: i+frame_num, :, s_point:])       
      
    '''             
classify the file into different folders according to the name of the file
'''
# build_name1 = {
#     '40%':['s1205','s1206','s1207','s1209','s1210','s1211'],
#     '25%':['s1305','s1306','s1307','s1309','s1310','s1311'],
#     '15%':['s1405','s1406','s1407','s1409','s1410','s1411'],
#     '10%':['s1505','s1506','s1507','s1509','s1510','s1511'],
# }
# build_name2 = {'low_salt':['s1205','s1206','s1207','s1305','s1306','s1307','s1405','s1406','s1407','s1505','s1506','s1507'],
#                 'high_salt':['s1209','s1210','s1211','s1309','s1310','s1311','s1409','s1410','s1411','s1509','s1510','s1511']}
# build_name3 = {'S':['s1205','s1209','s1305','s1309','s1405','s1409','s1505','s1509'],
#                 'S+N|3000':['s1206','s1210','s1306','s1310', 's1406','s1410','s1506','s1510'],
#                 'S+N|9000':['s1207','s1211','s1307','s1311','s1407','s1411','s1507','s1511']}

build_name1 = {
    '40%':['S1213','S1214','S1215','S1217','S1218','S1219'],
    '25%':['S1313','S1314','S1315','S1317','S1318','S1319'],
    '15%':['S1413','S1414','S1415','S1417','S1418','S1419'],
    '10%':['S1513','S1514','S1515','S1517','S1518','S1519'],
}
build_name2 = {'low_salt':['S1213','S1214','S1215','S1313','S1314','S1315','S1413','S1414','S1415','S1513','S1514','S1515'],
                'high_salt':['S1217','S1218','S1219','S1317','S1318','S1319','S1417','S1418','S1419','S1517','S1518','S1519']}
build_name3 = {'S':['S1213','S1217','S1313','S1317','S1413','S1417','S1513','S1517'],
                'S+N|3000':['S1214','S1218','S1314','S1318', 'S1414','S1418','S1514','S1518'],
                'S+N|9000':['S1215','S1219','S1315','S1319','S1415','S1419','S1515','S1519']}

# build_name3 ={
#     '1mW':['s1201','s1202','s1203','s1204'],
#     '100mW':['s1301','s1302','s1303','s1304'],
#     '300mW':['s1401','s1402','s1403','s1404'],
#     '500mW':['s1501','s1502','s1503','s1504'],
# }

build_name3 ={
    '50mW':['s1201','s1202','s1203','s1204','s1205'],
    '100mW':['s1301','s1302','s1303','s1304','s1305'],
    '300mW':['s1401','s1402','s1403','s1404','s1405'],
    '500mW':['s1501','s1502','s1503','s1504','s1505'],
    '1mW':['s1601','s1602','s1603','s1604','s1605'],
}
build_name3 = {
    '100%':['s1101','s1201','s1301','s1401','s1501'],
    '40%':['s1102','s1202','s1302','s1402','s1502'],
    '25%':['s1103','s1203','s1303','s1403','s1503'],
    '15%':['s1104','s1204','s1304','s1404','s1504'],
    '10%':['s1105','s1205','s1305','s1405','s1505'],
    '25%_12mw':['s1106','s1206','s1306','s1406','s1506'],
}
build_name3 = {
    'Fret101':['s1101','s1102','s1103','s1104','s1105','s1106'],
    'Fret102':['s1201','s1202','s1203','s1204','s1205','s1206'],
    'Fret103':['s1301','s1302','s1303','s1304','s1305','s1306'],
    'Fret104':['s1401','s1402','s1403','s1404','s1405','s1406'],
    'Fret105':['s1501','s1502','s1503','s1504','s1505','s1506'],
    'Fret106':['s1601','s1602','s1603','s1604','s1605','s1606'],
    'Fret107':['s1701','s1702','s1703','s1704','s1705','s1706'],
    'Fret108':['s1801','s1802','s1803','s1804','s1805','s1806'],
    'Fret109':['s1901','s1902','s1903','s1904','s1905','s1906'],
}
build_name3 = {
    '100%':['s1101','s1201','s1301','s1401','s1501'],
    '40%':['s1102','s1202','s1302','s1402','s1502'],
    '25%':['s1103','s1203','s1303','s1403','s1503'],
    '15%':['s1104','s1204','s1304','s1404','s1504'],
    '10%':['s1105','s1205','s1305','s1405','s1505'],
    '25%_12mw':['s1106','s1206','s1306','s1406','s1506'],
}
build_name3 = {
    '100%':['s1101','s1201','s1301','s1401','s1501'],
    '40%':['s1102','s1202','s1302','s1402','s1502'],
    '25%':['s1103','s1203','s1303','s1403','s1503'],
    '15%':['s1104','s1204','s1304','s1404','s1504'],
    '10%':['s1105','s1205','s1305','s1405','s1505'],
    '25%_8ms':['s1106','s1206','s1306','s1406','s1506'],
}
def classify_file(file_name, base_path):
# 获取文件名中的关键部分
    prefix = os.path.basename(file_name)[0:5]  # 's1205'类似的前缀
    
    # 进行分类
    for build1, files1 in build_name3.items():
        if prefix in files1:
            build1_path = os.path.join(base_path, build1)                    
            # 如果目录不存在就创建
            os.makedirs(build1_path, exist_ok=True)
            print(f'Creating directory: {build1_path}')
            print(f'Moving file: {file_name} to {build1_path}')
            # 移动文件到对应目录,中断情况重新传输
            try:
                shutil.move(file_name, build1_path)
            except shutil.Error as e:
                print(f'File already exists : {e}')
            return


# def classify_file(file_name, base_path):
# # 获取文件名中的关键部分
#     prefix = os.path.basename(file_name)[0:5]  # 's1205'类似的前缀
    
#     # 进行分类
#     for build1, files1 in build_name3.items():
#         if prefix in files1:
#             for build2, files2 in build_name2.items():
#                 if prefix in files2:
#                     for build3, files3 in build_name1.items():
#                         if prefix in files3:
#                             # 创建对应目录
#                             build1_path = os.path.join(base_path, build1)
#                             build2_path = os.path.join(build1_path, build2)
#                             build3_path = os.path.join(build2_path, build3)
                            
#                             # 如果目录不存在就创建
#                             os.makedirs(build3_path, exist_ok=True)
#                             print(f'Creating directory: {build3_path}')
#                             print(f'Moving file: {file_name} to {build3_path}')
#                             # 移动文件到对应目录,中断情况重新传输
#                             try:
#                                 shutil.move(file_name, build3_path)
#                             except shutil.Error as e:
#                                 print(f'File already exists : {e}')
#                             return

def organize_files(folder_path):
    # 获取文件夹下所有文件绝对路径
    files = scan(folder_path)
    # 遍历文件并进行分类/对于训练集采取文件夹分类
    # for file in files:
    # for file in set([os.path.dirname(os.path.dirname(file)) for file in files]):
    for file in set([os.path.dirname(file) for file in files]):
        # 如果是tif文件就进行分类
        # if os.path.isfile(file) and file.endswith('.tif'):
        # if file.endswith('.tif'):
        classify_file(file, folder_path)
'''
if needed, crop the image
'''
def Crop_tif(path,crop_shape = (10,260)):
    for img_name in scan(path):
        img = tiff.imread(img_name)
        print(img.shape)
        tiff.imwrite(img_name,img[:,crop_shape[0]:crop_shape[1],crop_shape[2]:crop_shape[3]])
        print('Cropped image:',img_name)

def concat_tif(path,save_path,concat_num=3):
    for file_dir in os.listdir(path):
        file_path = os.path.join(path,file_dir)
        file_save_path = os.path.join(save_path,file_dir)
        if not os.path.exists(file_save_path):
                os.makedirs(file_save_path, exist_ok=True)
        sorted_files = sorted(scan(file_path), key=lambda x: os.path.basename(x))
        # print(sorted_files)
        sorted_images = [tiff.imread(f) for f in sorted_files]
        # print(sorted_images[0].shape)
        for i in range(0, len(sorted_images)-concat_num+1):
            try:
                concat_image = np.stack(sorted_images[i:i+concat_num], axis=0)  
                tiff.imwrite(os.path.join(file_save_path, f'concat_{i}_Every{concat_num}.tif'), concat_image)
                print(f'Saved concatenated image: concat_{i}_Every{concat_num}.tif,shape: {concat_image.shape}')
            except Exception as e:
                print(f"Error saving image {i}: {e}")
        

if __name__ == '__main__':
    # after_align_adjust('/hdd/0/lsy/SMF_0401_dataset_test/test/dark','/hdd/0/lsy/SMF_0401_dataset_test/test/dark_h','/hdd/0/lsy/SMF_0401_dataset_test/test/combine_2',ratio=99.99995,scale = 0.90)
    # after_align_adjust('/hdd/0/lsy/SMF_0401_3H260/train/dark','/hdd/0/lsy/SMF_0401_3H260/train/dark_h','/hdd/0/lsy/SMF_0401_3H260/combine_2',ratio=99.99995,scale = 0.90)
    # Split_tif('/hdd/0/lsy/SMF_0401_dataset_test/test/combine_2','/hdd/0/lsy/SMF_0401_Dealeddata/test_3H260_v2/dark',s_point=252,frame_num=600,Need_file = 1e10,choice = 'left')
    # Split_tif('/hdd/0/lsy/SMF_0401_dataset_test/test/combine_2','/hdd/0/lsy/SMF_0401_Dealeddata/test_3H260_v2/dark_h',s_point=252,frame_num=600,Need_file = 1e10,choice = 'right')
    # Split_tif('/hdd/0/lsy/SMF_0401_3H260/combine_2','/hdd/0/lsy/SMF_0401_Dealeddata/train_3H260_v2/dark',s_point=252,frame_num=9,Need_file = 1e10,choice = 'left')
    # Split_tif('/hdd/0/lsy/SMF_0401_3H260/combine_2','/hdd/0/lsy/SMF_0401_Dealeddata/train_3H260_v2/dark_h',s_point=252,frame_num=9,Need_file = 1e10,choice = 'right')
    # organize_files('/hdd/0/lsy/SMF_0401_Dealeddata/train_3H260_v2/dark_h')
    # organize_files('/hdd/0/lsy/SMF_0401_Dealeddata/train_3H260_v2/dark')
    # organize_files('/hdd/0/lsy/SMF_0401_Dealeddata/test_3H260_v2/dark_h')
    # organize_files('/hdd/0/lsy/SMF_0401_Dealeddata/test_3H260_v2/dark')

    # for align data 0419
    # after_align_adjust('/hdd/0/lsy/SMF_0401_dataset_test/test/dark','/hdd/0/lsy/SMF_0401_dataset_test/test/dark_h','/hdd/0/lsy/SMF_0401_dataset_test/test/combine',ratio=99.99995,scale = 0.90)
    # Split_tif('/hdd/0/lsy/SMF_0401_dataset_test/test/combine','/hdd/0/lsy/SMF_0401_Dealeddata/test_3H260_v2/dark',s_point=252,frame_num=600,Need_file = 1e10,choice = 'left')
    # Split_tif('/hdd/0/lsy/SMF_0401_dataset_test/test/combine','/hdd/0/lsy/SMF_0401_Dealeddata/test_3H260_v2/dark_h',s_point=252,frame_num=600,Need_file = 1e10,choice = 'right')
    # organize_files('/hdd/0/lsy/SMF_0401_Dealeddata/test_3H260_v2/dark_h')
    # organize_files('/hdd/0/lsy/SMF_0401_Dealeddata/test_3H260_v2/dark')
    
    
    
    # after_align_adjust('/hdd/0/lsy/SMF_0401_beta/train/dark','/hdd/0/lsy/SMF_0401_beta/train/dark_h','/hdd/0/lsy/SMF_0401_beta/combine',ratio=99.99995,scale = 0.90)
    # Split_tif('/hdd/0/lsy/SMF_0401_beta/combine','/hdd/0/lsy/SMF_0401_Dealeddata/train_3H260_v2/dark',s_point=252,frame_num=9,Need_file = 1e10,choice = 'left')
    # Split_tif('/hdd/0/lsy/SMF_0401_beta/combine','/hdd/0/lsy/SMF_0401_Dealeddata/train_3H260_v2/dark_h',s_point=252,frame_num=9,Need_file = 1e10,choice = 'right')
    # organize_files('/hdd/0/lsy/SMF_0401_Dealeddata/train_3H260_v2/dark_h')
    # organize_files('/hdd/0/lsy/SMF_0401_Dealeddata/train_3H260_v2/dark')

    # after_align_adjust('/hdd/0/lsy/SMF_0401_dataset_test/test/dark/s1405_0020.tif','/hdd/0/lsy/SMF_0401_dataset_test/test/dark_h/s1405_0020.tif','/hdd/0/lsy/SMF_0401_dataset_test/test/combine/s1405_0020.tif',ratio=99.99995,scale = 0.90)
    # Split_tif('/hdd/0/lsy/SMF_0401_dataset_test/test/combine/s1405_0020.tif','/hdd/0/lsy/SMF_0401_Dealeddata/test_3H260/dark',s_point=252,frame_num=600,Need_file = 1e10,choice = 'left')
    # Split_tif('/hdd/0/lsy/SMF_0401_dataset_test/test/combine/s1405_0020.tif','/hdd/0/lsy/SMF_0401_Dealeddata/test_3H260/dark_h',s_point=252,frame_num=600,Need_file = 1e10,choice = 'right')
    # for folder in glob.glob('/hdd/0/lsy/SMF_0401_dataset_test/test/dark/*'):
    #     if os.path.basename(folder).split('_')[0] == 's1405':
    #         after_align_adjust(folder,folder.replace('dark','dark_h'),folder.replace('dark','combine'),ratio=99.99995,scale = 0.90)
    #         Split_tif(folder.replace('dark','combine'),'/hdd/0/lsy/SMF_0401_Dealeddata/test_3H260/dark',s_point=252,frame_num=600,Need_file = 1e10,choice = 'left')
    #         Split_tif(folder.replace('dark','combine'),'/hdd/0/lsy/SMF_0401_Dealeddata/test_3H260/dark_h',s_point=252,frame_num=600,Need_file = 1e10,choice = 'right')
    # for folder in glob.glob('/hdd/0/lsy/SMF_0401_beta/train/dark/*'):
    #     if os.path.basename(folder).split('_')[0] == 's1405':
    #         after_align_adjust(folder,folder.replace('dark','dark_h'),folder.replace('dark','combine'),ratio=99.99995,scale = 0.90)
    #         Split_tif(folder.replace('dark','combine'),'/hdd/0/lsy/SMF_0401_Dealeddata/train_3H260/dark',s_point=252,frame_num=9,Need_file = 1e10,choice = 'left')
    #         Split_tif(folder.replace('dark','combine'),'/hdd/0/lsy/SMF_0401_Dealeddata/train_3H260/dark_h',s_point=252,frame_num=9,Need_file = 1e10,choice = 'right')

    # organize_files('/hdd/0/lsy/SMF_0401_Dealeddata/train_3H260/dark_h')
    # organize_files('/hdd/0/lsy/SMF_0401_Dealeddata/train_3H260/dark')
    # organize_files('/hdd/0/lsy/SMF_0401_Dealeddata/test_3H260/dark_h')
    # organize_files('/hdd/0/lsy/SMF_0401_Dealeddata/test_3H260/dark')

    # Split_tif('/hdd/0/lsy/SMF_0401_beta/combine','/hdd/0/lsy/SMF_0401_Dealeddata/train_3H260_5f/dark',s_point=252,frame_num=5,Need_file = 1e10,choice = 'left')
    # Split_tif('/hdd/0/lsy/SMF_0401_beta/combine','/hdd/0/lsy/SMF_0401_Dealeddata/train_3H260_5f/dark_h',s_point=252,frame_num=5,Need_file = 1e10,choice = 'right')
    # organize_files('/hdd/0/lsy/SMF_0401_Dealeddata/train_3H260_5f/dark_h')
    # organize_files('/hdd/0/lsy/SMF_0401_Dealeddata/train_3H260_5f/dark')

    # lanczous 0401
    # after_align_adjust('/hdd/0/lsy/SMF_0401_Lanczos_test/test/dark','/hdd/0/lsy/SMF_0401_Lanczos_test/test/dark_h','/hdd/0/lsy/SMF_0401_Lanczos_test/test/combine',ratio=99.99995,scale = 0.90)
    # after_align_adjust('/hdd/0/lsy/SMF_0401_Lanczos/train/dark','/hdd/0/lsy/SMF_0401_Lanczos/train/dark_h','/hdd/0/lsy/SMF_0401_Lanczos/train/combine',ratio=99.99995,scale = 0.90)
    # Split_tif('/hdd/0/lsy/SMF_0401_Lanczos/train/combine','/hdd/0/lsy/SMF_0401_Dealeddata/train_3H260_lanc_5/dark',s_point=252,frame_num=5,Need_file = 1e10,choice = 'left')
    # Split_tif('/hdd/0/lsy/SMF_0401_Lanczos/train/combine','/hdd/0/lsy/SMF_0401_Dealeddata/train_3H260_lanc_5/dark_h',s_point=252,frame_num=5,Need_file = 1e10,choice = 'right')
    # organize_files('/hdd/0/lsy/SMF_0401_Dealeddata/train_3H260_lanc_5/dark')
    # organize_files('/hdd/0/lsy/SMF_0401_Dealeddata/train_3H260_lanc_5/dark_h')
    
    # Split_tif('/hdd/0/lsy/SMF_0401_Lanczos_test/test/combine','/hdd/0/lsy/SMF_0401_Dealeddata/test_3H260_lanc/dark',s_point=252,frame_num=600,Need_file = 1e10,choice = 'left')
    # Split_tif('/hdd/0/lsy/SMF_0401_Lanczos_test/test/combine','/hdd/0/lsy/SMF_0401_Dealeddata/test_3H260_lanc/dark_h',s_point=252,frame_num=600,Need_file = 1e10,choice = 'right')
    # organize_files('/hdd/0/lsy/SMF_0401_Dealeddata/train_3H260_lanc_5/dark')
    # organize_files('/hdd/0/lsy/SMF_0401_Dealeddata/train_3H260_lanc_5/dark_h')

    # Split_tif('/hdd/0/lsy/SMF_0401_Lanczos_test/test/combine','/hdd/0/lsy/SMF_0401_Dealeddata/test_lanc_1/dark',s_point=252,frame_num=1,Need_file = 1e10,choice = 'left')
    # Split_tif('/hdd/0/lsy/SMF_0401_Lanczos_test/test/combine','/hdd/0/lsy/SMF_0401_Dealeddata/test_lanc_1/dark_h',s_point=252,frame_num=1,Need_file = 1e10,choice = 'right')
    # organize_files('/hdd/0/lsy/SMF_0401_Dealeddata/test_lanc_1/dark')
    # organize_files('/hdd/0/lsy/SMF_0401_Dealeddata/test_lanc_1/dark_h')

    #Cell dataset
    # Crop_tif('/hdd/0/lsy/Cell_dataset/Proc_data/250626_mscarlet_in_vivo',crop_shape=(0,512,11,248))
    # Crop_tif('/hdd/0/lsy/Cell_dataset/Proc_data/20250701clc7-ms3',crop_shape=(0,512,11,248))
    # 400*237*512
    # Bright_baseMax_adjust_group_Perc_parallel('/hdd/0/lsy/Cell_dataset/Proc_data/20250701clc7-ms3_crop','/hdd/0/lsy/Cell_dataset/Proc_data/20250701clc7-ms3_cadj',ratio=99.999999,scale=0.90)
    # Bright_baseMax_adjust_group_Perc_parallel('/hdd/0/lsy/Cell_dataset/Proc_data/250626_mscarlet_in_vivo_crop','/hdd/0/lsy/Cell_dataset/Proc_data/250626_mscarlet_in_vivo_cadj',ratio=99.999999,scale=0.90)
    # Bright_baseMax_adjust_group_Perc_parallel('/hdd/0/lsy/Cell_dataset/Data_from_WuLiSuo/mNeonGreen0907update','/hdd/0/lsy/Cell_dataset/Proc_data/mNeonGreen0907update_adj',ratio=99.999999,scale=0.95)

    # organize_files('/hdd/0/lsy/SMF_0723_data/SMF_0723_NAdata/don/1mM')
    # organize_files('/hdd/0/lsy/SMF_0723_data/SMF_0723_NAdata/acc/1mM')
    # organize_files('/hdd/0/lsy/SMF_0723_data/SMF_0723_NAdata/don/50mM')
    # organize_files('/hdd/0/lsy/SMF_0723_data/SMF_0723_NAdata/acc/50mM')
    # organize_files('/hdd/0/lsy/SMF_0723_data/SMF_0723_NAdata/don/100mM')
    # organize_files('/hdd/0/lsy/SMF_0723_data/SMF_0723_NAdata/acc/100mM')
    # organize_files('/hdd/0/lsy/SMF_0723_data/SMF_0723_NAdata/don/300mM')
    # organize_files('/hdd/0/lsy/SMF_0723_data/SMF_0723_NAdata/acc/300mM')
    # organize_files('/hdd/0/lsy/SMF_0723_data/SMF_0723_NAdata/don/500mM')
    # organize_files('/hdd/0/lsy/SMF_0723_data/SMF_0723_NAdata/acc/500mM')
    # organize_files('/hdd/0/lsy/SMF_0723_data/SMF_0723_NAdata_SA3f/don/1mM')
    # organize_files('/hdd/0/lsy/SMF_0723_data/SMF_0723_NAdata_SA3f/acc/1mM')
    # organize_files('/hdd/0/lsy/SMF_0723_data/SMF_0723_NAdata_SA3f/don/50mM')
    # organize_files('/hdd/0/lsy/SMF_0723_data/SMF_0723_NAdata_SA3f/acc/50mM')
    # organize_files('/hdd/0/lsy/SMF_0723_data/SMF_0723_NAdata_SA3f/don/100mM')
    # organize_files('/hdd/0/lsy/SMF_0723_data/SMF_0723_NAdata_SA3f/acc/100mM')
    # organize_files('/hdd/0/lsy/SMF_0723_data/SMF_0723_NAdata_SA3f/don/300mM')
    # organize_files('/hdd/0/lsy/SMF_0723_data/SMF_0723_NAdata_SA3f/acc/300mM')
    # organize_files('/hdd/0/lsy/SMF_0723_data/SMF_0723_NAdata_SA3f/don/500mM')
    # organize_files('/hdd/0/lsy/SMF_0723_data/SMF_0723_NAdata_SA3f/acc/500mM')
    # for file in scan('/hdd/0/lsy/EMCCD_dataset/20250915_Asplit/6%_laser_power'):
    #     img_dir = os.path.dirname(file)
    #     new_name = file.split('/')[-2] + '.tif'
    #     new_path = os.path.join(img_dir, new_name)
    #     try:
    #         if file != new_path:
    #             os.rename(file, new_path)
    #             print(f"Renamed {file} -> {new_path}")
    #     except FileExistsError:
    #         print(f"目标文件已存在: {new_path}")
    # for file in scan('/hdd/0/lsy/EMCCD_dataset/20250915_Asplit/10%_laser_power'):
    #     img_dir = os.path.dirname(file)
    #     new_name = file.split('/')[-2] + '.tif'
    #     new_path = os.path.join(img_dir, new_name)
    #     try:
    #         if file != new_path:
    #             os.rename(file, new_path)
    #             print(f"Renamed {file} -> {new_path}")
    #     except FileExistsError:
    #         print(f"目标文件已存在: {new_path}")
    # Split_tif('/hdd/0/lsy/EMCCD_dataset/20250915_Asplit/10%_laser_power/acc','/hdd/0/lsy/EMCCD_dataset/20250915_AS3f/10%_laser_power/acc',s_point=None,frame_num=3,Need_file = 1e10,choice='left_x',mode='SlideSplit')
    # Split_tif('/hdd/0/lsy/EMCCD_dataset/20250915_Asplit/10%_laser_power/don','/hdd/0/lsy/EMCCD_dataset/20250915_AS3f/10%_laser_power/don',s_point=None,frame_num=3,Need_file = 1e10,choice='left_x',mode='SlideSplit')


    # Split_tif('/hdd/0/lsy/EMCCD_dataset/20250915_Asplit/6%_laser_power/acc','/hdd/0/lsy/EMCCD_dataset/20250915_AS3f/6%_laser_power/acc',s_point=None,frame_num=3,Need_file = 1e10,choice='left_x',mode='SlideSplit')
    # Split_tif('/hdd/0/lsy/EMCCD_dataset/20250915_Asplit/6%_laser_power/don','/hdd/0/lsy/EMCCD_dataset/20250915_AS3f/6%_laser_power/don',s_point=None,frame_num=3,Need_file = 1e10,choice='left_x',mode='SlideSplit')


    # 0723SCMOS 2binned 800/799*256*900
    # Split_tif('/hdd/0/lsy/sCMOS_723_dataset/2binned','/hdd/0/lsy/sCMOS_723_dataset/250723_sCMOS/sCMOS_723_NAdata_3f/acc',s_point=450,frame_num=3,Need_file = 1e10,choice='left_x')
    # Split_tif('/hdd/0/lsy/sCMOS_723_dataset/2binned','/hdd/0/lsy/sCMOS_723_dataset/250723_sCMOS/sCMOS_723_NAdata_3f/don',s_point=450,frame_num=3,Need_file = 1e10,choice='right_x')
    # Split_tif('/hdd/0/lsy/sCMOS_723_dataset/2binned','/hdd/0/lsy/sCMOS_723_dataset/250723_sCMOS/sCMOS_723_NAdata/acc',s_point=450,frame_num=3,Need_file = 1e10,choice='left_all')
    # Split_tif('/hdd/0/lsy/sCMOS_723_dataset/2binned','/hdd/0/lsy/sCMOS_723_dataset/250723_sCMOS/sCMOS_723_NAdata/don',s_point=450,frame_num=3,Need_file = 1e10,choice='right_all')



    # 1010_sCMOS 3000/800*256*930
    # Split_tif('/hdd/0/lsy/sCMOS_dataset/20251010_sCMOS','/hdd/0/lsy/sCMOS_dataset/20251010_sCMOS_3f/acc',s_point=460,frame_num=3,Need_file = 1e10,choice='left_x')
    # Split_tif('/hdd/0/lsy/sCMOS_dataset/20251010_sCMOS','/hdd/0/lsy/sCMOS_dataset/20251010_sCMOS_3f/don',s_point=470,frame_num=3,Need_file = 1e10,choice='right_x')
    # Split_tif('/hdd/0/lsy/sCMOS_dataset/20251010_sCMOS','/hdd/0/lsy/sCMOS_dataset/20251010_sCMOS_Split/acc',s_point=460,frame_num=3,Need_file = 1e10,choice='left_all')
    # Split_tif('/hdd/0/lsy/sCMOS_dataset/20251010_sCMOS','/hdd/0/lsy/sCMOS_dataset/20251010_sCMOS_Split/don',s_point=470,frame_num=3,Need_file = 1e10,choice='right_all')

    # Split_tif('/hdd/0/lsy/EMCCD_dataset/20251010_Asplit/acc','/hdd/0/lsy/EMCCD_dataset/20251010_AS3f/acc',s_point=None,frame_num=3,Need_file = 1e10,choice='left_x',mode='SlideSplit')
    # Split_tif('/hdd/0/lsy/EMCCD_dataset/20251010_Asplit/don','/hdd/0/lsy/EMCCD_dataset/20251010_AS3f/don',s_point=None,frame_num=3,Need_file = 1e10,choice='right_x',mode='SlideSplit')

    #1022EMCCD 
    #3000*256*512 
    # Bright_baseMax_adjust_group_Perc_parallel('/hdd/0/lsy/EMCCD_dataset/20251022/3000f','/hdd/0/lsy/EMCCD_dataset/20251022_adj/3000f',ratio=99.99998,scale = 0.95)
    # #8000*256*512
    # Bright_baseMax_adjust_group_Perc_parallel('/hdd/0/lsy/EMCCD_dataset/20251022/8000f','/hdd/0/lsy/EMCCD_dataset/20251022_adj/8000f',ratio=99.999992,scale = 0.95)
    # #800*256*512
    # Bright_baseMax_adjust_group_Perc_parallel('/hdd/0/lsy/EMCCD_dataset/20251022/800f','/hdd/0/lsy/EMCCD_dataset/20251022_adj/800f',ratio=99.99992,scale = 0.95)
    # #5000*256*512
    # Bright_baseMax_adjust_group_Perc_parallel('/hdd/0/lsy/EMCCD_dataset/20251022/5000f','/hdd/0/lsy/EMCCD_dataset/20251022_adj/5000f',ratio=99.99999,scale = 0.95)
    # Split_tif('/hdd/0/lsy/EMCCD_dataset/20251022_adj','/hdd/0/lsy/EMCCD_dataset/20251022_A3f/acc',s_point=256,frame_num=3,Need_file = 1e10,choice='left_x')
    # Split_tif('/hdd/0/lsy/EMCCD_dataset/20251022_adj','/hdd/0/lsy/EMCCD_dataset/20251022_A3f/don',s_point=256,frame_num=3,Need_file = 1e10,choice='right_x')
    # # Split_tif('/hdd/0/lsy/EMCCD_dataset/20251022_adj','/hdd/0/lsy/EMCCD_dataset/20251022_AS3f/acc',s_point=256,frame_num=3,Need_file = 1e10,choice='left_x',mode='SlideSplit')
    # # Split_tif('/hdd/0/lsy/EMCCD_dataset/20251022_adj','/hdd/0/lsy/EMCCD_dataset/20251022_AS3f/don',s_point=256,frame_num=3,Need_file = 1e10,choice='right_x',mode='SlideSplit')
    # Split_tif('/hdd/0/lsy/EMCCD_dataset/20251022_adj','/hdd/0/lsy/EMCCD_dataset/20251022_Asplit/acc',s_point=256,frame_num=3,Need_file = 1e10,choice='left_all')
    # Split_tif('/hdd/0/lsy/EMCCD_dataset/20251022_adj','/hdd/0/lsy/EMCCD_dataset/20251022_Asplit/don',s_point=256,frame_num=3,Need_file = 1e10,choice='right_all')

    #1022SCMOS
    # Split_tif('/hdd/0/lsy/sCMOS_dataset/20251022-sCMOS','/hdd/0/lsy/sCMOS_dataset/20251022_sCMOS_3f/acc',s_point=465,frame_num=3,Need_file = 1e10,choice='left_x')
    # Split_tif('/hdd/0/lsy/sCMOS_dataset/20251022-sCMOS','/hdd/0/lsy/sCMOS_dataset/20251022_sCMOS_3f/don',s_point=465,frame_num=3,Need_file = 1e10,choice='right_x')
    #Split_tif('/hdd/0/lsy/sCMOS_dataset/20251022-sCMOS','/hdd/0/lsy/sCMOS_dataset/20251022_sCMOS_Split/acc',s_point=465,frame_num=3,Need_file = 1e10,choice='left_all')
    #Split_tif('/hdd/0/lsy/sCMOS_dataset/20251022-sCMOS','/hdd/0/lsy/sCMOS_dataset/20251022_sCMOS_Split/don',s_point=465,frame_num=3,Need_file = 1e10,choice='right_all')

    # Split_tif('/hdd/0/lsy/EMCCD_dataset/20251010_Asplit/acc','/hdd/0/lsy/EMCCD_dataset/20251010_A3f/acc',s_point=None,frame_num=3,Need_file = 1e10,choice='left_x')
    # Split_tif('/hdd/0/lsy/EMCCD_dataset/20251010_Asplit/don','/hdd/0/lsy/EMCCD_dataset/20251010_A3f/don',s_point=None,frame_num=3,Need_file = 1e10,choice='right_x')

    #0915emccd
    # Bright_baseMax_adjust_group_Perc_parallel('/hdd/0/lsy/EMCCD_dataset/GT2','/hdd/0/lsy/EMCCD_dataset/GT2_adj',ratio=99.99992,scale = 0.95)
    # Split_tif('/hdd/0/lsy/EMCCD_dataset/GT2_adj','/hdd/0/lsy/EMCCD_dataset/20250915_Asplit/GT2/acc',s_point=256,frame_num=3,Need_file = 1e10,choice='left_all')
    # Split_tif('/hdd/0/lsy/EMCCD_dataset/GT2_adj','/hdd/0/lsy/EMCCD_dataset/20250915_Asplit/GT2/don',s_point=256,frame_num=3,Need_file = 1e10,choice='right_all')
    # Split_tif('/hdd/0/lsy/EMCCD_dataset/GT2_adj','/hdd/0/lsy/EMCCD_dataset/20250915_A3f/GT2/acc',s_point=256,frame_num=3,Need_file = 1e10,choice='left_x')
    # Split_tif('/hdd/0/lsy/EMCCD_dataset/GT2_adj','/hdd/0/lsy/EMCCD_dataset/20250915_A3f/GT2/don',s_point=256,frame_num=3,Need_file = 1e10,choice='right_x')

    #1104emccd
    # Bright_baseMax_adjust_group_Perc_parallel('/hdd/0/lsy/EMCCD_dataset/20251104/25%/3000f','/hdd/0/lsy/EMCCD_dataset/20251104_adj/3000f/25%',ratio=99.99998,scale = 0.95)
    # Bright_baseMax_adjust_group_Perc_parallel('/hdd/0/lsy/EMCCD_dataset/20251104/25%/800f','/hdd/0/lsy/EMCCD_dataset/20251104_adj/800f/25%',ratio=99.99992,scale = 0.95)
    # Split_tif('/hdd/0/lsy/EMCCD_dataset/20251104_adj','/hdd/0/lsy/EMCCD_dataset/20251104_AS3f/acc',s_point=256,frame_num=3,Need_file = 1e10,choice='left_x',mode='SlideSplit')
    # Split_tif('/hdd/0/lsy/EMCCD_dataset/20251104_adj','/hdd/0/lsy/EMCCD_dataset/20251104_AS3f/don',s_point=256,frame_num=3,Need_file = 1e10,choice='right_x',mode='SlideSplit')
    # Split_tif('/hdd/0/lsy/EMCCD_dataset/20251104_adj','/hdd/0/lsy/EMCCD_dataset/20251104_Asplit/acc',s_point=256,frame_num=3,Need_file = 1e10,choice='left_all')
    # Split_tif('/hdd/0/lsy/EMCCD_dataset/20251104_adj','/hdd/0/lsy/EMCCD_dataset/20251104_Asplit/don',s_point=256,frame_num=3,Need_file = 1e10,choice='right_all')

    #cell_data 233*100*200 - 233*300*2000
    # Bright_baseMax_adjust_group_Perc_parallel('/hdd/0/lsy/Cell_dataset/Data_from_WuLiSuo/20251030hekkd-tiff','/hdd/0/lsy/Cell_dataset/Proc_data/20251030hekkd-tiff_adj',ratio=99.998,scale=0.95)
    # Split_tif('/hdd/0/lsy/Cell_dataset/Proc_data/20251030hekkd-tiff_adj','/hdd/0/lsy/Cell_dataset/Proc_data/20251030hekkd-tiff_A3f',s_point=None,frame_num=3,Need_file = 1e10,choice='left_x')

    # cell_data 2000*233*479
    # Bright_baseMax_adjust_group_Perc_parallel('/hdd/0/lsy/Cell_dataset/Data_from_WuLiSuo/20251101hekkd-mstaygold-bhq','/hdd/0/lsy/Cell_dataset/Data_from_WuLiSuo/20251101hekkd-mstaygold-bhq_adj',ratio=99.99995,scale=0.95)
    # Split_tif('/hdd/0/lsy/Cell_dataset/Data_from_WuLiSuo/20251101hekkd-mstaygold-bhq_adj','/hdd/0/lsy/Cell_dataset/Proc_data/20251101hekkd-mstaygold-bhq_A3f',s_point=None,frame_num=3,Need_file = 1e10,choice='left_x')

    # EMCCD 1112c min=800*512*256
    # Bright_baseMax_adjust_group_Perc_parallel('/hdd/0/lsy/EMCCD_dataset/20251112','/hdd/0/lsy/EMCCD_dataset/20251112_adj',ratio=99.9999,scale = 0.95)
    # Split_tif('/hdd/0/lsy/EMCCD_dataset/20251112_adj','/hdd/0/lsy/EMCCD_dataset/20251112_A3f/acc',s_point=256,frame_num=3,Need_file = 1e10,choice='left_x')
    # Split_tif('/hdd/0/lsy/EMCCD_dataset/20251112_adj','/hdd/0/lsy/EMCCD_dataset/20251112_A3f/don',s_point=256,frame_num=3,Need_file = 1e10,choice='right_x')
    # Split_tif('/hdd/0/lsy/EMCCD_dataset/20251112_adj','/hdd/0/lsy/EMCCD_dataset/20251112_Asplit/acc',s_point=256,frame_num=3,Need_file = 1e10,choice='left_all')
    # Split_tif('/hdd/0/lsy/EMCCD_dataset/20251112_adj','/hdd/0/lsy/EMCCD_dataset/20251112_Asplit/don',s_point=256,frame_num=3,Need_file = 1e10,choice='right_all')


    #20251112-sCMOS  split\
    # Split_tif('/hdd/0/lsy/sCMOS_dataset/20251112-sCMOS','/hdd/0/lsy/sCMOS_dataset/20251112-sCMOS_3f/acc',s_point=465,frame_num=3,Need_file = 1e10,choice='left_x')
    # Split_tif('/hdd/0/lsy/sCMOS_dataset/20251112-sCMOS','/hdd/0/lsy/sCMOS_dataset/20251112-sCMOS_3f/don',s_point=465,frame_num=3,Need_file = 1e10,choice='right_x')
    # Split_tif('/hdd/0/lsy/sCMOS_dataset/20251112-sCMOS','/hdd/0/lsy/sCMOS_dataset/20251112-sCMOS_Split/acc',s_point=465,frame_num=1,Need_file = 1e10,choice='left_all')
    # Split_tif('/hdd/0/lsy/sCMOS_dataset/20251112-sCMOS','/hdd/0/lsy/sCMOS_dataset/20251112-sCMOS_Split/don',s_point=465,frame_num=1,Need_file = 1e10,choice='right_all')

    #20251104-sCMOS split
    # Split_tif('/hdd/0/lsy/sCMOS_dataset/20251104-sCMOS','/hdd/0/lsy/sCMOS_dataset/20251104-sCMOS_S3f/acc',s_point=465,frame_num=3,Need_file = 1e10,choice='left_x',mode='SlideSplit')
    # Split_tif('/hdd/0/lsy/sCMOS_dataset/20251104-sCMOS','/hdd/0/lsy/sCMOS_dataset/20251104-sCMOS_S3f/don',s_point=465,frame_num=3,Need_file = 1e10,choice='right_x',mode='SlideSplit')
    # Split_tif('/hdd/0/lsy/sCMOS_dataset/20251104-sCMOS','/hdd/0/lsy/sCMOS_dataset/20251104-sCMOS_3f/acc',s_point=465,frame_num=3,Need_file = 1e10,choice='left_x')
    # Split_tif('/hdd/0/lsy/sCMOS_dataset/20251104-sCMOS','/hdd/0/lsy/sCMOS_dataset/20251104-sCMOS_3f/don',s_point=465,frame_num=3,Need_file = 1e10,choice='right_x')
    # Split_tif('/hdd/0/lsy/sCMOS_dataset/20251104-sCMOS','/hdd/0/lsy/sCMOS_dataset/20251104-sCMOS_Split/acc',s_point=465,frame_num=1,Need_file = 1e10,choice='left_all')
    # Split_tif('/hdd/0/lsy/sCMOS_dataset/20251104-sCMOS','/hdd/0/lsy/sCMOS_dataset/20251104-sCMOS_Split/don',s_point=465,frame_num=1,Need_file = 1e10,choice='right_all')

    #20251121hekkd-mng 1800*230*460
    # Bright_baseMax_adjust_group_Perc_parallel('/hdd/0/lsy/Cell_dataset/Data_from_WuLiSuo/20251121hekkd-mng','/hdd/0/lsy/Cell_dataset/Data_from_WuLiSuo/20251121hekkd-mng_adj',ratio=99.9999,scale=0.95)
    # Split_tif('/hdd/0/lsy/Cell_dataset/Data_from_WuLiSuo/20251121hekkd-mng_adj','/hdd/0/lsy/Cell_dataset/Data_from_WuLiSuo/20251121hekkd-mng_AS3f',s_point=None,frame_num=3,Need_file = 1e10,choice='left_x',mode='SlideSplit')

    #251221_mNG 1200*239*468
    Bright_baseMax_adjust_group_Perc_parallel('/hdd/0/lsy/Cell_dataset/Data_from_WuLiSuo/251221_mNG','/hdd/0/lsy/Cell_dataset/Proc_data/251221_mNG_adj',ratio=99.9999,scale=0.95)
    Split_tif('/hdd/0/lsy/Cell_dataset/Proc_data/251221_mNG_adj','/hdd/0/lsy/Cell_dataset/Proc_data/251221_mNG_A3f',s_point=None,frame_num=3,Need_file = 1e10,choice='left_x')
