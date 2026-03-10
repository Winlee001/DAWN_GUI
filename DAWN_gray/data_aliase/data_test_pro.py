import os

import cv2
import matplotlib.pyplot as plt
import numpy
import numpy as np
import torch
from torch.utils.data import Dataset
import tifffile as tf
from pathlib import Path
import random
from PIL import Image

from PreprocessDataset import *


def get_files_by_extension(directory, extension):
    # 创建 Path 对象
    directory_path = Path(directory)
    # 筛选出指定类型的文件
    specified_files = list(directory_path.glob(f'*{extension}'))
    return [file.name for file in specified_files]



if __name__ == '__main__':
    # dirs = '../../../../data4/zsq/noist_HJ/20240606_shuqi/s1501_0000/s1501_0000.tif'
    # dirs = '../../../../data4/zsq/noist_HJ/20240606_shuqi/s1503_0000/s1503_0000.tif'
    # dirs = '../../../../data4/zsq/noist_HJ/20240606_shuqi/s2601_0000/s2601_0000.tif'
    np.random.seed(0)
    recalc_calibration = False

    Test_path_list = [
    's1205_0001',
    's1206_0001',
    's1207_0001',
    's1209_0001',
    's1210_0001',
    's1211_0001',
    's1305_0001',
    's1306_0001',
    's1307_0001',
    's1309_0001',
    's1310_0001',
    's1311_0001',
    's1405_0001',
    's1406_0001',
    's1407_0001',
    's1409_0001',
    's1410_0001',
    's1411_0001',
    's1505_0001',
    's1506_0001',
    's1507_0001',
    's1509_0001',
    's1510_0001',
    's1511_0001',
    ]
    # Test_path_list = [
    # 's1305_0001',
    # ],

    path_list = [
    # 's1205_0000',
    # 's1206_0000',
    # 's1207_0000',
    # 's1209_0000',
    # 's1210_0000',
    # 's1211_0000',
    's1305_0000',
    # 's1306_0000',
    # 's1307_0000',
    # 's1309_0000',
    # 's1310_0000',
    # 's1311_0000',
    # 's1405_0000',
    # 's1406_0000',
    # 's1407_0000',
    # 's1409_0000',
    # 's1410_0000',
    # 's1411_0000',
    # 's1505_0000',
    # 's1506_0000',
    # 's1507_0000',
    # 's1509_0000',
    # 's1510_0000',
    # 's1511_0000',
    ]
    path_list = [
        's1105_0000',
        's1106_0000',
        's1107_0000',
        's1109_0000',
        's1110_0000',
        's1111_0000']
    
    # dark_list = path_list # pathlist均为dark，只调整dark
    dark_list = [
        's1205_0000',
        's1206_0000',
        's1207_0000',
        's1209_0000',
        's1210_0000',
        's1211_0000',
        's1305_0000',
        's1306_0000',
        's1307_0000',
        's1309_0000',
        's1310_0000',
        's1311_0000'
        ]
    
    path_dark = 'dark'
    bright_list = [
        's1105_0000',
        's1106_0000',
        's1107_0000',
        's1109_0000',
        's1110_0000',
        's1111_0000']
    path_bright = 'bright'
    path_ex = '*.tif'
    # base_path = f"/data4/zsq/20241021_NoiseData4train"





    # base_save = '/ssd/1/zby/dataset/20241021_NoiseData4train_all'
    # base_path = f"/hdd/0/lsy/SMF_val_filter"
    base_path = f'/hdd/0/lsy/20250401_Noisy_data4Train'#训练集位置
    # base_path = '/hdd/0/lsy/debug_alignment/debug_align'

    # base_path_test = '/hdd/0/lsy/debug_alignment/debug_save'
    base_path_test = f'/hdd/0/lsy/20250401_Noisy_data4Test'#测试集位置
    # base_save = '/hdd/0/lsy/SMF_dataset/SMF_full_dataset'
    # base_save = '/hdd/0/lsy/SMF_0401_dataset'
    # base_save = '/hdd/0/lsy/SMF_0401_3H260'
    # base_save = '/hdd/0/lsy/SMF_0401_beta'
    # base_save = '/hdd/0/lsy/SMF_0401_Lanczos'#训练集配准后保存位置
    base_save = '/hdd/0/lsy/SMF_0401_Lanczos_bright'#bright_train配准后保存位置
    # base_save = '/hdd/0/lsy/debug_alignment/debug_save'
    # base_save_test = '/hdd/0/lsy/SMF_0401_dataset_test'
    # base_save_test = '/hdd/0/lsy/SMF_0401_Lanczos_test'#测试集依靠训练集配准结果保存位置
    base_save_test = '/hdd/0/lsy/SMF_0401_Lanczos_bright_test'#bright_test依靠训练集配准结果保存位置
    # base_save_test = '/hdd/0/lsy/debug_alignment/debug_save_test'

    for p in path_list:
        path = os.path.join(base_path, p, path_ex)
        img_path = get_files_by_extension(os.path.join(base_path, p), '.tif')
        test_path = os.path.join(base_path, p, img_path[0])

        all_data = robust_calc(
            sample_file_glob=path,
            # save_path=f"DataPreprocess/outputs/Data20240506-sCMOS_{iden:s}",
            # save_path=f"/ssd/1/zby",
            # save_path=f"/hdd/0/lsy/SMF_dataset/Cali_val",
            save_path = base_save,
            # chal_roi=[250, 1900, 0, 900, ],
            # chbl_original=[0, 2048, 900, 2048, ],
            # chal_roi=[128, 384, 0, 256, ],
            # chbl_original=[128, 384, 256, 512, ],
            # chal_roi=[0, 512, 0, 256, ],
            # chbl_original=[0, 512, 256, 512, ],
            # chal_roi=[0, 512, 0, 243, ],
            # chbl_original=[0, 512, 243, 512, ],
            # chal_roi=[0, 256, 256, 512, ],#target
            # chbl_original=[0, 256, 0, 256, ],
            chal_roi = [0, 256,260,512 ],  # target
            chbl_original = [0,256,0,260],  # source
            # chal_roi=[0, 256, 0, 256, ],  # target
            # chbl_original=[0, 256, 256, 512],  # source

        )  # chal_roi, chbl_original裁剪区间
        print(all_data)
        H = np.array(all_data["chbl_h_mat"])

        img = tifffile.imread(test_path)
        # img1 = img[0, 128:384, 0:256]
        # img2 = img[0, 128:384, 256:512]
        # img1 = img[0, 0:512, 0:256]
        # img2 = img[0, 0:512, 256:512]
        # img1 = img[0, 0:512, 0:243]
        # img2 = img[0, 0:512, 243:512]
        # img1 = img[0, 0:256,0:256]
        # img2 = img[0, 0:256, 256:512]
        img1 = img[0, 0:256, 0:260]
        img2 = img[0, 0:256, 260:512]   

        # imgOut = cv2.warpPerspective(
        #     img1,
        #     H,
        #     (img2.shape[1], img2.shape[0]),
        #     flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        # )  # 透视变换 H*target = img1,img1经变换变成target域对应的imgout
        imgOut = cv2.warpPerspective(
            img1,
            H,
            (img2.shape[1], img2.shape[0]),
            flags=cv2.INTER_LANCZOS4 + cv2.WARP_INVERSE_MAP,)
    #   Lanczos透视变换 H*target = img2,img2经变换变成target域对应的imgout

        plt.figure(figsize = [15,30])
        plt.subplot(1, 2, 1)
        plt.imshow(img1, cmap='gray')
        plt.title(f'x')
        plt.subplot(1, 2, 2)
        plt.imshow(img2, cmap='gray')
        plt.title(f'y')

        plt.figure(figsize = [15,30])
        plt.subplot(1, 2, 1)
        plt.imshow(imgOut, cmap='gray')
        plt.title(f'x\'')
        plt.subplot(1, 2, 2)
        plt.imshow(img2, cmap='gray')
        plt.title(f'y')
        # plt.figure()
        # plt.subplot(1, 2, 1)
        # plt.imshow(img2, cmap='gray')
        # plt.title(f'x')
        # plt.subplot(1, 2, 2)
        # plt.imshow(img1, cmap='gray')
        # plt.title(f'y')

        # plt.figure()
        # plt.subplot(1, 2, 1)
        # plt.imshow(imgOut, cmap='gray')
        # plt.title(f'x\'')
        # plt.subplot(1, 2, 2)
        # plt.imshow(img1, cmap='gray')
        # plt.title(f'y')

        plt.show()
        #test
        Modfy = lambda x: x[:-1]+'1'
        p_test = Modfy(p)

        # 调整数据集
        if p in dark_list:
            path_kind = path_dark
        elif p in bright_list:
            path_kind = path_bright
        else:
            path_kind = 'error'

        img_dir = os.path.join(base_path, p)
        img_dir_save = os.path.join(base_save, 'train', path_kind)
        img_path = get_files_by_extension(img_dir, '.tif')

        # 测试集保存
        img_dir_test = os.path.join(base_path_test, p_test)
        img_dir_save_test = os.path.join(base_save_test, 'test', path_kind)
        img_path_test = get_files_by_extension(img_dir_test, '.tif')

        print(img_dir)

        for i in range(len(img_path)):
            dir_i = os.path.join(img_dir, img_path[i])
            #TODO 
            print(dir_i)
            img = tf.imread(dir_i)
            if path_kind == path_dark:
                if not os.path.exists(os.path.join(img_dir_save, img_path[i], 'chal')):
                    os.makedirs(os.path.join(img_dir_save, img_path[i], 'chal'))
                if not os.path.exists(os.path.join(img_dir_save + '_h', img_path[i], 'chal')):
                    os.makedirs(os.path.join(img_dir_save + '_h', img_path[i], 'chal'))
            else:
                if not os.path.exists(os.path.join(img_dir_save, img_path[i], 'chah')):
                    os.makedirs(os.path.join(img_dir_save, img_path[i], 'chah'))
                if not os.path.exists(os.path.join(img_dir_save + '_h', img_path[i], 'chah')):
                    os.makedirs(os.path.join(img_dir_save + '_h', img_path[i], 'chah'))
                print(i)
            for j in range(img.shape[0]):
                # res_img = img[j, 128:384, 256:512]
                # res_img = img[j, 0:512, 256:512]
                # res_img = img[j, 0:512, 243:512]
                # res_img_l = img[j, 0:256, 0:256]
                # res_img_r = img[j, 0:256, 256:512]
                res_img_l = img[j, 0:256, 0:260]
                res_img_r = img[j, 0:256, 260:512]
                res_img_l_a = cv2.warpPerspective(res_img_l, H, (res_img_r.shape[1], res_img_r.shape[0]),
                                                flags=cv2.INTER_LANCZOS4 + cv2.WARP_INVERSE_MAP)  # Lanczos变换 透视变换 H*target = img2,img2经变换变成target域对应的imgout
                # res_img_l_a = cv2.warpPerspective(res_img_l, H, (res_img_r.shape[1], res_img_r.shape[0]),
                #                                 flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)  # 透视变换 H*target = img2,img2经变换变成target域对应的imgout
                # res_img = img[j, 128:384, 0:256]
                # res_img = img[j, 0:512, 0:256]
                #TODO Crop
                res_img = res_img_l
                
                if path_kind == path_dark:
                    tf.imwrite(os.path.join(img_dir_save, img_path[i], 'chal', '{}.tif'.format(j)), res_img_l_a)
                    tf.imwrite(os.path.join(img_dir_save + '_h', img_path[i], 'chal', '{}.tif'.format(j)), res_img_r)
                else:
                    tf.imwrite(os.path.join(img_dir_save, img_path[i], 'chah', '{}.tif'.format(j)), res_img_l_a)
                    tf.imwrite(os.path.join(img_dir_save + '_h', img_path[i], 'chah', '{}.tif'.format(j)), res_img_r)
                if j % 100 == 0:
                    print(j)
            print(i)

        #测试集保存
        for i in range(len(img_path_test)):
            dir_i = os.path.join(img_dir_test, img_path_test[i])
            #TODO 
            print(dir_i)
            img = tf.imread(dir_i)
            if path_kind == path_dark:
                if not os.path.exists(os.path.join(img_dir_save_test, img_path_test[i], 'chal')):
                    os.makedirs(os.path.join(img_dir_save_test, img_path_test[i], 'chal'))
                if not os.path.exists(os.path.join(img_dir_save_test + '_h', img_path_test[i], 'chal')):
                    os.makedirs(os.path.join(img_dir_save_test + '_h', img_path_test[i], 'chal'))
            else:
                if not os.path.exists(os.path.join(img_dir_save_test, img_path_test[i], 'chah')):
                    os.makedirs(os.path.join(img_dir_save_test, img_path_test[i], 'chah'))
                if not os.path.exists(os.path.join(img_dir_save_test + '_h', img_path_test[i], 'chah')):
                    os.makedirs(os.path.join(img_dir_save_test + '_h', img_path_test[i], 'chah'))
                print(i)
            for j in range(img.shape[0]):
                res_img_l = img[j, 0:256, 0:260]
                res_img_r = img[j, 0:256, 260:512]
                res_img_l_a = cv2.warpPerspective(res_img_l, H, (res_img_r.shape[1], res_img_r.shape[0]),
                                    flags=cv2.INTER_LANCZOS4 + cv2.WARP_INVERSE_MAP)  # Lanczos变换 透视变换 H*target = img2,img2经变换变成target域对应的imgout
                # res_img_l_a = cv2.warpPerspective(res_img_l, H, (res_img_r.shape[1], res_img_r.shape[0]),
                #                                 flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                res_img = res_img_l
                
                if path_kind == path_dark:
                    tf.imwrite(os.path.join(img_dir_save_test, img_path_test[i], 'chal', '{}.tif'.format(j)), res_img_l_a)
                    tf.imwrite(os.path.join(img_dir_save_test + '_h', img_path_test[i], 'chal', '{}.tif'.format(j)), res_img_r)
                else:
                    tf.imwrite(os.path.join(img_dir_save_test, img_path_test[i], 'chah', '{}.tif'.format(j)), res_img_l_a)
                    tf.imwrite(os.path.join(img_dir_save_test + '_h', img_path_test[i], 'chah', '{}.tif'.format(j)), res_img_r)
                if j % 100 == 0:
                    print(j)
            print(i)
    print('fine')
