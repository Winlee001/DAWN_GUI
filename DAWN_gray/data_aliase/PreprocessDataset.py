import sys
import os

from omegaconf import OmegaConf
import numpy as np
import tifffile
import glob
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import glob
import cv2

from os.path import join as opj
from os.path import dirname as opd
from os.path import basename as opb
from os.path import splitext as ops

from skimage import exposure

import Calibration

"""Orientation

 *--------->  Y
 |
 |
 |
 v
 X

"""


def crop_img(img, roi):  # 裁剪图片
    return img[roi[0]: roi[1], roi[2]: roi[3]]


def normalize_img(img):  # 标准化图片
    l_val = np.percentile(img, 0)
    h_val = np.percentile(img, 100)
    img = np.clip((img - l_val) / (h_val - l_val) * 255, 0, 255).astype(np.uint8)
    return img


def calc(
        chal_roi,
        avg_data,
        save_path=None,
        chah_original=[200, 512, 0, 300],  # 裁剪区间
        chbl_original=[0, 300, 200, 512],
        chbh_original=[200, 512, 200, 512],
):
    chal = crop_img(avg_data, chal_roi)  # 裁剪
    chbl_original = crop_img(avg_data, chbl_original)

    chal_img = normalize_img(chal)  # 标准化
    chbl_original = normalize_img(chbl_original)

    # 纸包鱼尝试预处理
    left_half = chal
    right_half = chbl_original

    # left_half = (left_half - np.mean(left_half)) / np.std(left_half) - 3
    # right_half = (right_half - np.mean(right_half)) / np.std(right_half) - 3
    # left_half = np.maximum(0, left_half)*10
    # right_half = np.maximum(0, right_half)*10
    # left_half = np.tanh(left_half)
    # right_half = np.tanh(right_half)
    # left_half = exposure.equalize_hist(left_half)
    # right_half = exposure.equalize_hist(right_half)
    left_half = left_half / np.max(left_half) * 255
    right_half = right_half / np.max(right_half) * 255
    left_half = left_half.astype('uint8')
    right_half = right_half.astype('uint8')


    chal_img = left_half
    chbl_original = right_half
    # END

    chbl_img, H_bl, status, chal_kp_img, chbl_kp_img,error_p = Calibration.siftImageAlignment(
        chal_img, chbl_original
    )  # sift+BFM，[img2经变换变成target域对应的imgout，单应性矩阵，匹配成功的特征点，绘制了关键点的图片]

    plt.figure(figsize=[10, 20])
    plt.subplot(121)
    plt.gca().set_title("CHAL")
    plt.imshow(chal_img)
    plt.grid()
    plt.subplot(122)
    plt.gca().set_title("CHBL")
    plt.imshow(chbl_img)
    plt.grid()

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path + ".png", format="png")
        plt.close()

    plt.figure(figsize=[10, 20])
    plt.subplot(121)
    plt.gca().set_title("CHAL")
    plt.imshow(chal_kp_img)
    plt.subplot(122)
    plt.gca().set_title("CHBL")
    plt.imshow(chbl_kp_img)
    plt.grid()

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path + "_keypoints.png", format="png")
        plt.close()

    # print("chah_roi = ", chah_roi)
    # print("chal_roi = ", chal_roi)
    # print("chbh_roi = ", chbh_roi)
    # print("chbl_roi = ", chbl_roi)

    return H_bl,status,error_p,chal_img,chbl_img # 单应性矩阵，H*target = img2

def draw_chal_chbl(chal_img,chbl_img):
    plt.figure(figsize=[10, 20])
    plt.subplot(121)
    plt.gca().set_title("CHAL")
    plt.imshow(chal_img)
    plt.grid()
    plt.subplot(122)
    plt.gca().set_title("CHBL")
    plt.imshow(chbl_img)
    plt.grid()
    plt.show()

def robust_calc(
        sample_file_glob, save_path, chal_roi, chbl_original
):  # chal_roi, chbl_original裁剪区间
    data = {}
    # 尝试，直到max_trial个样本计算的结果相似
    for _ in range(100):
        max_trial = 3
        chbl_trials = np.zeros([max_trial, 3, 3])

        for try_id in range(max_trial):  # 生成三个结果
            each_trial = 0
            while True:
               
                try:
                    # randomly fetch image
                    path_list = sorted(glob.glob(sample_file_glob))
                    
                    # path = path_list[np.random.randint(0, len(path_list))]  # 随机抽取一个tif文件
                    path = path_list[each_trial%len(path_list)]  # 顺序抽取tif文件，保证快速收敛
                    print(path,path_list)
                    # preview
                    data_3d = tifffile.imread(path)
                    # use_slices = np.random.randint(0, data_3d.shape[0] - 100)
                    # data_3d = data_3d[use_slices:use_slices + 100]
                    data_3d = data_3d[:300]  # 截取前300张图
                    plt.imshow(np.max(data_3d, axis=0))
                    plt.savefig(opj(save_path, f"prev_{try_id:d}" + ".png"), format="png")
                    plt.close()

                    # calibration
                    # chal_roi = [40, 220, 40, 200, ]
                    #######mip = np.max(data_3d, axis=0)
                    mip = np.mean(data_3d, axis=0)

                    # mip = tifffile.imread("/data2/cyx/dataset/20240506-sCMOS/1111_dsdna_manual_calibration.tif")[:, :, 0]
                    H_bl,status,error_p,chal_img,chbl_img = calc(
                        chal_roi,
                        mip,
                        # save_path=opj(save_path, f"calib_{try_id:d}"),
                        save_path=save_path,
                        chbl_original=chbl_original,
                    )  # 单应性矩阵，H*target = img2，H*chal = chbl
                    alpha = np.max(np.abs(H_bl - np.eye(3)))  # I为不变，若差距太大则不合理
                    alpha_diag = np.max(np.diagonal(np.abs(H_bl - np.eye(3))))  # I为不变，若差距太大则不合理
                    #越高越好
                    inlier_ratio = np.sum(status) / len(status)
                    if (alpha < 500) & (alpha_diag < 0.01):
                        draw_chal_chbl(chal_img,chbl_img)
                        break
                    print(H_bl)
                except Exception as e:
                    print(e)
                    alpha = 1e9
                    alpha_diag = 1e9
                    inlier_ratio = 1e9
                each_trial += 1
                print(f"Try again... ({alpha:.4f})({alpha_diag:.4f})({inlier_ratio:.4f})")

            chbl_trials[try_id] = H_bl




        # check if all trials have the same result
        all_std = np.sum(np.std(chbl_trials, axis=0))  # 对每个的标准差求和

        chbl_h_mat = np.mean(chbl_trials, axis=0)  # 对每个求均值

        print(f"all std: {all_std:.3f}")
        if all_std < 10:
            data["chal_roi"] = chal_roi
            data["chbl_roi"] = chbl_original
            data["chbl_h_mat"] = chbl_h_mat.tolist()

            print("chal_roi = ", chal_roi)
            print("chbl_h_mat = ", chbl_h_mat)
            print(f'all_std = {all_std}')
            print(f'alpha = {alpha}')
            print(f'alpha_diag = {alpha_diag}')
            print(f'inlier_ratio = {inlier_ratio}')
            print('error_p = ',error_p)
            break
        else:
            # print("Trials with different results.")
            print("Try again...")

    return data


def crop_img(img, roi):
    return img[roi[0]: roi[1], roi[2]: roi[3]]


def calc_calib_img(
        chal_roi,
        avg_data,
        chbl_roi,
        chbl_h_mat,
):
    chal = crop_img(avg_data, chal_roi)
    chbl_img = crop_img(avg_data, chbl_roi)

    chal_img = chal

    chbl_img = cv2.warpPerspective(
        chbl_img,
        np.array(chbl_h_mat).astype(float),
        (chal.shape[1], chal.shape[0]),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
    )

    return chal_img, chbl_img


def make_dataset(raw_data_exp, dataset_root, **args):
    data_block_path_list = glob.glob(raw_data_exp)
    np.random.shuffle(data_block_path_list)  # 随机打乱

    train_ratio = 0.8

    train_size = int(len(data_block_path_list) * train_ratio)
    for path_idx in range(len(data_block_path_list)):
        try:
            data_block_path = data_block_path_list[path_idx]
            print(data_block_path)
            data_block = tifffile.imread(data_block_path)
            d, h, w = data_block.shape
            dir_name = "_".join(data_block_path.split(os.sep)[-3:])
            if path_idx < train_size:
                save_dir = opj(dataset_root, "train", dir_name)
            else:
                save_dir = opj(dataset_root, "test", dir_name)
            print("Original mean val: ", np.mean(data_block))
            print("Original max val: ", np.max(data_block))

            img_noise_median = np.percentile(data_block, 50)
            print("Img noise median: ", img_noise_median)
            for d_i in range(d):
                # slices
                img2d = data_block[d_i]
                chal, chbl = calc_calib_img(avg_data=img2d, **args)

                ## Preprocess
                chal = np.clip(chal - img_noise_median, 0, 65535).astype(np.uint16)
                chbl = np.clip(chbl - img_noise_median, 0, 65535).astype(np.uint16)

                os.makedirs(opj(save_dir, "chal"), exist_ok=True)
                os.makedirs(opj(save_dir, "chbl"), exist_ok=True)
                tifffile.imwrite(opj(save_dir, "chal", f"{d_i:06d}.tif"), chal)
                tifffile.imwrite(opj(save_dir, "chbl", f"{d_i:06d}.tif"), chbl)

            # mean of z stacks
            zmean = np.mean(data_block, axis=0)
            chal, chbl = calc_calib_img(avg_data=zmean, **args)

            mip_gain = 60000 / np.percentile(chal, 99)
            chal = np.clip(chal * mip_gain, 0, 65535).astype(np.uint16)
            chbl = np.clip(chbl * mip_gain, 0, 65535).astype(np.uint16)
            os.makedirs(opj(save_dir, "mip"), exist_ok=True)
            tifffile.imwrite(opj(save_dir, "mip", "chal.tif"), chal)
            tifffile.imwrite(opj(save_dir, "mip", "chbl.tif"), chbl)
        except Exception as e:
            print(e)


if __name__ == "__main__":
    np.random.seed(0)
    recalc_calibration = False
    base_path = f"/data4/zsq/20241021_NoiseData4train/s1109_0000/*.tif"
    # read_path = []
    test_path = f"/data4/zsq/20241021_NoiseData4train/s1109_0000/s1109_0000.tif"
    # for path in os.listdir(base_path):
    #     read_path.append(os.path.join(base_path,path))

    all_data = robust_calc(
        sample_file_glob=base_path,
        # save_path=f"DataPreprocess/outputs/Data20240506-sCMOS_{iden:s}",
        save_path=f"/ssd/1/zby/",
        # chal_roi=[250, 1900, 0, 900, ],
        # chbl_original=[0, 2048, 900, 2048, ],
        # chal_roi=[128, 384, 0, 256, ],
        # chbl_original=[128, 384, 256, 512, ],
        chal_roi=[0, 512, 0, 256, ],
        chbl_original=[0, 512, 256, 512, ],
    )  # chal_roi, chbl_original裁剪区间
    print(all_data)
    H = np.array(all_data["chbl_h_mat"])

    img = tifffile.imread(test_path)
    # img1 = img[0, 128:384, 0:256]
    # img2 = img[0, 128:384, 256:512]
    img1 = img[0, 0:512, 0:256]
    img2 = img[0, 0:512, 256:512]

    imgOut = cv2.warpPerspective(
            img2,
            H,
            (img1.shape[1], img1.shape[0]),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        )  # 透视变换 H*target = img2,img2经变换变成target域对应的imgout

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(img1, cmap='gray')
    plt.title(f'x')
    plt.subplot(1, 2, 2)
    plt.imshow(img2, cmap='gray')
    plt.title(f'y')

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(img1, cmap='gray')
    plt.title(f'x')
    plt.subplot(1, 2, 2)
    plt.imshow(imgOut, cmap='gray')
    plt.title(f'y\'')

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(img2, cmap='gray')
    plt.title(f'y')
    plt.subplot(1, 2, 2)
    plt.imshow(imgOut, cmap='gray')
    plt.title(f'y\'')

    plt.show()