#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 16:37:02 2019
SIFT
@author: youxinlin
"""

import numpy as np
import cv2
import time
import tifffile
from matplotlib import pyplot as plt


def compute_reprojection_error(ptsA, ptsB, H):
    # Step 1: 使用 OpenCV 自带的透视变换函数
    projected_ptsA = cv2.perspectiveTransform(ptsA, H)  # shape: (N, 1, 2)

    # Step 2: 计算欧氏距离（reprojection error）
    error = np.linalg.norm(projected_ptsA - ptsB, axis=2)  # shape: (N, 1)

    # Step 3: 返回平均误差和每个点误差
    mean_error = np.mean(error)
    return mean_error, error  # error 是 (N, 1)，表示每个点的误差


def style_transfer(image, ref):
    out = np.zeros_like(ref)
    _, _ = image.shape

    hist_img, _ = np.histogram(image, 256)
    hist_ref, _ = np.histogram(ref, 256)
    cdf_img = np.cumsum(hist_img)
    cdf_ref = np.cumsum(hist_ref)

    for j in range(256):
        tmp = abs(cdf_img[j] - cdf_ref)
        tmp = tmp.tolist()
        idx = tmp.index(min(tmp))  # 找出tmp中最小的数，得到这个数的索引
        out[ref == j] = idx
    return out


def sift_kp(image):  # sift算法
    # gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(image, None, )  # 关键点，描述符
    kp_image = cv2.drawKeypoints(image, kp, None)  # 画出关键点
    return kp_image, kp, des


def get_good_match(des1, des2):  # 获取较好的匹配
    bf = cv2.BFMatcher()  # 暴力匹配
    matches = bf.knnMatch(des1, des2, k=2)  # 匹配到的最接近的两个特征点
    good_match_idx = np.argsort([m.distance / (n.distance + 1e-6) for m, n in matches])  # 两个候选特征点差距尽量大，排序，索引
    good = [list(matches)[i][0] for i in good_match_idx[:20]]  # 获取前40个差距大的作为匹配的点
    return good


def siftImageAlignment(target, img2):
    # img2 = style_transfer(img2, target)

    target_kp_img, kp1, des1 = sift_kp(target)  # 创建并获取[绘制了关键点的图片，关键点，描述符]
    img2_kp_img, kp2, des2 = sift_kp(img2)
    goodMatch = get_good_match(des1, des2)  # 获取前40个差距大的作为匹配的点

    # # 绘制匹配点
    # matched_image = cv2.drawMatches(
    #     target, kp1,
    #     img2, kp2,
    #     goodMatch,  # 绘制前 20 个匹配点
    #     None,
    #     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    # )
    # # 展示配对结果
    # plt.figure()
    # plt.imshow(matched_image)
    # plt.show()

    if len(goodMatch) > 4:
        ptsA = np.float32([kp1[m.queryIdx].pt for m in goodMatch]).reshape(-1, 1, 2)  # 关键点
        ptsB = np.float32([kp2[m.trainIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
        ransacReprojThreshold = 4  # 仅在使用RANSAC方法时有用，表示一个点到对应点的投影之间的最大允许距离
        H, status = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, ransacReprojThreshold)  # 找到两个图像之间的单应性矩阵
        # H, status = cv2.findTransformECC(ptsA, ptsB, cv2.FM_RANSAC, ransacReprojThreshold)
        # 其中H为求得的单应性矩阵矩阵
        # status则返回一个列表来表征匹配成功的特征点。
        # ptsA,ptsB为关键点
        # cv2.RANSAC, ransacReprojThreshold这两个参数与RANSAC有关

        # compute_reprojection_error(ptsA, ptsB, H)  # 计算重投影误差
        error_p = compute_reprojection_error(ptsA, ptsB, H)[0]  # 计算重投影误差
        error_before = compute_reprojection_error(ptsA,ptsB,np.eye(3))[0]  # 计算变换前误差
        print(f'投影前误差:{error_before}；重投影误差 :{error_p}')  # 打印两个误差
       

        imgOut = cv2.warpPerspective(
            img2,
            H,
            (target.shape[1], target.shape[0]),
            flags=cv2.INTER_LANCZOS4 + cv2.WARP_INVERSE_MAP,
        )  # Lancous透视变换 H*target = img2,img2经变换变成target域对应的imgout
    
        # imgOut = cv2.warpPerspective(
        #     img2,
        #     H,
        #     (target.shape[1], target.shape[0]),
        #     flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        # )  # 透视变换 H*target = img2,img2经变换变成target域对应的imgout
        
    return imgOut, H, status, target_kp_img, img2_kp_img,error_p


if __name__ == "__main__":
    img1 = tifffile.imread(
        "dataset/Dataset_0320/20230320_S5101/train/20230220_Dai_Final_S5101_0016_S5101_0016.tif/mip/chah.tif"
    )
    img1 = (img1 / 65535 * 255).astype(np.uint8)
    img2 = tifffile.imread(
        "dataset/Dataset_0320/20230320_S5101/train/20230220_Dai_Final_S5101_0016_S5101_0016.tif/mip/chbh.tif"
    )
    img2 = (img2 / 65535 * 255).astype(np.uint8)

    while img1.shape[0] > 1000 or img1.shape[1] > 1000:
        img1 = cv2.resize(img1, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    while img2.shape[0] > 1000 or img2.shape[1] > 1000:
        img2 = cv2.resize(img2, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

    result, _, _ = siftImageAlignment(img1, img2)
    allImg = np.concatenate((img1, img2, result), axis=1)

    tifffile.imsave("DataPreprocess/outputs/img1.tif", img1)
    tifffile.imsave("DataPreprocess/outputs/img2.tif", img2)
    tifffile.imsave("DataPreprocess/outputs/res.tif", result)

    # #cv2.imshow('Result',allImg)
    # if cv2.waitKey(2000) & 0xff == ord('q'):
    #     cv2.destroyAllWindows()
    #     cv2.waitKey(1)
