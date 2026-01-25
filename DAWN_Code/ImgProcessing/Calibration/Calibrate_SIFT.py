from skimage import io
import cv2
import os
from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt

def edge(img):
    #LoG算子
    gaussian = cv2.GaussianBlur(img, (5,5), 0)
    dst = cv2.Laplacian(gaussian, cv2.CV_16S, ksize = 3) 
    LoG = cv2.convertScaleAbs(dst)
    return LoG

def calibrate_SIFT(img1, img2):
    img1 = np.array(img1).astype(float)
    img2 = np.array(img2).astype(float)

    img1 = np.uint8(np.clip(img1 / np.mean(img1) * 100, 0, 255))
    img2 = np.uint8(np.clip(img2 / np.mean(img2) * 100, 0, 255))
    '''
    img1 = cv2.Canny(img1, 180, 300).astype(np.uint8)
    img2 = cv2.Canny(img2, 180, 300).astype(np.uint8)
    '''
    # sift = cv2.xfeatures2d.SIFT_create()
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1,des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)

    goodMatch = matches[:20]
    ptsA = np.float32([kp1[m.queryIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
    ptsB = np.float32([kp2[m.trainIdx].pt for m in goodMatch]).reshape(-1, 1, 2)

    ransacReprojThreshold = 4
    H, status = cv2.estimateAffinePartial2D(ptsA,ptsB,cv2.RANSAC,ransacReprojThreshold);

    imgOut = cv2.warpAffine(img2, H, (img1.shape[1],img1.shape[0]),flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    diff = np.abs(edge(img1)-edge(imgOut))

    return H, diff



def warpAffine(img, H):
    imgOut = cv2.warpAffine(img, H, (img.shape[1],img.shape[0]),flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    return imgOut


def readImgAs(path, resolution=None):
    img1 = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    if resolution==None:
        return img1
    else:
        return cv2.resize(img1,resolution)

def calAverageCalibration(imglist1,imglist2, show=False):
    Hs = []
    for i in trange(len(imglist1)):
        img_path1 = imglist1[i]
        img_path2 = imglist2[i]

        img1 = readImgAs(img_path1)
        img2 = cv2.flip(readImgAs(img_path2), 1)

        H, diff = calibrate_SIFT(img1, img2)
        if show:
            plt.subplot(221)
            plt.imshow(img1)
            plt.subplot(222)
            plt.imshow(img2)
            plt.subplot(223)
            plt.imshow(warpAffine(img2, H))
            plt.subplot(224)
            plt.imshow(diff, cmap='gray')

            plt.show()

        Hs.append(H)

    return np.mean(np.array(Hs), axis=0), Hs


if __name__=="__main__":
    RGB_PATH = "/Volumes/WD-SN550/calibration_v4/rgb"
    NIR_PATH = "/Volumes/WD-SN550/calibration_v4/nir"

    # rgb_list = []
    # nir_list = []
    # for _ in trange(CALC_NUM):
    #     random_dir_idx = int(np.random.random() * len(os.listdir(RGB_PATH)))
    #     random_rgb_dir = os.path.join(RGB_PATH, os.listdir(RGB_PATH)[random_dir_idx])
    #     random_nir_dir = os.path.join(NIR_PATH, os.listdir(NIR_PATH)[random_dir_idx])
    #     random_img_index = int(np.random.random() * len(os.listdir(random_rgb_dir)))
    #     random_rgb_path = os.path.join(random_rgb_dir, sorted(os.listdir(random_rgb_dir))[random_img_index])
    #     random_nir_path = os.path.join(random_nir_dir, sorted(os.listdir(random_nir_dir))[random_img_index])
    #     rgb_list.append(random_rgb_path)
    #     nir_list.append(random_nir_path)

    rgb_fnames = sorted(os.listdir(RGB_PATH))
    nir_fnames = sorted(os.listdir(NIR_PATH))
    rgb_list = [os.path.join(RGB_PATH, rgb_fnames[i]) for i in range(len(rgb_fnames))]
    nir_list = [os.path.join(NIR_PATH, nir_fnames[i]) for i in range(len(nir_fnames))]

    H, Hs = calAverageCalibration(rgb_list, nir_list, show=False)
    print(H)
    print(Hs)
    
    for i in trange(len(rgb_list)):
        img1 = readImgAs(rgb_list[i])
        img2 = cv2.flip(readImgAs(nir_list[i]), 1)
        img2_cali = warpAffine(img2,H)
        cv2.imwrite("./pic/pic%d_ref.png"%i, cv2.cvtColor(img1, cv2.COLOR_RGB2BGR))
        cv2.imwrite("./pic/pic%d_ori.png"%i, cv2.cvtColor(img2, cv2.COLOR_RGB2BGR))
        cv2.imwrite("./pic/pic%d_cali.png"%i, cv2.cvtColor(img2_cali, cv2.COLOR_RGB2BGR))
