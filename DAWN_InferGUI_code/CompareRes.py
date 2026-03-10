import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from ImgProcessing.ImgUtils import compare_imgs_gray, compare_imgs_rgb
import skimage.metrics as Metrics
import tqdm



def read_tiff_image(path, type=".tiff", crop=None, size=None, BGR=True, color="rgb"):
    # 8 bit image
    # luckyprint(path, p=0.02)
    if (os.path.splitext(path)[-1] == type):
        if color=="gray" or color=="GRAY":
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
        else:
            img = cv2.imread(path, cv2.IMREAD_COLOR) 
        
        if img is None:
            if size is None:
                return np.zeros([100,100,3])
            else:
                return np.zeros(size)
            
        if crop:
            img = img[crop[0]:crop[1],crop[2]:crop[3],:]
        if size != None:
            img = cv2.resize(img, (size[1], size[0]))
        if BGR:
            img = img[:, :, (2, 1, 0)]
        
        return img.astype(np.float32)
    else:
        return np.array([[0]])

def compare_imgs_rgb(vid, gtr, normalized=False):
    vid = np.array(np.clip(vid, 0, 1)).astype(float)
    gtr = np.array(np.clip(gtr, 0, 1)).astype(float)
    if normalized:
        vid = vid / np.mean(vid) * np.mean(gtr)
        vid = np.clip(vid, 0, 1)
    
    frame_num = vid.shape[0]
    psnr = []
    ssim = []
    if vid.shape[1]==3:
        vid = np.transpose(vid, (0, 2, 3, 1))
    if gtr.shape[1]==3:
        gtr = np.transpose(gtr, (0, 2, 3, 1))
    
    for i in range(frame_num):
        psnr.append(Metrics.peak_signal_noise_ratio(gtr[i], vid[i]))
        ssim.append(Metrics.structural_similarity(gtr[i], vid[i], multichannel=True))

    return np.mean(psnr), np.mean(ssim)


def compare_imgs_gray(vid, gtr, normalized=False):
    vid = np.array(np.clip(vid, 0, 1))
    gtr = np.array(np.clip(gtr, 0, 1))
    if normalized:
        vid = vid / np.mean(vid) * np.mean(gtr)
        vid = np.clip(vid, 0, 1)
    
    frame_num = vid.shape[0]
    psnr = []
    ssim = []
    
    for i in range(frame_num):
        psnr.append(Metrics.peak_signal_noise_ratio(gtr[i], vid[i]))
        ssim.append(Metrics.structural_similarity(gtr[i], vid[i]))

    return np.mean(psnr), np.mean(ssim)


def load_video(path, fnum=None, ftype='.png'):
        img_paths = [os.path.join(path, img_path) for img_path in sorted(os.listdir(path))]
        if fnum is None or fnum > len(img_paths):
            fnum = len(img_paths)

        img_list = []
        for fidx in range(fnum):
            # L.info(nir_paths[fidx], rgb_paths[fidx])
            img = read_tiff_image(img_paths[fidx], type=ftype, BGR=False) / 255.0
            img_list.append(img)
            
        return np.array(img_list)
            

if __name__=="__main__":
    dirs = ["/Users/cjarry/Desktop/files.nosync/proj-files-level2/dual-channel-low-light/test_res/PID448133_kc_40_40_test"]
    for dir_name in dirs:
        psnr_l = []
        ssim_l = []
        for folder in tqdm.tqdm(sorted(os.listdir(dir_name))):
            if os.path.isdir(os.path.join(dir_name, folder)):
                vid = load_video(os.path.join(dir_name, folder, "rgb_noisy"))
                truth = load_video(os.path.join(dir_name, folder, "rgb_label"))
                psnr, ssim = compare_imgs_rgb(vid, truth)
                psnr_l.append(psnr)
                ssim_l.append(ssim)
        print("Dir:   {:s}\n rgb_noisy vs. rgb_label \nPSNR: {:.3f}    SSIM: {:.3f}".format(
            dir_name,
            np.mean(psnr_l),
            np.mean(ssim_l)
        ))
        
        psnr_l = []
        ssim_l = []
        for folder in tqdm.tqdm(sorted(os.listdir(dir_name))):
            if os.path.isdir(os.path.join(dir_name, folder)):
                vid = load_video(os.path.join(dir_name, folder, "rgb_pred"))
                truth = load_video(os.path.join(dir_name, folder, "rgb_label"))
                psnr, ssim = compare_imgs_rgb(vid, truth)
                psnr_l.append(psnr)
                ssim_l.append(ssim)
        print("Dir:   {:s}\n rgb_noisy vs. rgb_label \nPSNR: {:.3f}    SSIM: {:.3f}".format(
            dir_name,
            np.mean(psnr_l),
            np.mean(ssim_l)
        ))