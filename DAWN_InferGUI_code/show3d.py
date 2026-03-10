import sys
import os

import numpy as np
import argparse
from os.path import join as opj
from os.path import dirname as opd
from os.path import basename as opb
from os.path import splitext as ops
import tifffile
from ImgProcessing.ImgUtils import show_bright_images

def show_3d_img(img, norm=True, gain=1, filename_prefix=""):
    if norm:
        img = img / np.max(img)
    img = img * gain
    sh = img.shape
    slice_num = img.shape[0]
    slices = np.zeros([12, sh[1], sh[2], sh[3]])
    slice_step = max(1, slice_num//12)
    slices[:min(sh[0], 12)] = img[:slice_step*12:slice_step,:,:]
    show_bright_images(slices.reshape(3,4,img.shape[1],img.shape[2]), norm=False, channel_first=False, filename_prefix=filename_prefix)    

    
if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description='single task for datacompress')
    parser.add_argument('-p', type=str, default="cyx_exp/experiments_outputs/ex0207_lqm_ffmpeg/outputs/sota_ffmpeg_2022_0208_202709/task_00001/decompressed/d_16_31-h_256_511-w_0_255.tif", help='yaml file path')
    # parser.add_argument('p', type=str, help='yaml file path')
    parser.add_argument('-g', type=float, default=1, help='image gain')
    parser.add_argument('-n', action="store_true", help='normalize to full scale')
    args = parser.parse_args()
    print("Reading data...")
    img = tifffile.imread(args.p)
    print("Data type: {:s}, Data shape: {:s}, Data range: {:f}-{:f}".format(
        str(img.dtype),
        str(img.shape),
        np.min(img),
        np.max(img)
    ))
    
    data_range = {"uint16":65535.0, "uint8":255.0, "float64":65535.0, "float32":2}[str(img.dtype)]
    show_3d_img(img/data_range, gain=args.g, norm=args.n)