import os
import cv2
import numpy as np


def get_testimg(index=None, imgsize=None):
    imgpath = os.path.join(os.path.dirname(__file__), "TestImg/img/")
    img_num = len(os.listdir(imgpath))

    if index is None:
        index = img_num #int(np.random.random(1) * img_num)

    # print(os.listdir(imgpath)[index % img_num])
    img = cv2.imread(os.path.join(
        imgpath,
        sorted(os.listdir(imgpath))[index % img_num]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if imgsize is not None:
        img = cv2.resize(img, (imgsize[1], imgsize[0]))
    return img


def get_testimg_gray(index=None, imgsize=None):
    imgpath = os.path.join(os.path.dirname(__file__), "TestImg")
    img_num = len(os.listdir(imgpath))

    if index is None:
        index = int(np.random.random(1) * img_num)

    # print(os.listdir(imgpath)[index % img_num])
    img = cv2.imread(os.path.join(
        imgpath,
        os.listdir(imgpath)[index % img_num]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if imgsize is not None:
        img = cv2.resize(img, (imgsize[1], imgsize[0]))
    return img
