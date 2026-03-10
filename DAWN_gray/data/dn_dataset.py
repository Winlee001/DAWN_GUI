import os
from functools import partial
import cv2
import random
import numpy as np
from . import imlib
from os.path import join
from data.base_dataset import BaseDataset
from sympy import *
from scipy.linalg import orth



class DnDataset(BaseDataset):
    def __init__(self, opt, split, dataset_name):
        super(DnDataset, self).__init__(opt, split, dataset_name)
        self.split = split
        self.mode = opt.mode  # RGB, YCrCb or L L_16
        self.noise_type = opt.noise_type
        self.preload = opt.preload
        self.batch_size = opt.batch_size
        self.patch_size = opt.patch_size
        #TODO
        self.Real_dataset_mode= opt.Real_dataset_mode
        self.bsn_ver = opt.bsn_ver
        self.flip = not opt.no_flip
        self.getimage = self.getimage_read
        self.multi_imreader = opt.multi_imreader


        # if self.split == 'train':
        #     self._getitem = self._getitem_train
        #     self._add_noise = getattr(self, '_add_noise_%s' % opt.noise_type)
        if self.split == 'train_CT':
            self._getitem = self._getitem_train_CT
        elif self.split == 'val_CT':
            self._getitem = self._getitem_test_CT
        else:
            raise ValueError('split should be train_CT or val_CT,and this server does not support others')

    # def load_data(self):
    #     self.len_data = len(self.names)
    #     if self.preload:
    #         if self.multi_imreader:
    #             read_images(self)
    #         else:
    #             self.images = [self.imio.read(p) for p in self.images]
    #         self.getimage = self.getimage_preload
    #         #print(self.images)

    #TODO
    def load_data_CT(self,Method:int):
        if Method==2:
            self.len_data = len(self.clr_names)
        elif Method==1:
            self.len_data = len(self.Nos_names)
        if self.preload:
            if self.multi_imreader:
                read_images(self,Method)
        else:
            self.images = [self.imio.read(p) for p in self.images]
        self.getimage = self.getimage_preload
#TODO
    def getimage_preload(self, index,Method=0):
        if Method == 0:
            return self.images[index], self.names[index]
        elif Method == 1:
            return self.Nos_images[index], self.Nos_names[index]
        elif Method == 2:
            return self.clr_images[index], self.clr_names[index]
    def getimage_read(self, index,Method=0):
        if Method==1:
            return self.imio.read(self.Nos_images[index]), self.Nos_names[index]
        elif Method==2:
            return self.imio.read(self.clr_images[index]), self.clr_names[index]
        else:
            return self.imio.read(self.images[index]), self.names[index]
    def _getitem_train_CT(self,index):
        #print(index,self.len_data)

        image_C, f_nameC = self.getimage(index,2)
        image_N,f_nameN = self.getimage(index,1)
        #print(image_C.shape)
        #TODO 
        image_C, image_N = self._crop_d(image_C, image_N, self.patch_size)
        if self.flip:
            image_C, image_N = self._augment_d(image_C, image_N)
   
        # image_C = self._crop(image_C)
        # image_N = self._crop(image_N)
        # # NOTE: controllable by opt.no_flip
        # image_C = self._augment(image_C) if self.flip else image_C
        # image_N = self._augment(image_N) if self.flip else image_N
        image_C = self.float(image_C)
        image_N = self.float(image_N)
        # return {'clean':image_C,
        #           'noisy':image_N,
        #         'f_name':f_name}
        return {'clean':image_C,'noisy':image_N,'clean_name':f_nameC,'noisy_name':f_nameN}
    

    # def _getitem_train(self, index):
    #     image, f_name = self.getimage(index,0)
    #     #print(index)
    #     '''
    #         getimage-->>getimage_read
    #       def getimage_read(self, index):
    #     return self.imio.read(self.images[index]), self.names[index]
    #     '''
    #     image = self._crop(image)
    #     # NOTE: controllable by opt.no_flip
    #     image = self._augment(image) if self.flip else image
    #     image = self.float(image)
    #     #print(image.shape,image.dtype)
    #     return {'clean': image,
    #             'noisy': self._add_noise(image),
    #             'fname': f_name} #从这可以看出是合成噪声图，所以是配对的

    def _getitem_test_CT(self, index):
        image_C, f_nameC = self.getimage(index, 2)
        image_N, f_nameN = self.getimage(index, 1)

        if image_C is None:
            print(f"Error: getimage returned None for index {index} and type 2")
        if image_N is None:
            print(f"Error: getimage returned None for index {index} and type 1")
    
        # NOTE: controllable by opt.no_flip
        image_C = self.float(image_C)
        image_N = self.float(image_N)
        return {'clean': image_C, 'noisy': image_N,'clean_name':f_nameC,'noisy_name':f_nameN}



    def __getitem__(self, index):
        return self._getitem(index) #内置方法，当对象被调用时，如object[]被调用

    def __len__(self):
        return self.len_data

    def _crop(self, image):
        ih, iw = image.shape[-2:] # HW for gray and CHW for RGB
        # print(image.shape,ih,iw)
        #ih, iw = image.shape[0:2]  # HWC for RGB
        ix = random.randrange(0, iw - self.patch_size + 1)
        iy = random.randrange(0, ih - self.patch_size + 1)
        return image[..., iy:iy+self.patch_size, ix:ix+self.patch_size]

    def _augment(self, img):
        if random.random() < 0.5:   img = img[:, ::-1, :]
        if random.random() < 0.5:   img = img[:, :, ::-1]
        if random.random() < 0.5:   img = img.transpose(0, 2, 1) # CHW
        return np.ascontiguousarray(img)
    def _crop_d(self, imagex,imagey):
        ihx, iwx = imagex.shape[-2:]
        ihy, iwy = imagey.shape[-2:]
        assert ihx == ihy and iwx == iwy
        ix = random.randrange(0, iwx - self.patch_size + 1)
        iy = random.randrange(0, ihx - self.patch_size + 1)
        return imagex[..., iy:iy+self.patch_size, ix:ix+self.patch_size], imagey[..., iy:iy+self.patch_size, ix:ix+self.patch_size]

    def _augment_d(self, imgx,imgy):
        if random.random() < 0.5:   imgx,imgy = imgx[:, ::-1, :], imgy[:, ::-1, :]
        if random.random() < 0.5:   imgx,imgy = imgx[:, :, ::-1], imgy[:, :, ::-1]
        if random.random() < 0.5:   imgx,imgy = imgx.transpose(0, 2, 1), imgy.transpose(0, 2, 1)
        return np.ascontiguousarray(imgx), np.ascontiguousarray(imgy) # transpose may destroy contigousness


def iter_obj(num, objs):
    for i in range(num):
        yield (i, objs)
#TODO
def imreader(arg):
        i, obj = arg
        obj.images[i] = obj.imio.read(obj.images[i])
def imreader_Nos(arg):
    i, obj = arg
    obj.Nos_images[i] = obj.imio.read(obj.Nos_images[i])
def imreader_clr(arg):
    i, obj = arg
    obj.clr_images[i] = obj.imio.read(obj.clr_images[i])
# def imreader(arg):
#     i, obj = arg
#     obj.images[i] = obj.imio.read(obj.images[i])
    # for _ in range(3):
    #     try:
    #         obj.images[i] = obj.imio.read(obj.images[i])
    #         failed = False
    #         break
    #     except:
    #         failed = True
    # if failed: print('%s fails!' % obj.names[i])

def read_images(obj,Method=0):
    # may use `from multiprocessing import Pool` instead, but less efficient and
    # NOTE: `multiprocessing.Pool` will duplicate given object for each process.
    from multiprocessing.dummy import Pool, freeze_support
    from tqdm import tqdm
    print('Starting to load images via multiple imreaders')
    pool = Pool() # use all threads by default
    # #TODO 使用 partial 来预设 Method 参数。因为pool.imap需要传递一个函数，并非函数调用后的返回值，即不能写成imreader(Method)
    # imreader_with_method = partial(imreader, Method)
    if Method == 0:
        for _ in tqdm(pool.imap(imreader, iter_obj(obj.len_data, obj)),
                  total=obj.len_data):
            pass
    elif Method == 1:
        for _ in tqdm(pool.imap(imreader_Nos, iter_obj(obj.len_data, obj)),
                      total=obj.len_data):
            pass
    elif Method == 2:
        for _ in tqdm(pool.imap(imreader_clr, iter_obj(obj.len_data, obj)),
                      total=obj.len_data):
            pass
    pool.close()
    pool.join()

if __name__ == '__main__':
    pass
