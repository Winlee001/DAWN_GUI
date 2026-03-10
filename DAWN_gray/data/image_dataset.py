import random
import numpy as np
import os
from os.path import join
from .dn_dataset import DnDataset
from . import imlib
from torch.utils.data import IterableDataset,get_worker_info
rootlist = {

    #LOCAL DATA
    'SMF0723_acc_15040001':[r"C:\Users\Admin\Desktop\EMCCD_test\SMF_0723_NAdata_SA3f\acc\500mM\15%\s1504_0001"],
    'SMF0723_don_15040001':[r"C:\Users\Admin\Desktop\EMCCD_test\SMF_0723_NAdata_SA3f\don\500mM\15%\s1504_0001"],
    'SMF0723_acc_1504val':[r"C:\Users\Admin\Desktop\EMCCD_test\SMF_0723_NAdata_SA3f_Val\acc\500mW\15%"],
    'SMF0723_don_1504val':[r"C:\Users\Admin\Desktop\EMCCD_test\SMF_0723_NAdata_SA3f_Val\don\500mW\15%"],

}


def _normalize_dataset_pair(dataset_name):
    if isinstance(dataset_name, str):
        names = [dataset_name]
    else:
        names = list(dataset_name)
    if len(names) == 1:
        return names[0], names[0]
    if len(names) >= 2:
        return names[0], names[1]
    raise ValueError("dataset_name should contain at least one item")


def _resolve_roots(name_token):
    if name_token in rootlist:
        return rootlist[name_token]
    if os.path.isdir(name_token):
        return [name_token]
    raise KeyError(f"Dataset key or folder path not found: {name_token}")

class ImageDataset(DnDataset): 
    def __init__(self, opt, split, dataset_name):
        super(ImageDataset, self).__init__(opt, split, dataset_name)
        if self.root == '':
            self.root = []
            self.rate = opt.data_rate
            self.repeat = opt.repeat
            #TODO
            if self.Real_dataset_mode==False:
                pass
            # above contents can be seen in bbncy 
            #TODO dual_fusion,temporalily use clr->Nos1 and Nos->Nos2
            # elif self.bsn_ver == 'dbsnl_fuse' or self.bsn_ver == 'dbsn_fuse':
            else:
                nos1_name, nos2_name = _normalize_dataset_pair(dataset_name)
                self.Nos1_root = _resolve_roots(nos1_name)
                self.Nos2_root = _resolve_roots(nos2_name)
                assert len(self.Nos1_root) == len(self.Nos2_root), "Nos1_root and Nos2_root must have same length"
                self.Nos1_names = []
                self.Nos2_names = []
                self.Nos1_images = []
                self.Nos2_images = []
                for i in range(len(self.Nos1_root)):
                    #cal rel path
                    tmp_names = imlib.scan(self.Nos1_root[i])
                    if opt.stratified_sample and self.rate < 1.0 and self.split == 'train_CT':
                        intermediate_folder = set(os.path.dirname(name) for name in tmp_names)
                        tmp_names = []
                        file_names = []
                        for folder in intermediate_folder:
                            file_names = imlib.scan(folder)
                            sample_index = np.random.choice(len(file_names),size = int(len(file_names) * self.rate), replace=False)
                            tmp_names.extend([file_names[i] for i in sample_index])
                    else:
                        sample_index = np.random.choice(len(tmp_names), size = int(len(tmp_names) * self.rate),replace=False)
                        tmp_names = np.array(tmp_names)[sample_index]
                    tmp_paths = [os.path.relpath(x, self.Nos1_root[i]) for x in tmp_names]
                    self.Nos1_names.extend(tmp_names)
                    self.Nos1_images.extend(tmp_names)
                    # make Nos1 and Nos2 have the same rel path to make sure the align image be read
                    tmp_names = [os.path.join(self.Nos2_root[i], x) for x in tmp_paths]
                    self.Nos2_names.extend(tmp_names)
                    self.Nos2_images.extend(tmp_names)
         
                self.clr_names = self.Nos1_names*self.repeat if self.split == 'train_CT' else self.Nos1_names
                self.Nos_names = self.Nos2_names*self.repeat if self.split == 'train_CT' else self.Nos2_names
                self.clr_images = self.Nos1_images*self.repeat if self.split == 'train_CT' else self.Nos1_images
                self.Nos_images = self.Nos2_images*self.repeat if self.split == 'train_CT' else self.Nos2_images
                self.imio = imlib.imlib(self.mode, lib=opt.imlib)
                self.load_data_CT(2)
                self.load_data_CT(1)

    def float(self, img):
        if self.Real_dataset_mode==False:
            return img.astype(np.float32) / 255.
        else:
            return img.astype(np.float32) / 65535.



class IterImageDataset(IterableDataset):
    def __init__(self, opt, split, dataset_name):
        self.split = split
        self.mode = opt.mode  # RGB, YCrCb or L L_16
        self.noise_type = opt.noise_type
        self.preload = opt.preload
        self.batch_size = opt.batch_size
        self.patch_size = opt.patch_size
        self.rate = opt.data_rate
        assert self.rate <= 1.0 and self.rate>0.0, 'data rate should be in (0, 1]'
        #TODO
        self.Real_dataset_mode= opt.Real_dataset_mode
        self.bsn_ver = opt.bsn_ver
        self.flip = not opt.no_flip if split == 'train_CT' else False
        self.repeat = opt.repeat
        #TODO
        if self.Real_dataset_mode==False:
            pass
        # above contents can be seen in bbncy 
        #TODO dual_fusion,temporalily use clr->Nos1 and Nos->Nos2
        else:
            nos1_name, nos2_name = _normalize_dataset_pair(dataset_name)
            self.Nos1_root = _resolve_roots(nos1_name)
            self.Nos2_root = _resolve_roots(nos2_name)
            assert len(self.Nos1_root) == len(self.Nos2_root), "Nos1_root and Nos2_root must have same length"
            self.Nos1_names = []
            self.Nos2_names = []
            self.Nos1_images = []
            self.Nos2_images = []
            for i in range(len(self.Nos1_root)):
                #cal rel path
                tmp_names = imlib.scan(self.Nos1_root[i])
                if opt.stratified_sample and self.rate < 1.0 and self.split == 'train_CT':
                    intermediate_folder = set(os.path.dirname(name) for name in tmp_names)
                    tmp_names = []
                    file_names = []
                    for folder in intermediate_folder:
                        file_names = imlib.scan(folder)
                        sample_index = np.random.choice(len(file_names),size = int(len(file_names) * self.rate), replace=False)
                        tmp_names.extend([file_names[i] for i in sample_index])
                else:
                    sample_index = np.random.choice(len(tmp_names), size = int(len(tmp_names) * self.rate),replace=False)
                    tmp_names = np.array(tmp_names)[sample_index]
                tmp_paths = [os.path.relpath(x, self.Nos1_root[i]) for x in tmp_names]
                self.Nos1_names.extend(tmp_names)
                self.Nos1_images.extend(tmp_names)
                # make Nos1 and Nos2 have the same rel path to make sure the align image be read
                tmp_names = [os.path.join(self.Nos2_root[i], x) for x in tmp_paths]
                self.Nos2_names.extend(tmp_names)
                self.Nos2_images.extend(tmp_names)
            # 图片读取
            self.clr_names = self.Nos1_names*self.repeat if self.split == 'train_CT' else self.Nos1_names
            self.Nos_names = self.Nos2_names*self.repeat if self.split == 'train_CT' else self.Nos2_names
            self.clr_images = self.Nos1_images*self.repeat if self.split == 'train_CT' else self.Nos1_images
            self.Nos_images = self.Nos2_images*self.repeat if self.split == 'train_CT' else self.Nos2_images
            self.imio = imlib.imlib(self.mode, lib=opt.imlib)

    def __iter__(self):
        worker_info = get_worker_info()
       # 构造索引并打乱
        indices = list(range(len(self.clr_images)))
        if self.split == 'train_CT':
            random.seed(self.epoch)
            random.shuffle(indices)
        if worker_info is None:
            start, end = 0, len(indices)
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            chunk_size = len(indices) // num_workers
            start = worker_id * chunk_size
            end = start + chunk_size if worker_id != num_workers - 1 else len(indices)
            print(f"Worker {worker_id} 读取数据范围：{start} - {end}")

        for idx in indices[start:end]:
            clr_path = self.clr_images[idx]
            nos_path = self.Nos_images[idx]

            clr_image = self.imio.read(clr_path)
            nos_image = self.imio.read(nos_path)
            if self.split == 'train_CT':
                clr_image, nos_image = self._crop_d(clr_image, nos_image)
            if self.flip:
                # print('我他妈来了')
                clr_image, nos_image = self._augment_d(clr_image, nos_image)
            clr_image = self.float(clr_image)
            nos_image = self.float(nos_image)

            yield {'clean': clr_image, 'noisy': nos_image, 'clean_name': clr_path, 'noisy_name': nos_path}

                # if worker_info is None:  # 单线程模式
        #     start, end = 0, len(self.Nos_names)
        # else:  # 多 worker 模式
        #     num_workers = worker_info.num_workers
        #     worker_id = worker_info.id
        #     chunk_size = len(self.Nos_names) // num_workers
        #     start = worker_id * chunk_size
        #     end = start + chunk_size if worker_id != num_workers - 1 else len(self.Nos_names) #最后一个worker处理剩下的所有数据
        #     print(f"Worker {worker_id} 读取数据范围：{start} - {end}")
        # """逐个读取并返回数据"""
        # for clr_path, nos_path in zip(self.clr_names[start:end], self.Nos_names[start:end]):
        #     clr_image = self.imio.read(clr_path)
        #     nos_image = self.imio.read(nos_path)
        #     # print(clr_image.shape,clr_image.max(),clr_image.min())
        #     # print(nos_image.shape,nos_image.max(),nos_image.min())
        #     clr_image,nos_image = self._crop_d(clr_image,nos_image)
        #     # NOTE: controllable by opt.no_flip
        #     # clr_image = self._augment(clr_image) if self.flip else clr_image
        #     # nos_image = self._augment(nos_image) if self.flip else nos_image
        #     if self.flip:
        #         clr_image, nos_image = self._augment_d(clr_image, nos_image)
        #     clr_image = self.float(clr_image)
        #     nos_image = self.float(nos_image)

        #     yield {'clean': clr_image, 'noisy': nos_image, 'clean_name': clr_path, 'noisy_name': nos_path}

    def _crop_d(self, imagex,imagey):
        ihx, iwx = imagex.shape[-2:]
        ihy, iwy = imagey.shape[-2:]
        assert ihx == ihy and iwx == iwy
        ix = random.randrange(0, iwx - self.patch_size + 1)
        iy = random.randrange(0, ihx - self.patch_size + 1)
        return imagex[..., iy:iy+self.patch_size, ix:ix+self.patch_size], imagey[..., iy:iy+self.patch_size, ix:ix+self.patch_size]

    def random_transform(self,input):
        p_trans = random.randrange(8)  # (64, 128, 128)
        if p_trans == 0:  # no transformation
            input = input
        elif p_trans == 1:  # left rotate 90
            input = np.rot90(input, k=1, axes=(1, 2))
        elif p_trans == 2:  # left rotate 180
            input = np.rot90(input, k=2, axes=(1, 2))
        elif p_trans == 3:  # left rotate 270
            input = np.rot90(input, k=3, axes=(1, 2))
        elif p_trans == 4:  # horizontal flip
            input = input[:, :, ::-1]
        elif p_trans == 5:  # horizontal flip & left rotate 90
            input = input[:, :, ::-1]
            input = np.rot90(input, k=1, axes=(1, 2))
        elif p_trans == 6:  # horizontal flip & left rotate 180
            input = input[:, :, ::-1]
            input = np.rot90(input, k=2, axes=(1, 2))
        elif p_trans == 7:  # horizontal flip & left rotate 270
            input = input[:, :, ::-1]
            input = np.rot90(input, k=3, axes=(1, 2))
        return input

    def _augment_d(self, imgx,imgy):
        # if random.random() < 0.5:   imgx,imgy = imgx[:, ::-1, :], imgy[:, ::-1, :]
        # if random.random() < 0.5:   imgx,imgy = imgx[:, :, ::-1], imgy[:, :, ::-1]
        # if random.random() < 0.5:   imgx,imgy = imgx.transpose(0, 2, 1), imgy.transpose(0, 2, 1)
        # return np.ascontiguousarray(imgx), np.ascontiguousarray(imgy) # transpose may destroy contigousness
        return self.random_transform(imgx), self.random_transform(imgy)

    
    # def _crop(self, image):
    #     ih, iw = image.shape[-2:] # HW for gray and CHW for RGB
    #     # print(image.shape,ih,iw)
    #     #ih, iw = image.shape[0:2]  # HWC for RGB
    #     ix = random.randrange(0, iw - self.patch_size + 1)
    #     iy = random.randrange(0, ih - self.patch_size + 1)
    #     return image[..., iy:iy+self.patch_size, ix:ix+self.patch_size]

    # def _augment(self, img):
    #     if random.random() < 0.5:   img = img[:, ::-1, :]
    #     if random.random() < 0.5:   img = img[:, :, ::-1]
    #     if random.random() < 0.5:   img = img.transpose(0, 2, 1) # CHW
    #     return np.ascontiguousarray(img)
    
    def length(self):
        return len(self.Nos_names)

    #TODO 这是非常的一步操作，在Pytorch的Dataloader中会默认将numpy转化为Tensor数据，但是归一化操作需要靠自己定义
    def float(self, img):
        return img.astype(np.float32) / (255. if not self.Real_dataset_mode else 65535.)
    
    

if __name__ == '__main__':
    pass