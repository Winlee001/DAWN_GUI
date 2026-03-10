import importlib
import torch.utils.data
from data.base_dataset import BaseDataset
from .image_dataset import ImageDataset,IterImageDataset 

dataset_map = {
    'image': ['bsd68','set12','kodak24','set14','mcmaster','cbsd68','bsd500','wed4744','div2k','imagenet_val'],
}
dataset_map = {i:j for j in dataset_map.keys() for i in dataset_map[j]}


def create_dataset(dataset_name, split, opt):
    data_loader = CustomDatasetDataLoader(dataset_name, split, opt)
    dataset = data_loader.load_data()
    return dataset


class CustomDatasetDataLoader():
    def __init__(self, dataset_name, split, opt):
        self.opt = opt
       
        #TODO
        # if self.opt.dynamic_load == True:
        #     self.dataset = IterImageDataset(opt, split, dataset_name)
        #     self.dataloader = torch.utils.data.DataLoader(
        #     self.dataset,
        #     batch_size=opt.batch_size if split=='train' or split=='train_CT' else 1,
        #     num_workers=int(opt.load_thread), #传递线程数目
        #     drop_last=False)
        # else:
        self.dataset = ImageDataset(opt, split, dataset_name)
        self.dataloader = torch.utils.data.DataLoader(
        self.dataset,
        batch_size=opt.batch_size if split=='train' or split=='train_CT' else 1,
        shuffle=opt.shuffle and (split=='train' or split=='train_CT'),#and逻辑操作，使得训练时打乱，不验证时不打乱
        num_workers=int(opt.load_thread), #传递线程数目
        drop_last=False)
        self.imio = self.dataset.imio
        print("dataset [%s(%s)] created" % (dataset_name, split))
     

    def load_data(self):
        return self


    #TODO 可以通过更改max_dataset_size来改变len(dataset)
    def __len__(self):
        """Return the number of data in the dataset"""
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            if i >= self.opt.max_dataset_size:
                break
            #TODO 这句子让含噪数据集和干净数据集不一样多的时候，以小的为准，注意load_data_CT的顺序也要注意
            elif i >= len(self.dataset):
                break
            yield data

