import os
import pdb
# from mpi4py import MPI
import numpy as np
import h5py
import torch.utils.data as data
import torchvision.transforms as transforms


from ..lib import utils
from .images import ImagesFolder, AbstractImagesDataset, default_loader
from .features import FeaturesDataset

def split_name(data_split):
    if data_split in ['train', 'val']:
        return data_split
    elif data_split == 'test':
        return data_split
    else:
        assert False, 'data_split {} not exists'.format(data_split)


class CLEVRImages(AbstractImagesDataset):

    def __init__(self, data_split, opt, transform=None, loader=default_loader):
        super(CLEVRImages, self).__init__(data_split, opt, transform, loader)
        self.split_name = split_name(self.data_split)
        self.dir_split = os.path.join(self.dir_raw, self.split_name)
        self.dataset = ImagesFolder(self.dir_split, transform=self.transform, loader=self.loader)
        self.name_to_index = self._load_name_to_index()

    def _raw(self):
        raise NotImplementedError

    def _load_name_to_index(self):
        self.name_to_index = {name:index for index, name in enumerate(self.dataset.imgs)}
        return self.name_to_index

    def __getitem__(self, index):
        item = self.dataset[index]
        item['name'] = os.path.join(self.split_name, item['name'])
        return item

    def __len__(self):
        return len(self.dataset)


def default_transform(size):
    transform = transforms.Compose([
        transforms.Scale(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], # resnet imagnet
                             std=[0.229, 0.224, 0.225])
    ])
    return transform

def factory(data_split, opt, transform=None):
    if data_split == 'trainval':
        raise NotImplementedError
    elif data_split in ['train', 'val', 'test']:
        if opt['mode'] == 'img':
            if transform is None:
                transform = default_transform(opt['size'])
            return CLEVRImages(data_split, opt, transform)
        elif opt['mode'] in ['noatt', 'att']:
            return FeaturesDataset(data_split, opt)
        else:
            raise ValueError
    else:
        raise ValueError
