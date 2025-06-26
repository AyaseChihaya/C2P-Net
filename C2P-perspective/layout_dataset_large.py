import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import random
# from nyu_transform import *
import glob
import os
# from demo_transform import *
# from lsun_transform import *
import matterport_transform
import h5py
import scipy.io as io


class ImageDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, transform_matterport=None):

        self.img_name = sorted(glob.glob(os.path.join('/home/ubuntu/panorama/dataset/implementation/dataset_new/train/img/', '*_image.jpg')))
        self.depth_name = sorted(glob.glob(os.path.join('/home/ubuntu/panorama/dataset/implementation/dataset_new/train/depth/', '*_depth.png')))

        # self.norm = sorted(glob.glob(os.path.join('/home/ubuntu/panorama/dataset/implementation/dataset_new/train/abcd/', '*_norm.mat')))
        self.a = sorted(glob.glob(os.path.join('/home/ubuntu/panorama/dataset/implementation/dataset_new/train/a/', '*_na.png')))
        self.b = sorted(glob.glob(os.path.join('/home/ubuntu/panorama/dataset/implementation/dataset_new/train/b/', '*_nb.png')))
        self.c = sorted(glob.glob(os.path.join('/home/ubuntu/panorama/dataset/implementation/dataset_new/train/c/', '*_nc.png')))
        self.d = sorted(glob.glob(os.path.join('/home/ubuntu/panorama/dataset/implementation/dataset_new/train/d/', '*_nd.png')))

        # self.seg_name = self.depth_name
        self.seg_name = sorted(glob.glob(os.path.join('/home/ubuntu/panorama/dataset/implementation/dataset_new/train/seg/', '*_seg.png')))
        # self.frame = pd.read_csv(csv_file, header=None)
        self.transform_matterport = transform_matterport

    def __getitem__(self, idx):

        # image_name = self.frame.ix[idx, 0]
        # depth_name = self.frame.ix[idx, 1]
        #
        # image = Image.open(image_name)
        # depth = Image.open(depth_name)
        #
        # sample = {'image': image, 'depth': depth}
        # if self.transform_nyu:
        #     sample = self.transform_nyu(sample)
        # nyu_image, nyu_depth= sample['image'], sample['depth']

        # sample_idx = random.randint(0,len(self.img_name)-1)
        # print(idx)
        sample_idx = idx
        image = Image.open(self.img_name[sample_idx])
        depth = Image.open(self.depth_name[sample_idx])
        seg = Image.open(self.seg_name[sample_idx])
        A = Image.open(self.a[sample_idx])
        B = Image.open(self.b[sample_idx])
        C = Image.open(self.c[sample_idx])
        D = Image.open(self.d[sample_idx])

        # seg = depth
        # norm_data = h5py.File(self.norm[sample_idx],'r')
        # A = Image.fromarray(norm_data['na'][:])
        # B = Image.fromarray(norm_data['nb'][:])
        # C = Image.fromarray(norm_data['nc'][:])
        # D = Image.fromarray(norm_data['nd'][:])

        # sample = {'image': image, 'depth': depth, 'seg': seg}
        sample = {'image': image, 'depth': depth, 'seg': seg, 'A': A, 'B': B, 'C': C, 'D': D}
        if self.transform_matterport:
            sample = self.transform_matterport(sample)

        return sample

    def __len__(self):
        return len(self.img_name)



class TestDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, transform_matterport=None):


        self.img_name = sorted(glob.glob(os.path.join('/home/ubuntu/work/geolayout/dataset/test_3d_img/', '*.jpg')))

        self.transform_matterport = transform_matterport

    def __getitem__(self, idx):

        sample_idx = idx
        image = Image.open(self.img_name[sample_idx])

        sample = {'image': image}
        if self.transform_matterport:
            sample = self.transform_matterport(sample)

        return sample

    def __len__(self):
        return len(self.img_name)



def getLayout(batch_size=8):
    __imagenet_pca = {
        'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
        'eigvec': torch.Tensor([
            [-0.5675,  0.7192,  0.4009],
            [-0.5808, -0.0045, -0.8140],
            [-0.5836, -0.6948,  0.4203],
        ])
    }
    __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}

    transformed_training = ImageDataset(transform_matterport=transforms.Compose([
                                            matterport_transform.Scale(224),
                                            # matterport_transform.RandomHorizontalFlip(),
                                            # matterport_transform.RandomRotate(5),
                                            # matterport_transform.CenterCrop([300, 240], [300, 240]),
                                            # matterport_transform.CenterCrop([224, 224], [256, 256]),
                                            # matterport_transform.CenterCrop([256, 256], [256, 256]),
                                            # matterport_transform.CenterCrop([304, 228], [304, 228]),
                                            matterport_transform.ToTensor(),
                                            matterport_transform.Lighting(0.1, __imagenet_pca[
                                                'eigval'], __imagenet_pca['eigvec']),
                                            matterport_transform.ColorJitter(
                                                brightness=0.4,
                                                contrast=0.4,
                                                saturation=0.4,
                                            ),
                                            matterport_transform.Normalize(__imagenet_stats['mean'],
                                                      __imagenet_stats['std'])
                                        ])
                                        )
    dataloader_training = DataLoader(transformed_training, batch_size,
                                     shuffle=True, num_workers=4, pin_memory=False)

    return dataloader_training

def getTestingData(batch_size=64):

    __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}
    # scale = random.uniform(1, 1.5)
    transformed_testing = TestDataset(transform_matterport=transforms.Compose([
                                           # matterport_transform.Scale(320),
                                           matterport_transform.CenterCrop([256, 256], [128, 128],is_test=True),
                                           matterport_transform.ToTensor(is_test=True),
                                           matterport_transform.Normalize(__imagenet_stats['mean'],
                                                     __imagenet_stats['std'])
                                       ]))

    dataloader_testing = DataLoader(transformed_testing, batch_size,
                                    shuffle=False, num_workers=0, pin_memory=False)

    return dataloader_testing
