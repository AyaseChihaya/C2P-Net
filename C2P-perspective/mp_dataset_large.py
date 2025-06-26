import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
# import PIL
import random
import glob
import os
import mp_transform_large as joint_transform
import h5py
import scipy.io as io
#import matplotlib.pyplot as plt


class ImageDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, transform_matterport=None):

        self.lsun_img_name = sorted(glob.glob(os.path.join('/home/ubuntu/work/geolayout/dataset/Matterport3D_Layout_v1/training/image/', '*.jpg')))
        self.lsun_seg_name = sorted(glob.glob(os.path.join('/home/ubuntu/work/geolayout/dataset/Matterport3D_Layout_v1/training/layout_seg/', '*_seg.png')))
        self.lsun_depth_name = sorted(glob.glob(os.path.join('/home/ubuntu/work/geolayout/dataset/Matterport3D_Layout_v1/training/layout_depth/', '*_layout.png')))
        self.lsun_raw_depth_name = sorted(glob.glob(os.path.join('/home/ubuntu/work/geolayout/dataset/Matterport3D_Layout_v1/training/depth/', '*.png')))
        # self.lsun_edge_name = sorted(glob.glob(os.path.join('/home/ubuntu/work/geolayout/dataset/matterport3d_edge/', '*.png')))


        self.transform_matterport = transform_matterport

    def __getitem__(self, idx):

        sample_idx = idx

        lsun_image = Image.open(self.lsun_img_name[sample_idx])
        lsun_seg = Image.open(self.lsun_seg_name[sample_idx])
        lsun_depth = Image.open(self.lsun_depth_name[sample_idx])
        lsun_raw_depth = Image.open(self.lsun_raw_depth_name[sample_idx])


        sample = {'lsun_image': lsun_image, 'lsun_seg': lsun_seg, 'lsun_depth': lsun_depth, 'lsun_raw_depth': lsun_raw_depth}
        if self.transform_matterport:
            sample = self.transform_matterport(sample)

        return sample

    def __len__(self):
        return len(self.lsun_img_name)


class TestDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, transform_matterport=None):

        # self.img_name = sorted(glob.glob(os.path.join('/home/ubuntu/panorama/dataset/implementation/pytorch_build/example/', '*.jpg')))
        self.img_name = sorted(glob.glob(os.path.join('/home/ubuntu/caffe/chihaya/layout/cvpr2020/val_img/', '*.jpg')))
        # self.img_name = sorted(glob.glob(os.path.join('/home/ubuntu/panorama/dataset/implementation/dataset_new/test/img/', '*_image.jpg')))
        self.depth_name = sorted(glob.glob(os.path.join('/home/ubuntu/panorama/dataset/implementation/dataset_new/test/depth/', '*_depth.png')))

        # self.norm = sorted(glob.glob(os.path.join('/home/ubuntu/panorama/dataset/implementation/dataset_new/train/abcd/', '*_norm.mat')))
        self.a = sorted(glob.glob(os.path.join('/home/ubuntu/panorama/dataset/implementation/dataset_new/test/a/', '*_na.png')))
        self.b = sorted(glob.glob(os.path.join('/home/ubuntu/panorama/dataset/implementation/dataset_new/test/b/', '*_nb.png')))
        self.c = sorted(glob.glob(os.path.join('/home/ubuntu/panorama/dataset/implementation/dataset_new/test/c/', '*_nc.png')))
        self.d = sorted(glob.glob(os.path.join('/home/ubuntu/panorama/dataset/implementation/dataset_new/test/d/', '*_nd.png')))

        # self.seg_name = self.depth_name
        self.seg_name = sorted(glob.glob(os.path.join('/home/ubuntu/panorama/dataset/implementation/dataset_new/train/seg/', '*_seg.png')))
        # self.frame = pd.read_csv(csv_file, header=None)
        self.transform_matterport = transform_matterport

    def __getitem__(self, idx):

        # io.savemat('/home/ubuntu/panorama/dataset/implementation/dataset_new/test/img_name.mat',mdict={'img_name':img_name})

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

        # sample = {'image': image, 'depth': depth, 'seg': seg}
        sample = {'image': image, 'depth': depth, 'seg': seg, 'A': A, 'B': B, 'C': C, 'D': D}
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
                                            joint_transform.Scale(270),
                                            # joint_transform.Scale(340),
                                            joint_transform.RandomHorizontalFlip(),
                                            joint_transform.RandomRotate(5),
                                            joint_transform.CenterCrop([256, 256], [128, 128]),
                                            # joint_transform.CenterCrop([320, 320], [160, 160]),
                                            joint_transform.ToTensor(),
                                            joint_transform.Lighting(0.1, __imagenet_pca[
                                                'eigval'], __imagenet_pca['eigvec']),
                                            joint_transform.ColorJitter(
                                                brightness=0.4,
                                                contrast=0.4,
                                                saturation=0.4,
                                            ),
                                            joint_transform.Normalize(__imagenet_stats['mean'],
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
                                           joint_transform.Scale(240),
                                           joint_transform.CenterCrop([304, 228], [304, 228],is_test=True),
                                           joint_transform.ToTensor(is_test=True),
                                           joint_transform.Normalize(__imagenet_stats['mean'],
                                                     __imagenet_stats['std'])
                                       ]))

    dataloader_testing = DataLoader(transformed_testing, batch_size,
                                    shuffle=False, num_workers=0, pin_memory=False)

    return dataloader_testing
