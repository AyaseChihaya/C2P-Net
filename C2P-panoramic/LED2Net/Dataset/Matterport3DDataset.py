import os
import random
import sys
import cv2
import json
import numpy as np
import torchvision
from imageio import imread
import torch
from scipy import io
import numpy as np
from scipy.signal import correlate2d
from scipy.ndimage import maximum_position
from torch.utils.data import Dataset as TorchDataset
from torchvision.transforms import Resize

from conversion import xyz2uv, uv2xyz, uv2lonlat, lonlat2uv, uv2pixel, xyz2depth, depth2xyz
from utils.boundary import corners2boundaries, corners2boundary, layout2depth, visibility_corners, boundary_type, \
    corners2boundaries1
from utils.visibility_polygon import calc_visible_polygon
from .BaseDataset import BaseDataset
from .SharedFunctions import read_depth, gen_path_
from ..Conversion import xyz2lonlat, lonlat2xyz
from .SharedFunctions import *
import torchvision.transforms as transforms
from PIL import Image

class Matterport3DDataset(BaseDataset):
    def __init__(self, dataset_image_path, dataset_label_path,
                 dataset_layout_depth_path,dataset_depth_path,dataset_seg_path,mode, shape,shape1,
                 image_name, wall_types, aug, camera_height,
                 **kwargs):
        super().__init__(**kwargs)
        self.shape = shape
        self.shape1 = shape1
        self.aug = aug
        self.camera_height = camera_height
        self.grid, self.unit_lonlat, self.unit_xyz = create_grid(shape)
        self.max_wall_num = max(wall_types)
        self.cH = 1.6

        with open('%s/mp3d_%s.txt' % (dataset_label_path, mode), 'r') as f: lst = [x.rstrip().split() for x in f]
        # 设置照片地址
        rgb_lst = [gen_path(dataset_image_path, x, image_name) for x in lst]
        layout_depth_lst = [gen_path(dataset_layout_depth_path, x, image_name) for x in lst]
        depth_lst = [gen_path(dataset_depth_path, x, image_name) for x in lst]
        seg_lst = [gen_path(dataset_seg_path, x, image_name) for x in lst]
        label_lst = ['%s/label/%s_%s.json' % (dataset_label_path, *x,) for x in lst]

        rgb_lst, layout_depth_lst,depth_lst,seg_lst, label_lst = filter_by_wall_num(rgb_lst, layout_depth_lst,depth_lst,seg_lst, label_lst,wall_types)

        self.data = list(zip(rgb_lst, layout_depth_lst,depth_lst,seg_lst, label_lst))

    def __getitem__(self, idx):

        rgb_path, layout_depth_path, depth_path, seg_path, label_path = self.data[idx]
        label = read_label(label_path, self.camera_height)
        pts = get_contour_3D_points(label['xyz'], label['point_idxs'], label['cameraCeilingHeight'])
        rgb = read_image(rgb_path, self.shape)
        num = pts.shape[0]
        h, w, _ = rgb.shape
        height = h//4
        width = w//4

        depth = np.array(Image.open(depth_path)) / 4000.
        depth = np.expand_dims(depth, axis=0)

        layout_depth = np.array(Image.open(layout_depth_path)) / 1000
        layout_depth = np.expand_dims(layout_depth, axis=0)

        seg = np.array(Image.open(seg_path))
        seg = np.expand_dims(seg, axis=0)


        aug = self.aug
        # if aug['flip'] and np.random.randint(2) == 0: #翻转
        #     rgb = np.flip(rgb, axis=1).copy()
        #     layout_depth = np.flip(layout_depth, axis=2).copy()
        #     depth = np.flip(depth, axis=2).copy()
        #     seg = np.flip(seg, axis=2).copy()

        # if aug['light']:
        #     p = np.random.uniform(1, 2)
        #     if np.random.randint(2) == 0: p = 1 / p
        #     rgb = rgb ** p
        #
        # if aug['color']:
        #     rgb = (rgb * 255).astype(np.uint8)
        #     rgb = Image.fromarray(rgb)
        #     color_jitter = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
        #     rgb = color_jitter(rgb)
        #     # img = np.array(rgb, np.float32)[..., :3] / 255.  #512   1024 3
        #     # rgb = torch.tensor(rgb)
        #
        # if aug['norm']:
        #     to_tensor = transforms.ToTensor()
        #     normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        #     rgb = to_tensor(rgb)
        #     rgb = normalize(rgb)
        #     # rgb = rgb.permute(1, 2, 0)

        depth = torch.tensor(depth)
        layout_depth = torch.tensor(layout_depth)
        seg = torch.tensor(seg).float()

        rgb = torch.tensor(rgb)
        rgb = rgb.permute(2, 0, 1)
        resize2 = torchvision.transforms.Resize((128, 256), interpolation=0)
        rgb = resize2(rgb)#.cpu().numpy()

        if aug['concat']:
            # 生成随机裁剪宽度
            max_crop_width = width - 1
            crop_width = random.randint(20, max_crop_width-20)
            cropped_img = rgb[:, :, :crop_width]
            padded_img = torch.cat((rgb[:, :, crop_width:], cropped_img), dim=2)
            # padded_img = padded_img.permute(1, 2, 0)
            # imshow(padded_img.detach())
            cropped_depth = depth[:, :, :crop_width]
            padded_depth = torch.cat((depth[:, :, crop_width:], cropped_depth), dim=2)
            cropped_seg = seg[:, :, :crop_width]
            padded_seg = torch.cat((seg[:, :, crop_width:], cropped_seg), dim=2)
            cropped_layout_depth = layout_depth[:, :, :crop_width]
            padded_layout_depth = torch.cat((layout_depth[:, :, crop_width:], cropped_layout_depth), dim=2)

            if padded_seg[:, 64, 1] != 3:
                padded_seg[padded_seg == 3] = 4
                lab = padded_seg[:, 64, 1]
                max_value = padded_seg.max()
                mask = (padded_seg[:, :, 128:] == lab)
                padded_seg[:, :, 128:][mask] = max_value+1
            # padded_seg1 = padded_seg.cpu().numpy()
        out = {
            'rgb': padded_img,
            'seg': padded_seg,
            'depth': padded_depth,
            'layout_depth': padded_layout_depth,
            'wall-num': num,
            'id': label['id'],
        }
        return out

import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

unloader = torchvision.transforms.ToPILImage()


def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    array1 = image.numpy()  # 将tensor数据转为numpy数据
    maxValue = array1.max()
    array1 = array1 * 255 / maxValue  # normalize，将图像数据扩展到[0,255]
    image = np.uint8(array1)  # float32-->uint8
    # image = unloader(image)
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    if title is not None:
        plt.title(title)
    plt.pause(1)  # pause a bit so that plots are updated