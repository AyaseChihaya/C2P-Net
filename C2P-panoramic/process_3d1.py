from genericpath import isdir
from logging import root
import os
from  utils.conversion import xyz2depth,xyz2uv,depth2xyz,uv2pixel,pixel2uv
from utils.boundary import calc_rotation, corners2boundary, boundary_type, corners2boundaries, layout2depth, \
    corners2boundaries1
from utils.conversion import uv2xyz
from posixpath import split
import numpy as np
from PIL import Image
from scipy import io
from shapely.geometry import LineString
from scipy.spatial.distance import cdist
import glob
import json
import itertools
import pandas as pd
import cv2
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
import panostretch
# from transformations_torch import *




def get_depth(corners, plan_y=1, length=1024, visible=True):
    # 两个角点之间的像素值----边界点坐标
    visible_floor_boundary = corners2boundary(corners, length=length, visible=visible)
    # The horizon-depth relative to plan_y
    visible_depth = xyz2depth(uv2xyz(visible_floor_boundary, plan_y), plan_y)
    return visible_depth


def sort_xy_filter_unique(xs, ys, y_small_first=True):
    xs, ys = np.array(xs), np.array(ys)
    idx_sort = np.argsort(xs + ys / ys.max() * (int(y_small_first)*2-1))
    xs, ys = xs[idx_sort], ys[idx_sort]
    _, idx_unique = np.unique(xs, return_index=True)
    xs, ys = xs[idx_unique], ys[idx_unique]
    assert np.all(np.diff(xs) > 0)
    return xs, ys

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

def list_img_files(directory):
    """
    列出指定目录下所有文件名
    """
    txt_file_names = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            txt_file_names.append( filename)
    return txt_file_names

def list_txt_files(directory):
    """
    列出指定目录下所有 .txt 文件名
    """
    txt_file_names = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            txt_file_names.append(directory +filename)
    return txt_file_names



if __name__ == '__main__':

            path = '/home/ps/文档/diu1/5ZKStnWn8Zo_54b7c9b192714638ac7b01b4a5400d29.txt'
            with open(os.path.join(path)) as f:
                cor = np.array([line.strip().split() for line in f if line.strip()], np.float32)
            # H, W, c = img.shape
            # cor  = np.array(cor)
            w = 256
            h = 128
            n_cor = len(cor)
            max_wall_num = 20
            seg_map = np.zeros((max_wall_num, h, w))
            cnt = 5.0
            max_value = cor[:,0].max()

            for i in range(n_cor // 2):
                if i>=20:
                    print(1)
                if cor[i * 2,0] != max_value:
                    long = np.abs(cor[i*2]-cor[(i*2+2)%n_cor])[0]
                    xys1 = panostretch.pano_connect_points(cor[i * 2],
                                                          cor[(i * 2 + 2) % n_cor],
                                                          z=-50, w=w, h=h)
                    if cor[i * 2,0]==cor[(i * 2 + 2) % n_cor,0]:
                        # cor[i * 2,0] = cor[i * 2,0]-1
                        continue

                    bon_ceil_x=(xys1[:, 0])
                    bon_ceil_y=(xys1[:, 1])
                    gt_ceil_boundary = (np.vstack([bon_ceil_x,bon_ceil_y]).T).astype(np.int)
                    gt_ceil_boundary = gt_ceil_boundary[0:-1,:]
                # for i in range(n_cor // 2):
                    xys2 = panostretch.pano_connect_points(cor[i * 2 + 1],
                                                          cor[(i * 2 + 3) % n_cor],
                                                          z=50, w=w, h=h)
                    if cor[i * 2+1,0]==cor[(i * 2 + 3) % n_cor,0]:
                        # cor[i * 2+1,0] = cor[i * 2+1,0]-1
                        continue
                    bon_floor_x=(xys2[:, 0])
                    bon_floor_y=(xys2[:, 1])
                    gt_floor_boundary = (np.vstack([bon_floor_x,bon_floor_y]).T).astype(np.int)
                    gt_floor_boundary = gt_floor_boundary[0:-1, :]
                    for mi in range(gt_ceil_boundary.shape[0]):
                        ceil1 = gt_ceil_boundary[mi,:]
                        floor1 = gt_floor_boundary[mi, :]

                        X1 = ceil1[0]
                        seg_map[i, ceil1[1]:floor1[1] + 1, X1] = cnt  # 墙面
                        seg_map[i, 0:ceil1[1], X1] = 1  # 天花板
                        seg_map[i, floor1[1] + 1:h, X1] = 2  # 地面
                else:
                    xys1 = panostretch.pano_connect_points(cor[i * 2],
                                                           cor[(i * 2 + 2) % n_cor],
                                                           z=-50, w=w, h=h)
                    bon_ceil_x = (xys1[:, 0])
                    bon_ceil_y = (xys1[:, 1])
                    gt_ceil_boundary = (np.vstack([bon_ceil_x, bon_ceil_y]).T).astype(np.int)
                    gt_ceil_boundary = gt_ceil_boundary[0:-1, :]
                    # for i in range(n_cor // 2):
                    xys2 = panostretch.pano_connect_points(cor[i * 2 + 1],
                                                           cor[(i * 2 + 3) % n_cor],
                                                           z=50, w=w, h=h)
                    bon_floor_x = (xys2[:, 0])
                    bon_floor_y = (xys2[:, 1])
                    gt_floor_boundary = (np.vstack([bon_floor_x, bon_floor_y]).T).astype(np.int)
                    gt_floor_boundary = gt_floor_boundary[0:-1, :]

                    n = gt_floor_boundary[:, 0] < (w / 2)
                    nn = n[n == True].size
                    nnn = len(n)-nn
                    gt_floor_boundary_last = gt_floor_boundary[0:nnn, :]
                    gt_ceil_boundary_last = gt_ceil_boundary[0:nnn:, :]
                    gt_floor_boundary_front = gt_floor_boundary[nnn:, :]
                    gt_ceil_boundary_front = gt_ceil_boundary[nnn:, :]
                    for mi in range(gt_floor_boundary_front.shape[0]):
                        ceil1 = gt_ceil_boundary_front[mi, :]
                        floor1 = gt_floor_boundary_front[mi, :]
                        X1 = ceil1[0]
                        seg_map[i, ceil1[1]:floor1[1] + 1, X1] = 3  # 墙面
                        seg_map[i, 0:ceil1[1], X1] = 1  # 天花板
                        seg_map[i, floor1[1] + 1:h, X1] = 2  # 地面
                    for mi in range(gt_floor_boundary_last.shape[0]):
                        ceil1 = gt_ceil_boundary_last[mi, :]
                        floor1 = gt_floor_boundary_last[mi, :]
                        X1 = ceil1[0]
                        seg_map[i, ceil1[1]:floor1[1] + 1, X1] = 4  # 墙面
                        seg_map[i, 0:ceil1[1], X1] = 1  # 天花板
                        seg_map[i, floor1[1] + 1:h, X1] = 2  # 地面
                cnt += 1
            seg_map = torch.as_tensor(seg_map)
            seg_map = torch.sum(seg_map, dim=0)
            imshow(seg_map.detach())
            # save_name_depth = r'/home/ubuntu/data/dataset/seg_mat/' + img_path.split("/")[
            #     -3] + "_" + \
            #                   img_path.split("/")[-1].split(".")[0] + ".mat"
            # io.savemat(save_name_depth, {'seg_map': seg_map.detach().data.cpu().numpy()})
            # flag = flag + 1
            # print(flag)

