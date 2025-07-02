import os
import sys
import cv2
import json
import numpy as np
import torch
from imageio import imread

from conversion import xyz2uv, uv2xyz
from ..Conversion import XY2xyz, xyz2XY, xyz2lonlat, lonlat2xyz

__all__ = [
    'gen_path',
    'filter_by_wall_num',
    'read_image',
    'read_label',
    'get_contour_3D_points',
    'plane_to_depth',
    'render_rgb',
    'create_grid'
]


def gen_path_(root_path, pano_id, file_name):
    path = root_path
    # for x1 in pano_id[:1]:
    #     path1 = os.path.join(path, x1)
    for x2 in pano_id[1:2]:
        path2 = os.path.join(path + x2 + '_')
    # return os.path.join(path, file_name)

    return path2 + file_name


def gen_path(root_path, pano_id, file_name):
    path = root_path
    for x1 in pano_id[:1]:
        path1 = os.path.join(path, x1)
    for x2 in pano_id[1:2]:
        path2 = os.path.join(path1 + '_' + x2)
    # return os.path.join(path, file_name)

    return path2 + file_name


def filter_by_wall_num(rgb_lst, layout_depth_lst,depth_lst,seg_lst, label_lst,wall_types):
    new_rgb = []
    new_layout_depth = []
    new_depth = []
    new_seg = []
    new_label = []
    for i, one in enumerate(label_lst):
        with open(one, 'r') as f:
            label = json.load(f)
        if label['layoutWalls']['num'] in wall_types:
            new_rgb.append(rgb_lst[i])
            new_layout_depth.append(layout_depth_lst[i])
            new_depth.append(depth_lst[i])
            new_seg.append(seg_lst[i])
            new_label.append(label_lst[i])

    return new_rgb, new_layout_depth,new_depth,new_seg,new_label


def read_image(image_path, shape):
    img = imread(image_path, pilmode='RGB').astype(np.float32) / 255
    if img.shape[0] != shape[0] or img.shape[1] != shape[1]: img = cv2.resize(img, dsize=tuple(shape[::-1]),
                                                                              interpolation=cv2.INTER_AREA)

    return img


def read_depth(depth_path, shape):
    depth = imread(depth_path, pilmode='RGB').astype(np.float32) / 255
    if depth.shape[0] != shape[0] or depth.shape[1] != shape[1]: depth = cv2.resize(depth, dsize=tuple(shape[::-1]),
                                                                                    interpolation=cv2.INTER_AREA)

    return depth


def read_label(label_path, cH):
    with open(label_path, 'r') as f: label = json.load(f)  # 加载json文件
    scale = cH / label['cameraHeight']
    camera_height = cH
    # 经过缩放计算后的摄像机到天花板高度
    camera_ceiling_height = scale * label.get('cameraCeilingHeight', label['layoutHeight'] - label['cameraHeight'])
    # 原始的摄像机到天花板高度
    camera_ceiling_height_ori = label.get('cameraCeilingHeight', label['layoutHeight'] - label['cameraHeight'])
    layout_height = scale * label['layoutHeight']
    up_down_ratio = camera_ceiling_height_ori / label['cameraHeight']  # 使用相机高度乘以比例值确定天花板在图像中的高度

    xyz = [one['xyz'] for one in label['layoutPoints']['points']]
    planes = [one['planeEquation'] for one in label['layoutWalls']['walls']]
    point_idxs = [one['pointsIdx'] for one in label['layoutWalls']['walls']]

    R_180 = cv2.Rodrigues(np.array([0, -1 * np.pi, 0], np.float32))[0]
    xyz = np.asarray(xyz)
    xyz[:, 0] *= -1
    xyz = xyz.dot(R_180.T)
    planes += [[0, 1, 0, camera_ceiling_height_ori], [0, 1, 0, -label['cameraHeight']]]
    planes = np.asarray(planes)
    planes[:, :3] = planes[:, :3].dot(R_180.T)

    xyz *= scale
    planes[:, 3] *= scale
    xyz_ = np.copy(xyz)
    xyz_[:, 1] = camera_height
    # xyz_[:, 2] *= -1
    coords = xyz2uv(xyz_)  # 返回角点的二维图像坐标
    # xxx = uv2xyz(coords, 1)

    xyz_1 = np.copy(xyz_)
    # xyz_1[:, 2] *= -1
    xyz_1[:, 1] = -camera_ceiling_height

    coords_ceil = xyz2uv(xyz_1)  # 返回角点的二维图像坐标
    coords_floor = np.copy(coords)
    normal = [one['normal'] for one in label['layoutWalls']['walls']]
    wall_number = label['layoutPoints']['num']

    xyzs = [one['xyz'] for one in label['layoutPoints']['points']]
    # assert len(xyzs) == len(point_idxs), "len(xyz) != len(point_idx)"
    # xyzs = [xyzs[i] for i in point_idxs]
    xyzs = np.asarray(xyzs, dtype=np.float32)
    xyzs[:, 2] *= -1  # 将xyz数组中所有行的第三列（z坐标）乘以-1，实现将z坐标反转的操作
    xyzs[:, 1] = camera_height  # 将xyz数组中所有行的第二列（y坐标）替换为camera_height的值，这样可以将所有点的y坐标设为相机的高度。
    corners = xyz2uv(xyzs)  # 返回角点的二维图像坐标


    out = {
        'cameraHeight': camera_height,
        'layoutHeight': layout_height,
        'cameraCeilingHeight': camera_ceiling_height,
        'xyz': xyz,
        'xyz_': xyz_,
        'coords': corners,
        'coords_floor': coords_floor,
        'coords_ceil': coords_ceil,
        'planes': planes,
        'point_idxs': point_idxs,
        'normal': normal,
        'wall_number': wall_number,
        'ratio': np.array([up_down_ratio], dtype=np.float32),
        'id': os.path.basename(label_path).split('.')[0]
    }
    return out


def get_contour_3D_points(xyz, points_idx, ccH):
    pts = np.asarray([xyz[i] for i in points_idx], np.float32).reshape([-1, 3])[::2, :].copy()
    pts[:, 1] = -ccH

    return pts

def get_contour_3D_points1(xyz, points_idx, ccH):
    pts = np.asarray([xyz[i] for i in points_idx], np.float32).reshape([-1, 3])[::2, :].copy()
    pts[:, 1] = ccH

    return pts


def plane_to_depth(grid, planes, points, idxs, ch, cch):
    [h, w, _] = grid.shape
    scale_lst = []
    inter_lst = []
    eps = 1e-2
    for i, plane in enumerate(planes):
        s = -plane[3] / np.dot(grid, plane[:3].reshape([3, 1]))
        intersec = s * grid
        inter_lst.append(intersec[:, :, None, :])
        if i <= len(planes) - 3:
            idx = idxs[i]
            rang = np.concatenate([points[idx[0]][None, :], points[idx[1]][None, :]], axis=0)
            mx_x, mn_x = np.max(rang[:, 0]), np.min(rang[:, 0])
            mx_z, mn_z = np.max(rang[:, 2]), np.min(rang[:, 2])
            mask_x = np.logical_and(intersec[:, :, 0] <= mx_x + eps, intersec[:, :, 0] >= mn_x - eps)
            mask_z = np.logical_and(intersec[:, :, 2] <= mx_z + eps, intersec[:, :, 2] >= mn_z - eps)
            mask = 1 - np.logical_and(mask_x, mask_z)
            mask = np.logical_or(mask, s[:, :, 0] < 0)
        else:
            mask = 1 - np.logical_and(intersec[:, :, 1] <= ch + eps, intersec[:, :, 1] >= -cch - eps)
            mask = np.logical_or(mask, s[:, :, 0] < 0)
        s[mask[:, :, None]] = np.inf
        scale_lst.append(s)
    scale = np.concatenate(scale_lst, axis=2)
    inter = np.concatenate(inter_lst, axis=2)
    min_idx = np.argmin(scale, axis=2)
    x, y = np.meshgrid(range(w), range(h))
    depth_pred = scale[y.ravel(), x.ravel(), min_idx.ravel()].reshape([h, w])
    intersec = inter[y.ravel(), x.ravel(), min_idx.ravel(), :].reshape([h, w, 3])

    return depth_pred, intersec


def render_rgb(pts, rgb, shape):
    xy = xyz2XY(pts.astype(np.float32), shape, mode='numpy')
    new_rgb = cv2.remap(rgb, xy[..., 0], xy[..., 1], interpolation=cv2.INTER_LINEAR)

    return new_rgb


def render_depth(pts, depth, shape):
    xy = xyz2XY(pts.astype(np.float32), shape, mode='numpy')
    new_depth = cv2.remap(depth, xy[..., 0], xy[..., 1], interpolation=cv2.INTER_LINEAR)

    return new_depth


def create_grid(shape):  # 执行坐标转换
    h, w = shape
    h = h//2
    w = w//2
    X = np.tile(np.arange(w)[None, :, None], (h, 1, 1))
    Y = np.tile(np.arange(h)[:, None, None], (1, w, 1))
    XY = np.concatenate([X, Y], axis=-1)
    xyz = XY2xyz(XY, shape, mode='numpy')  # 将网格坐标 XY 转换为三维坐标 xyz

    # mean_lonlat = torch.Tensor(XY) * (2.0 * torch.Tensor([np.pi, np.pi / 2]) / torch.Tensor([w, h])) - torch.Tensor([np.pi, np.pi / 2])
    # mean_xyz = lonlat2xyz(mean_lonlat, mode='numpy')  # 三维坐标mean_xyz

    l = w #// 4
    mean_lonlat = np.zeros([l, 2], dtype=np.float32)  # 第一列设置为经度，第二列设置为纬度
    mean_lonlat[:, 1] = 0
    mean_lonlat[:, 0] = ((np.arange(l) / float(l - 1)) * 2 * np.pi - np.pi).astype(np.float32)
    mean_xyz = lonlat2xyz(mean_lonlat, mode='numpy')  # 三维坐标mean_xyz

    return xyz, mean_lonlat, mean_xyz
