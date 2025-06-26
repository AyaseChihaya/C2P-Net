import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import cv2
from .. import Conversion
from ..Conversion.EquirecCoordinate import lonlat2uv


class RenderLoss(nn.Module):
    def __init__(self, camera_height=1.6):
        super(RenderLoss, self).__init__()
        assert camera_height > 0
        self.cH = camera_height
        self.grid = None
        self.c2d = Corner2Depth(None)
        self.et = Conversion.EquirecTransformer('torch')

    def setGrid(self, grid):
        self.grid = grid
        self.c2d.setGrid(grid)

    def lonlat2xyz_up(self, GT_up, up_down_ratio):
        GT_up_xyz = self.et.lonlat2xyz(GT_up)
        s = -(self.cH * up_down_ratio[..., None, None]) / GT_up_xyz[..., 1:2]
        GT_up_xyz *= s

        return GT_up_xyz

    def uv2lonlat(self, uv, axis=None):
        if axis is None:
            lon = (uv[..., 0:1] - 0.5) * 2 * np.pi
            lat = (uv[..., 1:] - 0.5) * np.pi
        elif axis == 0:
            lon = (uv - 0.5) * 2 * np.pi
            return lon
        elif axis == 1:
            lat = (uv - 0.5) * np.pi
            return lat
        else:
            assert False, "axis error"

        lst = [lon, lat]
        lonlat = np.concatenate(lst, axis=-1) if isinstance(uv, np.ndarray) else torch.cat(lst, dim=-1)
        return lonlat

    def forward(self, GT_up, corner_nums, up_down_ratio):
        # pred_up, pred_down, GT_up is lonlat
        assert self.grid is not None
        GT_cen_xyz = self.lonlat2xyz_up(GT_up, up_down_ratio)
        gt_normal = self.c2d(GT_cen_xyz, corner_nums)

        return gt_normal


class Corner2Depth(nn.Module):
    def __init__(self, grid):
        super(Corner2Depth, self).__init__()
        self.grid = grid

    def setGrid(self, grid):
        self.grid = grid

    def forward(self, corners, nums, shift=None, mode='origin'):
        if mode == 'origin':
            return self.forward_origin(corners, nums, shift)
        else:
            return self.forward_fast(corners, nums, shift)

    def forward_fast(self, corners, nums, shift=None):
        if shift is not None: raise NotImplementedError
        grid_origin = self.grid.to(corners.device)
        eps = 1e-2
        depth_maps = []
        normal_maps = []

        for i, num in enumerate(nums):
            grid = grid_origin.clone()
            corners_now = corners[i, ...].clone()
            corners_now = torch.cat([corners_now, corners_now[0:1, ...]], dim=0)
            diff = corners_now[1:, ...] - corners_now[:-1, ...]
            vec_yaxis = torch.zeros_like(diff)
            vec_yaxis[..., 1] = 1
            cross_result = torch.cross(diff, vec_yaxis, dim=1)
            d = -torch.sum(cross_result * corners_now[:-1, ...], dim=1, keepdim=True)
            planes = torch.cat([cross_result, d], dim=1)
            scale_all = -planes[:, 3] / torch.matmul(grid, planes[:, :3].T)

            intersec = []
            for idx in range(scale_all.shape[-1]):
                intersec.append((grid * scale_all[..., idx:idx + 1]).unsqueeze(-1))
            intersec = torch.cat(intersec, dim=-1)
            a = corners_now[1:, ...]
            b = corners_now[:-1, ...]

            x_cat = torch.cat([a[:, 0:1], b[:, 0:1]], dim=1)
            z_cat = torch.cat([a[:, 2:], b[:, 2:]], dim=1)

            max_x, min_x = torch.max(x_cat, dim=1)[0], torch.min(x_cat, dim=1)[0]
            max_z, min_z = torch.max(z_cat, dim=1)[0], torch.min(z_cat, dim=1)[0]

            mask_x = (intersec[:, :, :, 0, :] <= max_x + eps) & (intersec[:, :, :, 0, :] >= min_x - eps)
            mask_z = (intersec[:, :, :, 2, :] <= max_z + eps) & (intersec[:, :, :, 2, :] >= min_z - eps)
            mask_valid = scale_all > 0
            mask = ~(mask_x & mask_z & mask_valid)
            scale_all[mask] = float('inf')

            depth, min_idx = torch.min(scale_all, dim=-1)
            _, h, w = min_idx.shape
            normal = planes[min_idx.view(-1), :3].view(1, h, w, -1)
            depth_maps.append(depth)
            normal_maps.append(normal)
        depth_maps = torch.cat(depth_maps, dim=0).unsqueeze(1)
        normal_maps = torch.cat(normal_maps, dim=0)

        return depth_maps, normal_maps

    # def forward_origin(self, corners, corner_point, nums, shift=None):
    #     normal_maps = []
    #     for i, num in enumerate(nums):
    #         corners_now = corners[i, :num, ...].clone()  # num x 3
    #         if shift is not None:
    #             corners_now[..., 0] -= shift[i, 0]
    #             corners_now[..., 2] -= shift[i, 1]
    #         # equation: ax + by + cz + d = 0
    #         corners_now = torch.cat(
    #             [corners_now, corners_now[0:1, ...]], dim=0)
    #         planes = []
    #         for j in range(1, corners_now.shape[0]):
    #             vec_corner = corners_now[j:j + 1, ...] - corners_now[j - 1:j, ...]
    #             vec_yaxis = torch.zeros_like(vec_corner)
    #             vec_yaxis[..., 1] = 1
    #             cross_result = torch.cross(vec_corner, vec_yaxis)
    #             # now corss_result is a b c
    #             cross_result = cross_result / \
    #                            torch.norm(cross_result, p=2, dim=-1)[..., None]
    #             d = -torch.sum(cross_result *
    #                            corners_now[j:j + 1, ...], dim=-1)[..., None]
    #             abcd = torch.cat([cross_result, d], dim=-1)  # abcd is 1, 4
    #             planes.append(abcd)
    #         planes = torch.cat(planes, dim=0)  # planes is num x 4
    #         abc = planes[:, :3]
    #         assert abc.shape[0] == num
    #         abc = torch.cat([abc, abc[0:1, :]], dim=0)
    #         each_corner = corner_point[i, :, :]
    #         corner_index = torch.nonzero(each_corner.sum(dim=0)).squeeze()
    #         real_corner = each_corner[:, corner_index]  # 2, n
    #         mean_u_corner = real_corner[0:1, :]  # bz,n
    #         # 根据真实值角点坐标提取索引位置
    #         normal_map = []
    #         index_cu_image1 = torch.round(mean_u_corner * (1024 - 1)).long()  # 点u的索引位置
    #         for jj in range(index_cu_image1.shape[1] - 1):
    #             number = index_cu_image1[:, jj + 1] - index_cu_image1[:, jj]
    #             abc1 = abc[jj, :].unsqueeze(1)
    #             abc1 = abc1.cpu().numpy()
    #             abc1 = np.tile(abc1, (1, number.cpu()))
    #             abc1 = torch.as_tensor(abc1)
    #             normal_map.append(abc1)
    #         normal_map_ = torch.cat(normal_map, dim=1)
    #         normal_map_ = torch.cat([normal_map_, normal_map_[:, 0:1]], dim=1).unsqueeze(0)  # 3,1024
    #         normal_maps.append(normal_map_)
    #     normal_maps = torch.cat(normal_maps, dim=0)
    #     return normal_maps  # 8 3 1024

    def forward_origin(self, corners, nums, shift=None):
        normal_maps=[]
        for i, num in enumerate(nums):
            corners_now = corners[i, :num, ...].clone()  # num x 3
            if shift is not None:
                corners_now[..., 0] -= shift[i, 0]
                corners_now[..., 2] -= shift[i, 1]
            # m = corners_now.shape[0]
            # corners_now = torch.cat([corners_now[m-1:m, ...], corners_now, corners_now[0:1, ...]], dim=0)
            corners_now = torch.cat([corners_now, corners_now[0:1, ...]], dim=0)
            planes = []
            for j in range(1, corners_now.shape[0]):
                vec_corner = corners_now[j:j+1, ...] - corners_now[j-1:j, ...]
                vec_yaxis = torch.zeros_like(vec_corner)
                vec_yaxis[..., 1] = 1
                cross_result = torch.cross(vec_corner, vec_yaxis)
                cross_result = cross_result /torch.norm(cross_result, p=2, dim=-1)[..., None]
                d = -torch.sum(cross_result *
                               corners_now[j:j+1, ...], dim=-1)[..., None]
                abcd = torch.cat([cross_result, d], dim=-1)  # abcd is 1, 4
                planes.append(abcd)
            planes = torch.cat(planes, dim=0)  # planes is num x 4
            assert planes.shape[0] == num
            normal = planes[:, :3]
            normal = torch.cat([normal, normal[0:1, ...]], dim=0)
            normals = torch.zeros(13, 3)
            num_1 = normal.shape[0]
            normals[:num_1, :] = normal
            normals = normals.unsqueeze(0)
            normal_maps.append(normals)
        normal_maps = torch.cat(normal_maps, dim=0)
        return normal_maps

class ShiftSampler(nn.Module):
    def __init__(self, dim=256, down_ratio=0.5):
        super(ShiftSampler, self).__init__()
        self.dim = dim
        self.down_ratio = down_ratio
        self.grid_x, self.grid_z = np.meshgrid(range(dim), range(dim))

    def _GetAngle(self, pred):
        [num, _] = pred.shape
        tmp = np.concatenate([pred, pred[0:1, :]], axis=0)
        abs_cos = []

    def forward(self, pred_xyz, pred_corner_num, gt_xyz, gt_corner_num):
        #
        # pred_xyz bs x 12 x 3
        # pred_corner_num bs,
        # gt_xyz bs x 12 x 3
        # gt_corner_num bs,
        #
        device = pred_xyz.device
        out = np.zeros([pred_xyz.shape[0], 2], dtype=np.float32)

        pred_xyz = pred_xyz.data.cpu().numpy() * self.down_ratio
        pred_corner_num = pred_corner_num.data.cpu().numpy()
        gt_xyz = gt_xyz.data.cpu().numpy() * self.down_ratio
        gt_corner_num = gt_corner_num.data.cpu().numpy()

        for i in range(pred_xyz.shape[0]):
            # first find boundary (max/min xz)
            max_x1 = pred_xyz[i, :pred_corner_num[i], 0].max()
            max_x2 = gt_xyz[i, :gt_corner_num[i], 0].max()
            min_x1 = pred_xyz[i, :pred_corner_num[i], 0].min()
            min_x2 = gt_xyz[i, :gt_corner_num[i], 0].min()

            max_z1 = pred_xyz[i, :pred_corner_num[i], 2].max()
            max_z2 = gt_xyz[i, :gt_corner_num[i], 2].max()
            min_z1 = pred_xyz[i, :pred_corner_num[i], 2].min()
            min_z2 = gt_xyz[i, :gt_corner_num[i], 2].min()

            max_x = np.max([max_x1, max_x2])
            min_x = np.min([min_x1, min_x2])
            max_z = np.max([max_z1, max_z2])
            min_z = np.min([min_z1, min_z2])

            pred_xyz_now_normalized = pred_xyz[i, :pred_corner_num[i], :].copy()
            self._GetAngle(pred_xyz_now_normalized)

            pred_xyz_now_normalized[:, 0] = (pred_xyz_now_normalized[:, 0] - min_x) / (max_x - min_x)
            pred_xyz_now_normalized[:, 2] = (pred_xyz_now_normalized[:, 2] - min_z) / (max_z - min_z)

            gt_xyz_now_normalized = gt_xyz[i, :gt_corner_num[i], :].copy()
            gt_xyz_now_normalized[:, 0] = (gt_xyz_now_normalized[:, 0] - min_x) / (max_x - min_x)
            gt_xyz_now_normalized[:, 2] = (gt_xyz_now_normalized[:, 2] - min_z) / (max_z - min_z)

            pred_xz_now_normalized = (pred_xyz_now_normalized[..., ::2] * (self.dim - 1)).round().astype(np.int)
            gt_xz_now_normalized = (gt_xyz_now_normalized[..., ::2] * (self.dim - 1)).round().astype(np.int)

            pred_mask = np.zeros([self.dim, self.dim], np.uint8)
            gt_mask = np.zeros([self.dim, self.dim], np.uint8)

            cv2.drawContours(pred_mask, [pred_xz_now_normalized], -1, 255, cv2.FILLED)
            cv2.drawContours(gt_mask, [gt_xz_now_normalized], -1, 255, cv2.FILLED)

            mask = np.logical_and(pred_mask.astype(np.bool), gt_mask.astype(np.bool))
            x_valid = self.grid_x[mask]
            z_valid = self.grid_z[mask]
            idx_choice = np.random.choice(range(z_valid.shape[0]))
            if False:
                plt.subplot('311')
                plt.imshow(pred_mask)
                plt.subplot('312')
                plt.imshow(gt_mask)
                plt.subplot('313')
                plt.imshow(mask)
                plt.show()
            x_choose = x_valid[idx_choice].astype(np.float32)
            z_choose = z_valid[idx_choice].astype(np.float32)
            out[i, 0] = (x_choose / (self.dim - 1)) * (max_x - min_x) + min_x
            out[i, 1] = (z_choose / (self.dim - 1)) * (max_z - min_z) + min_z

        return torch.FloatTensor(out).to(device)
