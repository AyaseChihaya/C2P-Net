import os
import sys
# import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpherePadGrid(object):
    def __init__(self, cube_dim, equ_h, FoV=90.0):
        self.cube_dim = cube_dim
        self.equ_h = equ_h
        self.equ_w = equ_h * 2
        self.FoV = FoV / 180.0 * np.pi
        self.r_lst = np.array([
            [0, -180.0, 0],
            [90.0, 0, 0],
            [0, 0, 0],
            [0, 90, 0],
            [0, -90, 0],
            [-90, 0, 0]
        ], np.float32) / 180.0 * np.pi
        self.R_lst = [cv2.Rodrigues(x)[0] for x in self.r_lst]
        self._getCubeGrid()

    def _getCubeGrid(self):
        f = 0.5 * self.cube_dim / np.tan(0.5 * self.FoV)
        cx = (self.cube_dim - 1) / 2
        cy = cx
        self.intrisic = {
                    'f': float(f),
                    'cx': float(cx),
                    'cy': float(cy)
                }
        x = np.tile(np.arange(self.cube_dim)[None, ..., None], [self.cube_dim, 1, 1])
        y = np.tile(np.arange(self.cube_dim)[..., None, None], [1, self.cube_dim, 1])
        ones = np.ones_like(x)
        xyz = np.concatenate([x, y, ones], axis=-1)
        K = np.array([
            [f, 0, cx],
            [0, f, cy],
            [0, 0, 1]
        ], np.float32)
        xyz = xyz @ np.linalg.inv(K).T
        xyz /= np.linalg.norm(xyz, axis=-1, keepdims=True)
        self.K = K
        self.grids = []
        self.grids_xyz = []
        for R in self.R_lst:
            tmp = xyz @ R # Don't need to transpose since we are doing it for points not for camera
            self.grids_xyz.append(tmp)
            lon = np.arctan2(tmp[..., 0:1], tmp[..., 2:]) / np.pi
            lat = np.arcsin(tmp[..., 1:2]) / (0.5 * np.pi)
            lonlat = np.concatenate([lon, lat], axis=-1)
            self.grids.append(torch.FloatTensor(lonlat[None, ...]))
    
    def __call__(self):
        R_lst = [torch.FloatTensor(x) for x in self.R_lst]
        grids_xyz = [torch.FloatTensor(x).view(1, self.cube_dim, self.cube_dim, 3) for x in self.grids_xyz]
        K = torch.FloatTensor(self.K)
        return R_lst, grids_xyz, self.intrisic

class SpherePad(nn.Module):
    def __init__(self, pad_size):
        super(SpherePad, self).__init__()
        self.pad_size = pad_size
        self.data = {}
        # pad order: up, down, left, right sides
        # use yes/no flag to choose flip/transpose or not
        # notation: #face-#side_#flip-hor_#flip_ver_#transpose
        # transpose is applied first
        self.relation = {
            'back': ['top-up_yes_yes_no', 'down-down_yes_yes_no', 'right-right_no_no_no', 'left-left_no_no_no'],
            'down': ['front-down_no_no_no', 'back-down_yes_yes_no', 'left-down_yes_no_yes', 'right-down_no_yes_yes'],
            'front': ['top-down_no_no_no', 'down-up_no_no_no', 'left-right_no_no_no', 'right-left_no_no_no'],

            'left': ['top-left_yes_no_yes', 'down-left_no_yes_yes', 'back-right_no_no_no', 'front-left_no_no_no'],
            'right': ['top-right_no_yes_yes', 'down-right_yes_no_yes', 'front-right_no_no_no', 'back-left_no_no_no'],
            'top': ['back-up_yes_yes_no', 'front-up_no_no_no', 'left-up_no_yes_yes', 'right-up_yes_no_yes']
        }

    def _GetLoc(self, R_lst, grid_lst, K):
        out = {}
        pad = self.pad_size
        f, cx, cy = K['f'], K['cx'], K['cy']
        K_mat = torch.FloatTensor(
            np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]]))
        grid_front = grid_lst[2]  # 1 x h x h x 3
        orders = ['back', 'down', 'front', 'left', 'right', 'top']
        for i, face in enumerate(orders):
            out[face] = {}
            for j, connect_side in enumerate(['up', 'down', 'left', 'right']):
                connected_face = self.relation[face][j].split('-')[0]
                idx = orders.index(connected_face)
                R_world_to_connected = R_lst[idx]  # 3 x 3
                R_world_to_itself = R_lst[i]  # 3 x 3
                R_itself_to_connected = torch.matmul(
                    R_world_to_connected, R_world_to_itself.transpose(0, 1))
                new_grid = torch.matmul(
                    grid_front, R_itself_to_connected.transpose(0, 1))
                proj = torch.matmul(new_grid, K_mat.transpose(0, 1))
                x = proj[:, :, :, 0:1] / proj[:, :, :, 2:3]
                y = proj[:, :, :, 1:2] / proj[:, :, :, 2:3]
                x = (x - cx) / cx
                y = (y - cy) / cy
                xy = torch.cat([x, y], dim=3)  # 1 x h x w x 2
                out[face][connect_side] = {}
                x = xy[:, :, :, 0:1]
                y = xy[:, :, :, 1:2]
                '''
                mask1 = np.logical_and(x >= -1.01, x <= 1.01)
                mask2 = np.logical_and(y >= -1.01, y <= 1.01)
                mask = np.logical_and(mask1, mask2)
                '''
                mask1 = (x >= -1.01) & (x <= 1.01)
                mask2 = (y >= -1.01) & (y <= 1.01)
                mask = mask1 & mask2

                xy = torch.clamp(xy, -1, 1)
                if connect_side == 'up':
                    out[face][connect_side]['mask'] = mask[:, :pad, :, :]
                    out[face][connect_side]['xy'] = xy[:, :pad, :, :]
                elif connect_side == 'down':
                    out[face][connect_side]['mask'] = mask[:, -pad:, :, :]
                    out[face][connect_side]['xy'] = xy[:, -pad:, :, :]
                elif connect_side == 'left':
                    out[face][connect_side]['mask'] = mask[:, :, :pad, :]
                    out[face][connect_side]['xy'] = xy[:, :, :pad, :]
                elif connect_side == 'right':
                    out[face][connect_side]['mask'] = mask[:, :, -pad:, :]
                    out[face][connect_side]['xy'] = xy[:, :, -pad:, :]

        return out

    def forward(self, inputs):
        [bs, c, h, w] = inputs.shape
        assert bs % 6 == 0 and h == w
        key = '(%d,%d,%d)' % (h, w, self.pad_size)
        if key not in self.data:
            theta = 2 * np.arctan((0.5 * h + self.pad_size) / (0.5 * h))
            grid_ori = SpherePadGrid(h, 2*h, 90)
            grid = SpherePadGrid(h+2*self.pad_size, 2*h, theta/np.pi * 180)
            R_lst, grid_lst, _ = grid()
            _, _, K = grid_ori()
            self.data[key] = self._GetLoc(R_lst, grid_lst, K)
        pad = self.pad_size
        orders = ['back', 'down', 'front', 'left', 'right', 'top']
        out = []
        for i, face in enumerate(orders):
            this_face = inputs[i::6]
            this_face = F.pad(this_face, (pad, pad, pad, pad))
            repeats = this_face.shape[0]
            for j, connect_side in enumerate(['up', 'down', 'left', 'right']):
                connected_face_name = self.relation[face][j].split('-')[0]
                connected_face = inputs[orders.index(connected_face_name)::6]
                mask = self.data[key][face][connect_side]['mask'].cuda().repeat(repeats, 1, 1, c).permute(0, 3, 1, 2).to(inputs.device)
                xy = self.data[key][face][connect_side]['xy'].cuda().repeat(repeats, 1, 1, 1).to(inputs.device)
                interpo = F.grid_sample(connected_face, xy, align_corners=True, mode='bilinear')
                if connect_side == 'up':
                    this_face[:, :, :pad, :][mask] = interpo[mask]
                elif connect_side == 'down':
                    this_face[:, :, -pad:, :][mask] = interpo[mask]
                elif connect_side == 'left':
                    this_face[:, :, :, :pad][mask] = interpo[mask]
                elif connect_side == 'right':
                    this_face[:, :, :, -pad:][mask] = interpo[mask]
            out.append(this_face)
        out = torch.cat(out, dim=0)
        [bs, c, h, w] = out.shape
        out = out.view(-1, bs//6, c, h, w).transpose(0,
                                                     1).contiguous().view(bs, c, h, w)
        return out
