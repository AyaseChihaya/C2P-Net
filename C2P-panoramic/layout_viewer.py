import json
import numpy as np
import open3d as o3d
import torch
import torchvision
from PIL import Image
from scipy.signal import correlate2d
from scipy.ndimage import shift

from misc.post_proc import np_coor2xy, np_coorx2u, np_coory2v
from eval_general import layout_2_depth


import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

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

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--img', default='/home/ps/data/Z/Depth2Layout/picture/image/UwV83HsGsw3_a3630444bbd94cd6ac4edbe2dc1221de.png',
                        help='Image texture in equirectangular format')
    # parser.add_argument('--mask',
    #                     default='/home/ps/data/Z/Depth2Layout/picture/mini1/seg/1.png',
    #                     help='Image texture in equirectangular format')
    # parser.add_argument('--depth',
    #                     default='/home/ps/data/Z/Depth2Layout/picture/mini1/depth/1.png',
    #                     help='Image texture in equirectangular format')
    parser.add_argument('--layout', default='/home/ps/data/Z/label/UwV83HsGsw3_a3630444bbd94cd6ac4edbe2dc1221de.txt',
                        help='Txt or json file containing layout corners (cor_id)')
    parser.add_argument('--out')
    parser.add_argument('--vis', default='view/',action='store_true')
    parser.add_argument('--ignore_floor', action='store_true',
                        help='Skip rendering floor')
    parser.add_argument('--ignore_ceiling', action='store_true',default=True,
                        help='Skip rendering ceiling')
    parser.add_argument('--ignore_wall', action='store_true',
                        help='Skip rendering wall')
    parser.add_argument('--ignore_wireframe', action='store_true',default=True,
                        help='Skip rendering wireframe')
    args = parser.parse_args()

    if not args.out and not args.vis:
        print('You may want to export (via --out) or visualize (via --vis)')
        import sys; sys.exit()

    # Reading source (texture img, cor_id txt)
    equirect_texture = np.array(Image.open(args.img))

    cor_id = np.loadtxt(args.layout).astype(np.float32)

    # equirect_texture = torch.tensor(equirect_texture)
    # equirect_texture = equirect_texture.permute(2,0,1)
    # resize1 = torchvision.transforms.Resize((1024, 2048), interpolation=0)
    # equirect_texture = resize1(equirect_texture)
    # equirect_texture = equirect_texture.permute(1,2,0)
    # equirect_texture = np.array(equirect_texture)
    H, W = equirect_texture.shape[:2]
    depth, floor_mask, ceil_mask, wall_mask = layout_2_depth(cor_id, H, W, return_mask=True)

    # seg = np.array(Image.open(args.mask))
    # seg = torch.tensor(seg)
    # seg = torch.unsqueeze(seg, dim=0)
    # resize1 = torchvision.transforms.Resize((1024, 2048), interpolation=0)
    # seg = resize1(seg)
    # seg = seg.squeeze(0)
    # seg = np.array(seg)
    #
    #
    # depth = np.array(Image.open(args.depth))/4000
    # depth = torch.tensor(depth)
    # depth = torch.unsqueeze(depth, dim=0)
    # resize1 = torchvision.transforms.Resize((1024, 2048), interpolation=0)
    # depth = resize1(depth)
    # depth = depth.squeeze(0)
    # depth = np.array(depth)

    # floor_mask = (seg == 2)
    # ceil_mask = (seg == 1)
    # wall_mask = (~floor_mask) & (~ceil_mask)

    coorx, coory = np.meshgrid(np.arange(W), np.arange(H))
    us = np_coorx2u(coorx, W)
    vs = np_coory2v(coory, H)
    zs = depth * np.sin(vs)
    cs = depth * np.cos(vs)
    xs = cs * np.sin(us)
    ys = -cs * np.cos(us)

    # Aggregate mask
    mask = np.ones_like(floor_mask)
    if args.ignore_floor:
        mask &= ~floor_mask
    if args.ignore_ceiling:
        mask &= ~ceil_mask
    if args.ignore_wall:
        mask &= ~wall_mask

    # Prepare ply's points and faces
    xyzrgb = np.concatenate([
        xs[...,None], ys[...,None], zs[...,None],
        equirect_texture], -1)
    xyzrgb = np.concatenate([xyzrgb, xyzrgb[:,[0]]], 1)
    mask = np.concatenate([mask, mask[:,[0]]], 1)
    lo_tri_template = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 1, 1]])
    up_tri_template = np.array([
        [0, 0, 0],
        [0, 1, 1],
        [0, 0, 1]])
    ma_tri_template = np.array([
        [0, 0, 0],
        [0, 1, 1],
        [0, 1, 0]])
    lo_mask = (correlate2d(mask, lo_tri_template, mode='same') == 3)
    up_mask = (correlate2d(mask, up_tri_template, mode='same') == 3)
    ma_mask = (correlate2d(mask, ma_tri_template, mode='same') == 3) & (~lo_mask) & (~up_mask)
    ref_mask = (
        lo_mask | (correlate2d(lo_mask, np.flip(lo_tri_template, (0,1)), mode='same') > 0) |\
        up_mask | (correlate2d(up_mask, np.flip(up_tri_template, (0,1)), mode='same') > 0) |\
        ma_mask | (correlate2d(ma_mask, np.flip(ma_tri_template, (0,1)), mode='same') > 0)
    )
    points = xyzrgb[ref_mask]

    ref_id = np.full(ref_mask.shape, -1, np.int32)
    ref_id[ref_mask] = np.arange(ref_mask.sum())
    faces_lo_tri = np.stack([
        ref_id[lo_mask],
        ref_id[shift(lo_mask, [1, 0], cval=False, order=0)],
        ref_id[shift(lo_mask, [1, 1], cval=False, order=0)],
    ], 1)
    faces_up_tri = np.stack([
        ref_id[up_mask],
        ref_id[shift(up_mask, [1, 1], cval=False, order=0)],
        ref_id[shift(up_mask, [0, 1], cval=False, order=0)],
    ], 1)
    faces_ma_tri = np.stack([
        ref_id[ma_mask],
        ref_id[shift(ma_mask, [1, 0], cval=False, order=0)],
        ref_id[shift(ma_mask, [0, 1], cval=False, order=0)],
    ], 1)
    faces = np.concatenate([faces_lo_tri, faces_up_tri, faces_ma_tri])

    if args.vis:
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(points[:, :3])
        mesh.vertex_colors = o3d.utility.Vector3dVector(points[:, 3:] / 255.)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        draw_geometries = [mesh]
        o3d.visualization.draw_geometries(draw_geometries, mesh_show_back_face=True)

