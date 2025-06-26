import os
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision
import torchvision as tv
import util
import numpy as np
import sobel
import grad
import grad_perimeter
from models1 import modules_large, net_large, senet
import random
from PIL import Image
import matplotlib.pyplot as plt
# import mp_dataset_pretrain as train_dataset
import mp_dataset_inpaint as train_dataset
# from loss import hinge_embedding_loss
import scipy.io as io
import itertools
import torch.nn.functional as F
# import loaddata_nyudepth as loaddata
from itertools import cycle
# from pytorch_lamb import Lamb
# import torch_optimizer as optim
# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')
# os.environ['CUDA_VISIBLE_DEVICES']='1, 0'
torch.cuda.set_device(1)

use_cuda = torch.cuda.is_available()
dtype = 'float32' if use_cuda else 'float64'
torchtype = {'float32': torch.float32, 'float64': torch.float64}


parser = argparse.ArgumentParser(description='PyTorch DenseNet Training')
parser.add_argument('--epochs', default=1000, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    help='weight decay (default: 1e-4)')


def define_model():

    original_model = senet.senet154(pretrained='imagenet')
    Encoder = modules_large.E_senet(original_model)
    model = net_large.model(Encoder, num_features=2048, block_channel = [256, 512, 1024, 2048])

    return model


def make_one_hot(labels, C=2):
    # one_hot = torch.FloatTensor(labels.size(0), C, labels.size(2), labels.size(3)).zero_()

    labels = labels.type(torch.cuda.LongTensor)
    one_hot = torch.cuda.FloatTensor(labels.size(0), C, labels.size(2), labels.size(3)).zero_()
    target = one_hot.scatter_(1, labels.data, 1)

    target = torch.autograd.Variable(target)

    return target

def si_loss(di, w):

    n_pix = di.size(2) * di.size(3)
    di2 = torch.pow(di, 2)
    first_term = torch.sum(di2,(1,2,3)) / n_pix
    second_term = w * torch.pow(torch.sum(di,(1,2,3)), 2) / (n_pix**2)
    loss = (first_term - second_term).mean()

    return loss


def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum()
    loss = (1. - ((2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)))

    return loss.mean()

def ms_grad_loss(output, gt, valid=None, trim_factor=0.8):

    cos = nn.CosineSimilarity(dim=1, eps=0)
    get_gradient = sobel.Sobel().cuda()
    ones = torch.ones(gt.size(0), 1, gt.size(2),gt.size(3),requires_grad=True).float().cuda()
    # ones = torch.autograd.Variable(ones)

    gt_grad = get_gradient(gt)
    output_grad = get_gradient(output)
    gt_grad_dx = gt_grad[:, 0, :, :].contiguous().view_as(gt)
    gt_grad_dy = gt_grad[:, 1, :, :].contiguous().view_as(gt)
    output_grad_dx = output_grad[:, 0, :, :].contiguous().view_as(gt)
    output_grad_dy = output_grad[:, 1, :, :].contiguous().view_as(gt)

    gt_normal = torch.cat((-gt_grad_dx, -gt_grad_dy, ones), 1)
    output_normal = torch.cat((-output_grad_dx, -output_grad_dy, ones), 1)

    if valid == None:

        dist_dx = torch.reshape(torch.abs(output_grad_dx - gt_grad_dx),(gt.size(0),-1))
        dist_dx_sorted, _ = torch.sort(dist_dx,1)
        loss_dx = dist_dx_sorted[:,:round(gt.size(2)*gt.size(3)*trim_factor)].mean()

        dist_dy = torch.reshape(torch.abs(output_grad_dy - gt_grad_dy),(gt.size(0),-1))
        dist_dy_sorted, _ = torch.sort(dist_dy,1)
        loss_dy = dist_dy_sorted[:,:round(gt.size(2)*gt.size(3)*trim_factor)].mean()

        dist_normal = torch.reshape(torch.abs(1 - cos(output_normal, gt_normal)),(gt.size(0),-1))
        dist_normal_sorted, _ = torch.sort(dist_normal,1)
        loss_normal = dist_normal_sorted[:,:round(gt.size(2)*gt.size(3)*trim_factor)].mean()

    else:
        dist_dx = torch.abs(output_grad_dx - gt_grad_dx)
        dist_dy = torch.abs(output_grad_dy - gt_grad_dy)
        dist_normal = torch.abs(1 - cos(output_normal, gt_normal))
        loss_dx = 0.
        loss_dy = 0.
        loss_normal = 0.
        for fi in range(gt.size(0)):
            tmp_valid = valid[fi,0]

            tmp_dist_dx = dist_dx[fi,0]
            tmp_dist_dx_sorted, _ = torch.sort(tmp_dist_dx[tmp_valid])
            loss_dx += tmp_dist_dx_sorted[:round(len(tmp_dist_dx_sorted)*trim_factor)].mean()

            tmp_dist_dy = dist_dy[fi,0]
            tmp_dist_dy_sorted, _ = torch.sort(tmp_dist_dy[tmp_valid])
            loss_dy += tmp_dist_dy_sorted[:round(len(tmp_dist_dy_sorted)*trim_factor)].mean()

            tmp_dist_normal = dist_normal[fi]
            tmp_dist_normal_sorted, _ = torch.sort(tmp_dist_normal[tmp_valid])
            loss_normal += tmp_dist_normal_sorted[:round(len(tmp_dist_normal_sorted)*trim_factor)].mean()
        loss_dx /= gt.size(0)
        loss_dy /= gt.size(0)
        loss_normal /= gt.size(0)

    return loss_dx + loss_dy + loss_normal


def trim_loss(output, gt, valid=None, trim_factor=0.8):

    if valid == None:

        output_reshape = torch.reshape(output,(gt.size(0),-1))
        gt_reshape = torch.reshape(gt,(gt.size(0),-1))

        t_output, _ = torch.median(output_reshape,1,keepdim=True)
        t_gt, _ = torch.median(gt_reshape,1,keepdim=True)

        s_output = torch.mean(torch.abs(output_reshape - t_output),1,keepdim=True)
        s_gt = torch.mean(torch.abs(gt_reshape - t_gt),1,keepdim=True)

        norm_output = (output_reshape - t_output) / (s_output + 1e-8)
        norm_gt = (gt_reshape - t_gt) / (s_gt + 1e-8)

        dist_ssi = torch.abs(norm_output - norm_gt)
        dist_ssi_sorted, _ = torch.sort(dist_ssi,1)
        loss_ssitrim = dist_ssi_sorted[:,:round(gt.size(2)*gt.size(3)*trim_factor)].mean()

    else:

        loss_ssitrim = 0.
        for fi in range(gt.size(0)):
            tmp_valid = valid[fi,0]
            tmp_output = output[fi,0]
            tmp_gt = gt[fi,0]
            output_reshape = tmp_output[tmp_valid]
            gt_reshape = tmp_gt[tmp_valid]

            t_output = torch.median(output_reshape)
            t_gt = torch.median(gt_reshape)

            s_output = torch.abs(output_reshape - t_output).mean()
            s_gt = torch.abs(gt_reshape - t_gt).mean()

            norm_output = (output_reshape - t_output) / (s_output + 1e-8)
            norm_gt = (gt_reshape - t_gt) / (s_gt + 1e-8)

            dist_ssi = torch.abs(norm_output - norm_gt)
            dist_ssi_sorted, _ = torch.sort(dist_ssi)
            loss_ssitrim += dist_ssi_sorted[:round(len(gt_reshape)*trim_factor)].mean()
        loss_ssitrim /= gt.size(0)

    return loss_ssitrim


def main():
    global args
    args = parser.parse_args()
    cudnn.benchmark = True

    model = define_model()
    # model = model.cuda()
    # model = torch.nn.DataParallel(model, device_ids=[0, 1]).cuda()
    batch_size = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # checkpoint = torch.load('trained_model/train_pretrain_v2.pth.tar')
    # model.load_state_dict(checkpoint['state_dict'])
    # model.module.load_state_dict(checkpoint['state_dict'])
    model.to(device)


    # optimizer = optim.AdamP(model.parameters(), args.lr)
    optimizer = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=args.weight_decay)
    # optimizer = Lamb(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay)
    # optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=0.5, weight_decay=args.weight_decay)


    train_loader = train_dataset.getLayout(batch_size)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        train(train_loader, model, optimizer, epoch)

        torch.save({
                    'state_dict': model.module.state_dict(),
                    }, 'trained_model/train_inpaint_v2.pth.tar')

def train(train_loader, model, optimizer, epoch):
    criterion = nn.L1Loss()
    batch_time = AverageMeter()
    losses = AverageMeter()
    # losses_intrinsic = AverageMeter()

    model.train()
    # model_intrinsic_new.train()

    cos = nn.CosineSimilarity(dim=1, eps=0)
    get_gradient = sobel.Sobel().cuda()
    # sobel_gradient = sobel.Sobel().cuda()
    get_param = grad.Grad().cuda()
    # get_cpt = grad_perimeter.Cpt().cuda()
    bce = nn.BCELoss(reduction='none')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # bin_mean_shift = Bin_Mean_Shift(device=device, bandwidth=0.5)

    end = time.time()
    for i, sample_layout in enumerate(train_loader):
        lsun_image, lsun_seg, lsun_depth, lsun_raw_depth = sample_layout['lsun_image'], sample_layout['lsun_seg'], sample_layout['lsun_depth'], sample_layout['lsun_raw_depth']

        lsun_image = lsun_image.cuda()
        lsun_image = torch.autograd.Variable(lsun_image)
        lsun_seg = lsun_seg.cuda()
        lsun_seg = torch.autograd.Variable(lsun_seg)
        lsun_depth = lsun_depth.cuda()
        lsun_depth = torch.autograd.Variable(lsun_depth)
        lsun_raw_depth = lsun_raw_depth.cuda()
        lsun_raw_depth = torch.autograd.Variable(lsun_raw_depth)

        # depth = lsun_depth
        inv_depth = 1 / (lsun_depth+1e-8)
        # inv_raw_depth = 1 / (lsun_raw_depth+1e-8)
        inv_raw_depth = 1 / (lsun_raw_depth + 1e-8)


        mask_plane_gt = torch.exp(-torch.pow((lsun_depth - lsun_raw_depth) * 50, 2) / 2)
        mask_valid = (lsun_raw_depth > 0.1)
        mask_invalid = (lsun_raw_depth <= 0.1)

        # dilate_temp = torch.ones(1, 1, 3, 3).cuda()
        # mask_invalid_dilate = torch.clamp(F.conv2d(mask_invalid.float(), dilate_temp, padding=(1, 1)), 0, 1)
        # mask_valid_dilate = (mask_invalid_dilate == 0)
#
        # inv_raw_depth[mask_invalid] = 0.

        # mask_boundary = mask_valid & (~mask_valid_dilate)
        # for fi in range(lsun_depth.size(0)):
        #     tmp_depth = lsun_raw_depth[fi,0]
        #     tmp_inv_depth = inv_raw_depth[fi,0]
        #     tmp_mask = mask_boundary[fi,0]
        #     tmp_invalid = mask_invalid[fi,0]
        #     # tmp_depth[tmp_invalid] = torch.median(tmp_depth[tmp_mask])
        #     tmp_inv_depth[tmp_invalid] = 1 / (torch.median(tmp_depth[tmp_mask])+1e-8)
        #     inv_raw_depth[fi,0] = tmp_inv_depth

        n_faces = 8

        ones = torch.ones(lsun_depth.size(0), 1, lsun_depth.size(2),lsun_depth.size(3),requires_grad=True).float().cuda()
        # ones = torch.autograd.Variable(ones)

        cmx = (torch.arange(lsun_depth.size(3))).repeat(lsun_depth.size(2), 1).float().cuda()
        cmx = torch.unsqueeze(torch.unsqueeze(cmx,0).repeat(lsun_depth.size(0),1,1),1)
        cmx = cmx / (lsun_depth.size(3)-1)
        cmx = torch.autograd.Variable(cmx)

        cmy = (torch.arange(lsun_depth.size(2))).repeat(lsun_depth.size(3), 1).float().cuda()
        cmy = cmy.permute(1,0)
        cmy = torch.unsqueeze(torch.unsqueeze(cmy,0).repeat(lsun_depth.size(0),1,1),1)
        cmy = cmy / (lsun_depth.size(2)-1)
        cmy = torch.autograd.Variable(cmy)

        uv1 = torch.ones(lsun_depth.size(2),lsun_depth.size(3),3,requires_grad=True).cuda()
        uv1[:,:,0]= cmx[0,0]
        uv1[:,:,1]= cmy[0,0]

        grid_x = 16
        grid_y = 16

        lsun_seg_mat = make_one_hot(lsun_seg, n_faces+1)
        lsun_seg_mat = lsun_seg_mat[:,1:,:,:]

        optimizer.zero_grad()

        inv_depth_grad = get_param(inv_depth)
        # output_grad = get_gradient(output)
        p_gt_raw = (inv_depth_grad[:, 0, :, :] * lsun_depth.size(3)).contiguous().view_as(lsun_depth)
        q_gt_raw = (inv_depth_grad[:, 1, :, :] * lsun_depth.size(2)).contiguous().view_as(lsun_depth)
        r_gt_raw = inv_depth - p_gt_raw * cmx - q_gt_raw * cmy

        mean_p_raw = torch.sum((p_gt_raw.repeat(1,n_faces,1,1) * lsun_seg_mat).view(lsun_seg.size(0),n_faces,-1),2) \
                 / (torch.sum(lsun_seg_mat.view(lsun_seg.size(0),n_faces,-1),2)+1e-10)
        mp_raw = torch.unsqueeze(torch.unsqueeze(mean_p_raw,2),3).repeat(1,1,lsun_depth.size(2),lsun_depth.size(3))
        p_gt = torch.sum(mp_raw * lsun_seg_mat, 1, keepdim=True)

        mean_q_raw = torch.sum((q_gt_raw.repeat(1,n_faces,1,1) * lsun_seg_mat).view(lsun_seg.size(0),n_faces,-1),2) \
                 / (torch.sum(lsun_seg_mat.view(lsun_seg.size(0),n_faces,-1),2)+1e-10)
        mq_raw = torch.unsqueeze(torch.unsqueeze(mean_q_raw,2),3).repeat(1,1,lsun_depth.size(2),lsun_depth.size(3))
        q_gt = torch.sum(mq_raw * lsun_seg_mat, 1, keepdim=True)

        mean_r_raw = torch.sum((r_gt_raw.repeat(1,n_faces,1,1) * lsun_seg_mat).view(lsun_seg.size(0),n_faces,-1),2) \
                 / (torch.sum(lsun_seg_mat.view(lsun_seg.size(0),n_faces,-1),2)+1e-10)
        mr_raw = torch.unsqueeze(torch.unsqueeze(mean_r_raw,2),3).repeat(1,1,lsun_depth.size(2),lsun_depth.size(3))
        r_gt = torch.sum(mr_raw * lsun_seg_mat, 1, keepdim=True)


        inv_output, output1, output2, output3 = model(lsun_image)
        inv_output = torch.relu(inv_output)
        # print(inv_output.shape)

        output = 1/(inv_output + 1e-8)
        pred_valid = ((output > 0.1) & (output < 30)).float()

        tf_raw = 0.8
        loss_ssitrim = trim_loss(inv_output, inv_raw_depth, trim_factor=tf_raw)

        loss_s1 = ms_grad_loss(inv_output, inv_raw_depth, trim_factor=tf_raw)

        output_2 = F.interpolate(inv_output, scale_factor=0.5, mode='bilinear')
        depth_2 = F.interpolate(inv_raw_depth, scale_factor=0.5, mode='bilinear')
        # valid_2 = F.interpolate(mask_valid_dilate.float(), scale_factor=0.5, mode='nearest') > 0
        loss_s2 = ms_grad_loss(output_2, depth_2, trim_factor=tf_raw)

        output_3 = F.interpolate(inv_output, scale_factor=0.25, mode='bilinear')
        depth_3 = F.interpolate(inv_raw_depth, scale_factor=0.25, mode='bilinear')
        # valid_3 = F.interpolate(mask_valid_dilate.float(), scale_factor=0.25, mode='nearest') > 0
        loss_s3 = ms_grad_loss(output_3, depth_3, trim_factor=tf_raw)

        output_4 = F.interpolate(inv_output, scale_factor=0.125, mode='bilinear')
        depth_4 = F.interpolate(inv_raw_depth, scale_factor=0.125, mode='bilinear')
        # valid_4 = F.interpolate(mask_valid_dilate.float(), scale_factor=0.125, mode='nearest') > 0
        loss_s4 = ms_grad_loss(output_4, depth_4, trim_factor=tf_raw)

        # loss_ssitrim = trim_loss(inv_output, inv_raw_depth, mask_invalid, trim_factor=0.8)
        #
        # loss_s1 = ms_grad_loss(inv_output, inv_raw_depth, mask_valid_dilate, trim_factor=0.8)
        #
        # output_2 = F.interpolate(inv_output, scale_factor=0.5, mode='bilinear')
        # depth_2 = F.interpolate(inv_raw_depth, scale_factor=0.5, mode='bilinear')
        # valid_2 = F.interpolate(mask_valid_dilate.float(), scale_factor=0.5, mode='nearest') > 0
        # loss_s2 = ms_grad_loss(output_2, depth_2, valid_2, trim_factor=0.8)
        #
        # output_3 = F.interpolate(inv_output, scale_factor=0.25, mode='bilinear')
        # depth_3 = F.interpolate(inv_raw_depth, scale_factor=0.25, mode='bilinear')
        # valid_3 = F.interpolate(mask_valid_dilate.float(), scale_factor=0.25, mode='nearest') > 0
        # loss_s3 = ms_grad_loss(output_3, depth_3, valid_3, trim_factor=0.8)
        #
        # output_4 = F.interpolate(inv_output, scale_factor=0.125, mode='bilinear')
        # depth_4 = F.interpolate(inv_raw_depth, scale_factor=0.125, mode='bilinear')
        # valid_4 = F.interpolate(mask_valid_dilate.float(), scale_factor=0.125, mode='nearest') > 0
        # loss_s4 = ms_grad_loss(output_4, depth_4, valid_4, trim_factor=0.8)

        # loss_cor = torch.log(torch.abs(output[:,:,118:127,118:127]-lsun_raw_depth[:,:,118:127,118:127])*mask_valid[:,:,118:127,118:127]+0.5).mean()
        # loss_depth = torch.log(torch.abs(output - lsun_raw_depth) + 0.5).mean()
        loss_depth = torch.log(torch.abs(output[:,:,118:127,118:127] - lsun_raw_depth[:,:,118:127,118:127]) + 0.5).mean()
        # print(loss_depth)
        # print(loss_s2)
        # print(loss_s3)
        # print(loss_s4)

        loss_raw_depth = loss_ssitrim + loss_s1 + loss_s2 + loss_s3 + loss_s4 + loss_depth



        mask_u = output2[:,0:grid_x,:,:]
        mask_v = output2[:,grid_x:grid_x+grid_y,:,:]

        mean_u = torch.sum((cmx.repeat(1,n_faces,1,1) * lsun_seg_mat).view(lsun_seg.size(0),n_faces,-1),2) \
                 / (torch.sum(lsun_seg_mat.view(lsun_seg.size(0),n_faces,-1),2)+1e-10)
        mean_v = torch.sum((cmy.repeat(1,n_faces,1,1) * lsun_seg_mat).view(lsun_seg.size(0),n_faces,-1),2) \
                 / (torch.sum(lsun_seg_mat.view(lsun_seg.size(0),n_faces,-1),2)+1e-10)

        # mean_u = torch.sum((cmx.repeat(1,n_faces,1,1) * lsun_seg_mat).view(lsun_seg.size(0),n_faces,-1),2) \
        #          / (torch.sum(lsun_seg_mat.view(lsun_seg.size(0),n_faces,-1),2)+1e-10)
        # mean_u_map = torch.unsqueeze(torch.unsqueeze(mean_u,2),3).repeat(1,1,lsun_depth.size(2),lsun_depth.size(3))
        # mean_u_map = torch.sum(mean_u_map * lsun_seg_mat, 1, keepdim=True)
        #
        # mean_v = torch.sum((cmy.repeat(1,n_faces,1,1) * lsun_seg_mat).view(lsun_seg.size(0),n_faces,-1),2) \
        #          / (torch.sum(lsun_seg_mat.view(lsun_seg.size(0),n_faces,-1),2)+1e-10)
        # mean_v_map = torch.unsqueeze(torch.unsqueeze(mean_v,2),3).repeat(1,1,lsun_depth.size(2),lsun_depth.size(3))
        # mean_v_map = torch.sum(mean_v_map * lsun_seg_mat, 1, keepdim=True)
        #
        # loc_weight = (torch.abs(cmx - mean_u_map) + torch.abs(cmy - mean_v_map)) * (lsun_depth.size(2)-1)
        # print(loc_weight[0,0])
        # loc_weight[loc_weight>=1] = 0
        # print(loc_weight[loc_weight>0])
        # print(loc_weight[0,0])
        # loc_weight = torch.exp(-(torch.pow((cmx - mean_u_map), 2) + torch.pow((cmy - mean_v_map), 2)) * 10)
        # print(loc_weight[0][0])
        # print(cmx[0][0])

        invalid_uv = (mean_u==0).float() * (mean_v==0).float()
        gt_rc = torch.zeros(lsun_depth.size(0), 1, grid_y, grid_x).float().cuda()
        # gt_param = torch.zeros(lsun_depth.size(0), 4, grid_y, grid_x).float().cuda()
        param_map = torch.zeros(lsun_seg.size(0), 3, lsun_seg.size(2),lsun_seg.size(3)).float().cuda()
        # fp_gt_param = torch.zeros(lsun_depth.size(0), 4, grid_y, grid_x).float().cuda()
        # output_cp = output.detach()
        output1_cp = torch.sigmoid(output1.detach())
        loss_mask = 0.
        loss_fp = 0.
        loss_gen = 0.
        weighted_invd = torch.zeros(lsun_seg.size(0), 1, lsun_seg.size(2),lsun_seg.size(3)).float().cuda()
        for fi in range(lsun_seg.size(0)):
            tmp_inv_output = inv_output[fi,0]
            tmp_invalid = invalid_uv[fi:fi+1]
            index_u = mean_u[fi:fi+1]
            index_u = index_u[tmp_invalid!=1]
            index_u_grid = torch.round(index_u * (grid_x-1)).long()
            index_u_image = torch.round(index_u * (lsun_seg.size(3)-1)).long()
            index_v = mean_v[fi:fi+1]
            index_v = index_v[tmp_invalid!=1]
            index_v_grid = torch.round(index_v * (grid_y-1)).long()
            index_v_image = torch.round(index_v * (lsun_seg.size(2)-1)).long()
            gt_rc[fi,:,index_v_grid,index_u_grid] = 1
            # tmp_gt_rc = gt_rc[fi][0]
            # tmp_plane_gt = mask_plane_gt[fi][0]
            tmp_loss_mask = 0.
            # tmp_loss_param = 0.
            tmp_param_map = torch.zeros(3, lsun_seg.size(2),lsun_seg.size(3)).float().cuda()
            all_mask = torch.zeros(index_v_grid.size(0), lsun_seg.size(2),lsun_seg.size(3)).float().cuda()
            all_invd = torch.zeros(index_v_grid.size(0), lsun_seg.size(2),lsun_seg.size(3)).float().cuda()
            for gi in range(index_v_grid.size(0)):
                tmp_mask = mask_u[fi,index_u_grid[gi]] * mask_v[fi,index_v_grid[gi]]
                tmp_label = lsun_seg[fi,0,index_v_image[gi],index_u_image[gi]]
                # tmp_mask_gt = tmp_plane_gt * (lsun_seg[fi][0] == tmp_label).float()
                tmp_mask_gt = (lsun_seg[fi][0] == tmp_label).float()
                tmp_loss_mask += dice_loss(tmp_mask, tmp_mask_gt) + bce(tmp_mask, tmp_mask_gt).mean()
                # tmp_mask_gt = (lsun_seg[fi][0] == tmp_label).float()
                tmp_weight = tmp_mask * output3[fi][0] * pred_valid[fi][0]
                sorted, _ = torch.sort(torch.reshape(tmp_weight, [1, lsun_seg.size(2)*lsun_seg.size(3)]), descending=True)
                # thre = sorted[0, grid_x * grid_y * 8]
                # thre = 0.1
                thre = min(0.7, sorted[0, grid_x * grid_y * 8])
                x = uv1[tmp_weight > thre]
                w = torch.diag(tmp_weight[tmp_weight > thre])
                y = torch.unsqueeze(tmp_inv_output[tmp_weight > thre],1)
                tmp_param = torch.mm(torch.mm(torch.mm(torch.linalg.inv(torch.mm(torch.mm(torch.transpose(x,0,1), w), x)), torch.transpose(x,0,1)), w), y)
                tmp_param = torch.unsqueeze(tmp_param,1).repeat(1,lsun_seg.size(2),lsun_seg.size(3))
                tmp_param_map += tmp_param * torch.unsqueeze(tmp_mask_gt,0).repeat(3,1,1)
                all_mask[gi] = tmp_mask
                all_invd[gi] = tmp_param[0] * cmx[0,0] + tmp_param[1] * cmy[0,0] + tmp_param[2]

            weighted_invd[fi,0] = (all_mask * all_invd).mean(0) / all_mask.mean(0)

            tmp_loss_mask /= (index_v_grid.size(0) + 1e-10)
            loss_mask += tmp_loss_mask
            param_map[fi] = tmp_param_map

            # fp_rc = output1_cp[fi,0] - gt_rc[fi,0]
            # fp_v, fp_u = (fp_rc>0.3).nonzero(as_tuple=True)
            # fp_u_image = torch.round(fp_u / (grid_x-1) * (lsun_seg.size(3)-1)).long()
            # fp_v_image = torch.round(fp_v / (grid_y-1) * (lsun_seg.size(2)-1)).long()
            # tmp_fp_loss_mask = 0.
            # for ki in range(fp_u.size(0)):
            #     fp_mask = mask_u[fi,fp_u[ki]] * mask_v[fi,fp_v[ki]]
            #     fp_label = lsun_seg[fi,0,fp_v[ki],fp_u[ki]]
            #     fp_mask_gt = (lsun_seg[fi][0] == fp_label).float()
            #     tmp_fp_loss_mask += (dice_loss(fp_mask, fp_mask_gt) + bce(fp_mask, fp_mask_gt).mean()) * fp_rc[fp_v[ki],fp_u[ki]]
            # tmp_fp_loss_mask /= (fp_u.size(0) + 1e-10)
            # loss_fp += tmp_fp_loss_mask

        loss_mask /= lsun_seg.size(0)
        # loss_fp /= lsun_seg.size(0)


        loss_param = (torch.abs(param_map[:,0:1,:,:] - p_gt)).mean() + (torch.abs(param_map[:,1:2,:,:] - q_gt)).mean() \
                   + (torch.abs(param_map[:,2:3,:,:] - r_gt)).mean()

        gen_inv_depth = param_map[:,0:1,:,:] * cmx + param_map[:,1:2,:,:] * cmy + param_map[:,2:3,:,:]

        # tf_gen = 1
        # loss_ssitrim_gen = trim_loss(gen_inv_depth, inv_depth, trim_factor=tf_gen)
        # loss_s1_gen = ms_grad_loss(gen_inv_depth, inv_depth, trim_factor=tf_gen)
        # output_2_gen = F.interpolate(gen_inv_depth, scale_factor=0.5, mode='bilinear')
        depth_2_gen = F.interpolate(inv_depth, scale_factor=0.5, mode='bilinear')
        # loss_s2_gen = ms_grad_loss(output_2_gen, depth_2_gen, trim_factor=tf_gen)
        # output_3_gen = F.interpolate(gen_inv_depth, scale_factor=0.25, mode='bilinear')
        depth_3_gen = F.interpolate(inv_depth, scale_factor=0.25, mode='bilinear')
        # loss_s3_gen = ms_grad_loss(output_3_gen, depth_3_gen, trim_factor=tf_gen)
        # output_4_gen = F.interpolate(gen_inv_depth, scale_factor=0.125, mode='bilinear')
        depth_4_gen = F.interpolate(inv_depth, scale_factor=0.125, mode='bilinear')
        # loss_s4_gen = ms_grad_loss(output_4_gen, depth_4_gen, trim_factor=tf_gen)

        # loss_gen_invd = loss_ssitrim_gen + loss_s1_gen + loss_s2_gen + loss_s3_gen + loss_s4_gen

        tf_wei = 1
        loss_ssitrim_w = trim_loss(weighted_invd, inv_depth, trim_factor=tf_wei)
        loss_s1_w = ms_grad_loss(weighted_invd, inv_depth, trim_factor=tf_wei)
        output_2_w = F.interpolate(weighted_invd, scale_factor=0.5, mode='bilinear')
        loss_s2_w = ms_grad_loss(output_2_w, depth_2_gen, trim_factor=tf_wei)
        output_3_w = F.interpolate(weighted_invd, scale_factor=0.25, mode='bilinear')
        loss_s3_w = ms_grad_loss(output_3_w, depth_3_gen, trim_factor=tf_wei)
        output_4_w = F.interpolate(weighted_invd, scale_factor=0.125, mode='bilinear')
        loss_s4_w = ms_grad_loss(output_4_w, depth_4_gen, trim_factor=tf_wei)

        loss_weighted_invd = loss_ssitrim_w + loss_s1_w + loss_s2_w + loss_s3_w + loss_s4_w


        bce_weight = gt_rc * 24 + 1
        # output_rc = torch.sigmoid(output1)
        # loss_rc = (bce(output_rc, gt_rc) * (gt_rc * 24 + 1)).mean()
        # gt_rc_s1 = torch.clamp(F.interpolate(gt_rc, scale_factor=0.5, mode='bilinear') * 4, 0, 1)
        # output_rc_s1 = torch.clamp(F.interpolate(output_rc, scale_factor=0.5, mode='bilinear') * 4, 0, 1)
        # loss_rc_s1 = (bce(output_rc_s1, gt_rc_s1) * (gt_rc_s1 * 4 + 1)).mean()
        # gt_rc_s2 = torch.clamp(F.interpolate(gt_rc_s1, scale_factor=0.5, mode='bilinear') * 4, 0, 1)
        # output_rc_s2 = torch.clamp(F.interpolate(output_rc_s1, scale_factor=0.5, mode='bilinear') * 4, 0, 1)
        # loss_rc_s2 = bce(output_rc_s2, gt_rc_s2).mean()
        #
        # loss_cent = loss_rc + 0.5 * loss_rc_s1 + 0.25 * loss_rc_s2


        # loss_rc = tv.ops.focal_loss.sigmoid_focal_loss(output1, gt_rc, alpha = 0.5).mean()
        loss_rc = (bce(torch.sigmoid(output1), gt_rc) * bce_weight).mean() + 0.2 * tv.ops.focal_loss.sigmoid_focal_loss(output1, gt_rc, alpha = 0.8).mean()
        # loss_rc = (bce(torch.sigmoid(output1), gt_rc) * bce_weight).mean()  + 0.2 * tv.ops.focal_loss.sigmoid_focal_loss(output1, gt_rc, alpha = 0.8).mean()
        # loss_rc = 0 * tv.ops.focal_loss.sigmoid_focal_loss(output1, gt_rc, alpha = 0.25).mean() + (bce(torch.sigmoid(output1), gt_rc) * bce_weight).mean()
        loss_wei = torch.relu(dice_loss(output3, mask_plane_gt) - 0.1)
        # loss_wei = torch.relu(0.1 - output3.mean())


        loss = 10 * loss_rc + 2 * loss_raw_depth + 1 * loss_mask + 20 * loss_param + 2 * loss_weighted_invd + 1 * loss_wei


        # train_inpaint_v2.pth.tar initialization:train_pretrain_v2 folder:train_large_v10 9.56 2.64 3.04
        # loss = 10 * loss_rc + 2 * loss_raw_depth + 1 * loss_mask + 20 * loss_param + 2 * loss_weighted_invd + 0.01 * loss_wei


        # train_large_v4 10.53 2.96 3.48
        # loss = 10 * loss_rc + 2 * loss_raw_depth + 1 * loss_mask + 0.01 * loss_fp + 20 * loss_param + 1 * loss_gen_invd \
        #     + 1 * loss_weighted_invd

        # train_large_v3 10.57 2.96 3.47 threshold 0.3 0.1
        # loss = 10 * loss_cent + 2 * loss_raw_depth + 1 * loss_mask + 0.01 * loss_fp + 20 * loss_param + 1 * loss_gen_invd \
            # + 1 * loss_weighted_invd

        # train_large_v2 10.60 10.60 2.97 3.50
        # loss = 10 * loss_rc + 2 * loss_raw_depth + 1 * loss_mask + 0.01 * loss_fp + 30 * loss_param + 1 * loss_gen_invd \
        #     + 1 * loss_weighted_invd

        # train_large_v2_320 flip 11.05 3.12 3.63

        # train_large_v1 256 11.05 3.30 3.99  flip 10.71 3.08 3.53
        # loss = 10 * loss_rc + 1 * loss_raw_depth + 1 * loss_mask + 0.01 * loss_fp + 10 * loss_param + 1 * loss_gen_invd \
        #     + 1 * loss_weighted_invd

        # train_rc_v6_2_6  12.88 3.30 4.00
        # loss = 10 * loss_rc + 1 * loss_raw_depth + 1 * loss_mask + 0.01 * loss_fp + 0 * loss_plane + 10 * loss_param + 1 * loss_gen_invd \
        #      + 1 * loss_weighted_invd

        # train_rc_v6_2_6 13.15 3.41 4.09
        # loss = 10 * loss_rc + 1 * loss_raw_depth + 1 * loss_mask + 0.03 * loss_fp + 1 * loss_plane + 10 * loss_param + 1 * loss_gen_invd \
        #      + 1 * loss_weighted_invd

        # train_rc_v6_2_4 13.76 3.34 4.17 gen_depth weighted_depth log+grad+normal
        # loss = 10 * loss_rc + 3 * loss_raw_depth + 10 * loss_mask + 0.03 * loss_fp + 1 * loss_plane + 30 * loss_param + 3 * loss_depth \
        #      + 3 * loss_weighted_depth

        # train_rc_v6_2_3 13.55 3.43 4.20
        # loss = 10 * loss_rc + 3 * loss_raw_depth + 1 * loss_mask + 0.03 * loss_fp + 1 * loss_plane + 30 * loss_param + 100 * loss_inv_depth \
        #      + 100 * loss_gen

        # train_rc_v6_2_2 13.86 3.47 4.38
        # loss = 10 * loss_rc + 3 * loss_raw_depth + 1 * loss_mask + 0.03 * loss_fp + 1 * loss_plane + 10 * loss_param + 10 * loss_inv_depth \
        #      + 10 * loss_gen

        # model_rc_new_v5 15.62 3.72 4.66
        # loss = 100 * loss_rc + 10 * loss_raw_depth + 1 * loss_normal_pqr + 1 * loss_mask \
        #      + 1 * loss_planar_l1 + 10 * loss_param + 100 * loss_inv_depth + 0.03 * loss_fp + 0.01 * loss_plane + 300 * loss_gen

        # model_rc_new_v4 14.71 3.76 4.77
        # loss = 10 * loss_rc + 10 * loss_raw_depth + 1 * loss_normal_pqr + 1 * loss_mask \
        #      + 1 * loss_planar_l1 + 10 * loss_param + 100 * loss_inv_depth + 0.03 * loss_fp + 0.01 * loss_plane + 300 * loss_gen



        losses.update(loss.item(), lsun_image.size(0))
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 1.)
        optimizer.step()


        batch_time.update(time.time() - end)
        end = time.time()

        batchSize = lsun_depth.size(0)

        print('Epoch: [{0}][{1}/{2}]\t'
          'Time {batch_time.val:.3f} ({batch_time.sum:.3f})\t'
          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
          .format(epoch, i, len(train_loader), batch_time=batch_time, loss=losses))


def adjust_learning_rate(optimizer, epoch):
    lr = args.lr * (0.2 ** (epoch // 70))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
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
    main()
