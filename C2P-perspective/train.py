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
import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')
# os.environ['CUDA_VISIBLE_DEVICES']='1, 0'
torch.cuda.set_device(0)

# use_cuda = torch.cuda.is_available()
# dtype = 'float32' if use_cuda else 'float64'
# torchtype = {'float32': torch.float32, 'float64': torch.float64}


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
    # model = torch.nn.DataParallel(model, device_ids=[0, 1]).cuda()
    batch_size = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    model.train()

    cos = nn.CosineSimilarity(dim=1, eps=0)
    get_gradient = sobel.Sobel().cuda()
    # sobel_gradient = sobel.Sobel().cuda()
    get_param = grad.Grad().cuda()
    # get_cpt = grad_perimeter.Cpt().cuda()
    bce = nn.BCELoss(reduction='none')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    end = time.time()
    for i, sample_layout in enumerate(train_loader):
        lsun_image, lsun_seg, lsun_depth, lsun_raw_depth = sample_layout['lsun_image'], sample_layout['lsun_seg'], sample_layout['lsun_depth'], sample_layout['lsun_raw_depth']

        lsun_image = lsun_image.cuda()
        lsun_image = torch.autograd.Variable(lsun_image)
        lsun_seg = lsun_seg.cuda()
        lsun_seg = torch.autograd.Variable(lsun_seg)
        lsun_depth = lsun_depth.cuda()
        lsun_depth = torch.autograd.Variable(lsun_depth)

        inv_depth = 1 / (lsun_depth+1e-8)

        n_faces = 8

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
        output = 1/(inv_output + 1e-8)
        pred_valid = ((output > 0.1) & (output < 30)).float()

        mask_u = output2[:,0:grid_x,:,:]
        mask_v = output2[:,grid_x:grid_x+grid_y,:,:]

        mean_u = torch.sum((cmx.repeat(1,n_faces,1,1) * lsun_seg_mat).view(lsun_seg.size(0),n_faces,-1),2) \
                 / (torch.sum(lsun_seg_mat.view(lsun_seg.size(0),n_faces,-1),2)+1e-10)
        mean_v = torch.sum((cmy.repeat(1,n_faces,1,1) * lsun_seg_mat).view(lsun_seg.size(0),n_faces,-1),2) \
                 / (torch.sum(lsun_seg_mat.view(lsun_seg.size(0),n_faces,-1),2)+1e-10)

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
                tmp_mask_gt = (lsun_seg[fi][0] == tmp_label).float()
                tmp_loss_mask += dice_loss(tmp_mask, tmp_mask_gt) + bce(tmp_mask, tmp_mask_gt).mean()
                tmp_weight = tmp_mask * output3[fi][0] * pred_valid[fi][0]
                sorted, _ = torch.sort(torch.reshape(tmp_weight, [1, lsun_seg.size(2)*lsun_seg.size(3)]), descending=True)
                # thre = sorted[0, grid_x * grid_y * 8]
                # thre = 0.1
                thre = min(0.39, sorted[0, grid_x * grid_y * 8])
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

        loss_mask /= lsun_seg.size(0)

        loss_param = (torch.abs(param_map[:,0:1,:,:] - p_gt)).mean() + (torch.abs(param_map[:,1:2,:,:] - q_gt)).mean() \
                   + (torch.abs(param_map[:,2:3,:,:] - r_gt)).mean()

        gen_inv_depth = param_map[:,0:1,:,:] * cmx + param_map[:,1:2,:,:] * cmy + param_map[:,2:3,:,:]


        depth_2_gen = F.interpolate(inv_depth, scale_factor=0.5, mode='bilinear')
        depth_3_gen = F.interpolate(inv_depth, scale_factor=0.25, mode='bilinear')
        depth_4_gen = F.interpolate(inv_depth, scale_factor=0.125, mode='bilinear')

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
        loss_rc = (bce(torch.sigmoid(output1), gt_rc) * bce_weight).mean() + 0.2 * tv.ops.focal_loss.sigmoid_focal_loss(output1, gt_rc, alpha = 0.8).mean()
        #loss_wei = torch.relu(0.1 - output3.mean())

        loss = 10 * loss_rc + 1 * loss_mask + 20 * loss_param + 2 * loss_weighted_invd #+ 0.01 * loss_wei

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



if __name__ == '__main__':
    main()
