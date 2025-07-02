import argparse
import torch
import torch.nn as nn
import torch.nn.parallel
from models import modules_large, net_large, resnet, senet
# import loaddata
import util
import numpy as np
import sobel
import layout_dataset_large as layout_dataset
# import layout_dataset_vgg as layout_dataset
# from normal2seg import *
# from bin_mean_shift import Bin_Mean_Shift
# from pykeops.torch import LazyTensor
import scipy.io as io
# from net_intrinsic_new import *
import time
# from vit_pytorch import ViT
import timm
import glob
import os
# from net_intrinsic_square import *
import torch.nn.functional as F

# torch.cuda.set_device(1)

# def define_model():
#
#     original_model = senet.senet154(pretrained='imagenet')
#     Encoder = modules_rc_new.E_senet(original_model)
#     model = net_rc_new.model(Encoder, num_features=2048, block_channel = [256, 512, 1024, 2048])
#
#     return model

def define_model():

    original_model = senet.senet154(pretrained='imagenet')
    Encoder = modules_large.E_senet(original_model)
    model = net_large.model(Encoder, num_features=2048, block_channel = [256, 512, 1024, 2048])

    return model


def main():

    model = define_model()
    # model = define_model(is_resnet=True, is_densenet=False, is_senet=False)
    model = torch.nn.DataParallel(model, device_ids=[0, 1]).cuda()
    # model = model.cuda()

    checkpoint = torch.load('trained_model/train_inpaint(1-17-1)).pth.tar')
    model.module.load_state_dict(checkpoint['state_dict'])
    # model.eval()


    test_loader = layout_dataset.getTestingData(1)
    test(test_loader, model, 0.25)


def test(test_loader, model, thre):
    model.eval()
    # model_s.eval()
    # model_intrinsic_new.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # x_range = torch.linspace(0, 1, 112).cuda()
    # y_range = torch.linspace(0, 1, 112).cuda()
    # smy, smx = torch.meshgrid(y_range, x_range)
    # smx = smx.expand([1, 1, -1, -1])
    # smy = smy.expand([1, 1, -1, -1])
    # coord_feat = torch.cat([smx, smy], 1)
    # coord_feat = torch.autograd.Variable(coord_feat)

    for i, sample_batched in enumerate(test_loader):
        image = sample_batched['image']

        # depth = depth.cuda()
        image = image.cuda()

        image = torch.autograd.Variable(image, volatile=True)
        # depth = torch.autograd.Variable(depth, volatile=True)


        torch.cuda.synchronize()
        start = time.time()

        output, output1, output2, output3 = model(image)
        # output, output1, output2, output3 = model(image, coord_feat)
        image_flip = torch.flip(image,[3])
        output_flip, output1_flip, output2_flip, output3_flip = model(image_flip)

        torch.cuda.synchronize()
        end = time.time()


        save_name = r'/home/ubuntu/work/regiongrow/predict_param/train_inpaint(1-17-1)/'+str(i)+'.mat'

        # io.savemat(save_name, {'output':output.detach().data.cpu().numpy(), 'output1':output1.detach().data.cpu().numpy(),
        #                        'output2':output2.detach().data.cpu().numpy(),'output3':output3.detach().data.cpu().numpy()})

        io.savemat(save_name, {'output':output.detach().data.cpu().numpy(), 'output1':output1.detach().data.cpu().numpy(), 'output2':output2.detach().data.cpu().numpy(), 'output3':output3.detach().data.cpu().numpy(), \
                               'output_flip':output_flip.detach().data.cpu().numpy(), 'output1_flip':output1_flip.detach().data.cpu().numpy(), 'output2_flip':output2_flip.detach().data.cpu().numpy(), 'output3_flip':output3_flip.detach().data.cpu().numpy()})



if __name__ == '__main__':
    main()
