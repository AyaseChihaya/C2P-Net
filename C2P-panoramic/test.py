import os
import sys

import cv2
import scipy
import torchvision
import yaml
import argparse

from scipy.ndimage import convolve, maximum_filter
from torchvision.transforms import Resize
from tqdm import tqdm
import numpy as np
import torch
import glob
import json
import LED2Net
import scipy.io as sio
from model import *
from network.FreDSNet_model import FDS
from network.NewCRFDepth import NewCRFDepth

ENCODER_RESNET = [
    'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
    'resnext50_32x4d', 'resnext101_32x8d'
]
ENCODER_DENSENET = [
    'densenet121', 'densenet169', 'densenet161', 'densenet201'
]
import torch
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as io
from models import modules, senet, net

unloader = torchvision.transforms.ToPILImage()





def define_model(is_senet,is_new):
    if is_senet:
        original_model = senet.senet154(pretrained='imagenet')
        Encoder = modules.E_senet(original_model)
        model = net.model(Encoder, num_features=2048, block_channel=[256, 512, 1024, 2048])
        return model
    if is_new:
        model = NewCRFDepth(version=args.encoder, inv_depth=False, max_depth=10, pretrained=args.pretrain)
        return model
    # if is_Fred:
    #     model = FDS(num_classes=18)
    #     return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training script for LED^2-Net',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='config/config_mp3d.yaml', help='config.yaml path')
    parser.add_argument('--src', type=str, default='/home/ps/data/Z/matterport_dataset/test/image/',
                        help='The folder that contain *.png or *.jpg')
    parser.add_argument('--backbone', default='resnet50',
                        choices=ENCODER_RESNET + ENCODER_DENSENET,
                        help='backbone of the network')
    parser.add_argument('--model_name', type=str, help='model name', default='newcrfs')
    parser.add_argument('--encoder', type=str, help='type of encoder, base07, large07', default='large07')
    parser.add_argument('--pretrain', type=str, help='path of pretrained encoder',
                        default='/home/ps/data/Z/pano(matterport)/pretrained_model/swin_transformer/swin_large_patch4_window7_224_22k.pth')  #
    parser.add_argument('--variance_focus', type=float,
                        help='lambda in paper: [0, 1], higher value more focus on minimizing variance of error',
                        default=0.85)
    args = parser.parse_args()


    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    equi_shape = config['exp_args']['visualizer_args']['equi_shape'] 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(0)
    model = define_model(is_senet=False, is_new=True)
    # model = torch.nn.DataParallel(model, device_ids=[0, 1]).cuda()
    model.to(device)
    checkpoint = torch.load('trained_model/n64(6-20-1weight).pth.tar')  # map_location={'cuda:0':'cuda:1'})
    model.load_state_dict(checkpoint['state_dict'], False)
    model.eval()
    src = args.src
    test_data = sorted(glob.glob(src + '/*.png') + glob.glob(src + '/*.jpg'))
    save_path = 'predict_param/predict_surface_param(n64(6-20-1weight))/'

    for i, one in enumerate(test_data):
        img = LED2Net.Dataset.SharedFunctions.read_image(one,
                                                         equi_shape)  
        image = torch.FloatTensor(img).permute(2, 0, 1)[None, ...].to(device)
        image = image.cuda()
        image = torch.autograd.Variable(image, volatile=True)
        torch_resize = Resize([128, 256])  
        image = torch_resize(image)

        #normalize
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        rgb = to_tensor(rgb)
        rgb = normalize(rgb)
        #     # rgb = rgb.permute(1, 2, 0)

        inv_output, output1, output2, output3 = model(image)

        image_flip = torch.flip(image, [3])
        inv_output_flip, output1_flip, output2_flip, output3_flip = model(image_flip)

        print(save_path + one[46:90] + '.mat')  # 34
        save_name = r'predict_param/predict_surface_param(n64(6-20-1weight))/' + one[46:90] + '.mat'
        io.savemat(save_name, {'inv_output_depth': inv_output.detach().data.cpu().numpy(),
                               'center_map': output1.detach().data.cpu().numpy(),
                               'segmentation': output2.detach().data.cpu().numpy(),
                               'weight': output3.detach().data.cpu().numpy(),
                               'inv_output_depth_flip': inv_output_flip.detach().data.cpu().numpy(),
                               'center_map_flip': output1_flip.detach().data.cpu().numpy(),
                               'segmentation_flip': output2_flip.detach().data.cpu().numpy(),
                               'weight_flip': output3_flip.detach().data.cpu().numpy()})




