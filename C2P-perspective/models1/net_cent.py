from collections import OrderedDict
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils import model_zoo
import copy
import numpy as np
# import modules
from torchvision import utils
# import senet
# import resnet
# import densenet
from models import modules_cent, resnet, densenet, senet


class model(nn.Module):
    def __init__(self, Encoder, num_features, block_channel):

        super(model, self).__init__()

        self.E = Encoder
        self.D = modules_cent.D(num_features)
        self.MFF = modules_cent.MFF(block_channel)
        self.R_PARAM = modules_cent.R_PARAM(block_channel)

        # self.C = modules_cent.C(block_channel)
        # self.R_LOC = modules_cent.R_LOC(block_channel)

        self.D_MASK = modules_cent.D_MASK(num_features)
        self.R_MASK = modules_cent.R_MASK(block_channel)

        self.D_PLANE = modules_cent.D_PLANE(num_features)
        self.R_PLANE = modules_cent.R_PLANE(block_channel)

    def forward(self, x, coord_feat):
        x_block1, x_block2, x_block3, x_block4 = self.E(x)
        x_decoder = self.D(x_block1, x_block2, x_block3, x_block4)
        x_mff = self.MFF(x_block1, x_block2, x_block3, x_block4,[x_decoder.size(2),x_decoder.size(3)])
        out_param = self.R_PARAM(torch.cat((x_decoder, x_mff, coord_feat), 1))
        # raw_inv_depth = (out_param[:,0:1,:,:] * cmx + out_param[:,1:2,:,:] * cmy + out_param[:,2:3,:,:]) * out_param[:,3:4,:,:]

        # input_feat = self.C(x_block4)
        # x_up = F.upsample(input_feat, scale_factor=2, mode='bilinear')
        # out_loc = self.R_LOC(torch.cat((x_up, coord_feat1), 1))

        x_decoder1 = self.D_MASK(x_block1, x_block2, x_block3, x_block4)
        out_mask = self.R_MASK(torch.cat((x_decoder1, x_mff), 1))

        x_decoder2 = self.D_PLANE(x_block1, x_block2, x_block3, x_block4)
        out_cent = self.R_PLANE(torch.cat((x_decoder2, x_mff, coord_feat), 1))

        return out_param, out_mask, out_cent

    # def forward(self, x, coord_feat):
    #     x_block1, x_block2, x_block3, x_block4 = self.E(x)
    #     x_decoder = self.D(x_block1, x_block2, x_block3, x_block4)
    #     x_mff = self.MFF(x_block1, x_block2, x_block3, x_block4,[x_decoder.size(2),x_decoder.size(3)])
    #     out_param = self.R_PARAM(torch.cat((x_decoder, x_mff, coord_feat), 1))
    #
    #     input_feat = self.C(x_block4)
    #     x_up = F.upsample(input_feat, scale_factor=2, mode='bilinear')
    #     out_loc = self.R_LOC(x_up)
    #
    #     x_decoder1 = self.D_MASK(x_block1, x_block2, x_block3, x_block4)
    #     out_mask = self.R_MASK(torch.cat((x_decoder1, x_mff), 1))
    #
    #     return out_param, out_loc, out_mask
