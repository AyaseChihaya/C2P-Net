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
from models1 import modules_large, resnet, densenet, senet


class model(nn.Module):
    def __init__(self, Encoder, num_features, block_channel):

        super(model, self).__init__()

        self.E = Encoder
        self.D = modules_large.D(num_features)
        self.MFF = modules_large.MFF(block_channel)
        self.R = modules_large.R(block_channel)

        self.C = modules_large.C(block_channel)
        self.R_LOC = modules_large.R_LOC(block_channel)

        self.D_MASK = modules_large.D_MASK(num_features)
        self.R_MASK = modules_large.R_MASK(block_channel)

        self.D_PLANE = modules_large.D_PLANE(num_features)
        self.R_PLANE = modules_large.R_PLANE(block_channel)

    def forward(self, x):
        x_block1, x_block2, x_block3, x_block4 = self.E(x)
        x_decoder = self.D(x_block1, x_block2, x_block3, x_block4)
        x_mff = self.MFF(x_block1, x_block2, x_block3, x_block4,[x_decoder.size(2),x_decoder.size(3)])
        out_depth = self.R(torch.cat((x_decoder, x_mff), 1))

        input_feat = self.C(x_block4)

        x_up = F.upsample(input_feat, scale_factor=2, mode='bilinear')
        out_loc = self.R_LOC(x_up)
        # print(x_block4.shape)
        # print(out_loc.shape)

        x_decoder1 = self.D_MASK(x_block1, x_block2, x_block3, x_block4)
        out_mask = self.R_MASK(torch.cat((x_decoder1, x_mff), 1))

        x_decoder2 = self.D_PLANE(x_block1, x_block2, x_block3, x_block4)
        out_plane = self.R_PLANE(torch.cat((x_decoder2, x_mff), 1))

        return out_depth, out_loc, out_mask, out_plane
