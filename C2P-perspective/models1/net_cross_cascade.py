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
from models import modules_cross_cascade, resnet, densenet, senet


class model(nn.Module):
    def __init__(self, Encoder, num_features, block_channel):

        super(model, self).__init__()

        self.E = Encoder
        self.D = modules_cross_cascade.D(num_features)
        self.MFF = modules_cross_cascade.MFF(block_channel)
        self.R = modules_cross_cascade.R(block_channel)
        self.R1 = modules_cross_cascade.R1(block_channel)
        self.R2 = modules_cross_cascade.R2(block_channel)
        # self.EMB = modules_cross_cascade.EMB()
        # self.DEP = modules_cross_cascade.DEP(block_channel)


    def forward(self, x, cmx, cmy):
        x_block1, x_block2, x_block3, x_block4 = self.E(x)
        x_decoder = self.D(x_block1, x_block2, x_block3, x_block4)
        # embedding = self.EMB(x_decoder)
        embedding = self.R2(x_decoder)
        x_mff = self.MFF(x_block1, x_block2, x_block3, x_block4,[x_decoder.size(2),x_decoder.size(3)])
        out = self.R(torch.cat((x_decoder, x_mff, embedding), 1))
        # embedding = self.R(torch.cat((x_decoder, x_mff), 1))
        # out = self.R1(torch.cat((x_decoder, x_mff, embedding), 1))
        pix_depth = (out[:,0:1,:,:] * cmx + out[:,1:2,:,:] * cmy + out[:,2:3,:,:]) * out[:,3:4,:,:]
        # ref_depth = self.DEP(torch.cat((x_decoder, x_mff, out, pix_depth), 1), pix_depth)
        ref_depth = self.R1(torch.cat((x_decoder, x_mff, out, pix_depth), 1))

        return out, embedding, ref_depth
