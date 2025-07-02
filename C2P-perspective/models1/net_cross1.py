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
from models import modules_cross1, resnet, densenet, senet


class model(nn.Module):
    def __init__(self, Encoder, num_features, block_channel):

        super(model, self).__init__()

        self.E = Encoder
        self.D = modules_cross1.D(num_features)
        # self.MFF = modules_cross.MFF(block_channel)
        # self.R = modules_cross.R(block_channel)
        # self.P = modules_cross.P()


    def forward(self, x):
        x_block1, x_block2, x_block3, x_block4 = self.E(x)
        out = self.D(x_block1, x_block2, x_block3, x_block4)
        # x_mff = self.MFF(x_block1, x_block2, x_block3, x_block4,[x_decoder.size(2),x_decoder.size(3)])
        # out = self.R(torch.cat((x_decoder, x_mff), 1))
        # param = self.P(x_block4)

        return out
