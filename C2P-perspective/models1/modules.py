from collections import OrderedDict
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils import model_zoo
import copy
import numpy as np
# import senet
# import resnet
# import densenet
from models import resnet, densenet, senet

class _UpProjection(nn.Sequential):

    def __init__(self, num_input_features, num_output_features):
        super(_UpProjection, self).__init__()

        self.conv1 = nn.Conv2d(num_input_features, num_output_features,
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(num_output_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(num_output_features, num_output_features,
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_2 = nn.BatchNorm2d(num_output_features)

        self.conv2 = nn.Conv2d(num_input_features, num_output_features,
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(num_output_features)

    def forward(self, x, size):
        x = F.upsample(x, size=size, mode='bilinear')
        x_conv1 = self.relu(self.bn1(self.conv1(x)))
        bran1 = self.bn1_2(self.conv1_2(x_conv1))
        bran2 = self.bn2(self.conv2(x))

        out = self.relu(bran1 + bran2)

        return out

class E_resnet(nn.Module):

    def __init__(self, original_model, num_features = 2048):
        super(E_resnet, self).__init__()
        self.conv1 = original_model.conv1
        self.bn1 = original_model.bn1
        self.relu = original_model.relu
        self.maxpool = original_model.maxpool

        self.layer1 = original_model.layer1
        self.layer2 = original_model.layer2
        self.layer3 = original_model.layer3
        self.layer4 = original_model.layer4


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x_block1 = self.layer1(x)
        x_block2 = self.layer2(x_block1)
        x_block3 = self.layer3(x_block2)
        x_block4 = self.layer4(x_block3)

        return x_block1, x_block2, x_block3, x_block4

class E_densenet(nn.Module):

    def __init__(self, original_model, num_features = 2208):
        super(E_densenet, self).__init__()
        self.features = original_model.features

    def forward(self, x):
        x01 = self.features[0](x)
        x02 = self.features[1](x01)
        x03 = self.features[2](x02)
        x04 = self.features[3](x03)

        x_block1 = self.features[4](x04)
        x_block1 = self.features[5][0](x_block1)
        x_block1 = self.features[5][1](x_block1)
        x_block1 = self.features[5][2](x_block1)
        x_tran1 = self.features[5][3](x_block1)

        x_block2 = self.features[6](x_tran1)
        x_block2 = self.features[7][0](x_block2)
        x_block2 = self.features[7][1](x_block2)
        x_block2 = self.features[7][2](x_block2)
        x_tran2 = self.features[7][3](x_block2)

        x_block3 = self.features[8](x_tran2)
        x_block3 = self.features[9][0](x_block3)
        x_block3 = self.features[9][1](x_block3)
        x_block3 = self.features[9][2](x_block3)
        x_tran3 = self.features[9][3](x_block3)

        x_block4 = self.features[10](x_tran3)
        x_block4 = F.relu(self.features[11](x_block4))

        return x_block1, x_block2, x_block3, x_block4

class E_senet(nn.Module):

    def __init__(self, original_model, num_features = 2048):
        super(E_senet, self).__init__()
        self.base = nn.Sequential(*list(original_model.children())[:-3])

    def forward(self, x):
        x = self.base[0](x)
        x_block1 = self.base[1](x)
        x_block2 = self.base[2](x_block1)
        x_block3 = self.base[3](x_block2)
        x_block4 = self.base[4](x_block3)

        return x_block1, x_block2, x_block3, x_block4

class D(nn.Module):

    def __init__(self, num_features = 2048):
        super(D, self).__init__()
        self.conv = nn.Conv2d(num_features, num_features //
                               2, kernel_size=1, stride=1, bias=False)
        num_features = num_features // 2
        self.bn = nn.BatchNorm2d(num_features)

        self.up1 = _UpProjection(
            num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2

        self.up2 = _UpProjection(
            num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2

        self.up3 = _UpProjection(
            num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2

        self.up4 = _UpProjection(
            num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2


    def forward(self, x_block1, x_block2, x_block3, x_block4):
        x_d0 = F.relu(self.bn(self.conv(x_block4)))
        x_d1 = self.up1(x_d0, [x_block3.size(2), x_block3.size(3)])
        x_d2 = self.up2(x_d1, [x_block2.size(2), x_block2.size(3)])
        x_d3 = self.up3(x_d2, [x_block1.size(2), x_block1.size(3)])
        x_d4 = self.up4(x_d3, [x_block1.size(2)*2, x_block1.size(3)*2])

        return x_d4

class MFF(nn.Module):

    def __init__(self, block_channel, num_features=64):

        super(MFF, self).__init__()

        self.up1 = _UpProjection(
            num_input_features=block_channel[0], num_output_features=16)

        self.up2 = _UpProjection(
            num_input_features=block_channel[1], num_output_features=16)

        self.up3 = _UpProjection(
            num_input_features=block_channel[2], num_output_features=16)

        self.up4 = _UpProjection(
            num_input_features=block_channel[3], num_output_features=16)

        self.conv = nn.Conv2d(
            num_features, num_features, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn = nn.BatchNorm2d(num_features)


    def forward(self, x_block1, x_block2, x_block3, x_block4, size):
        x_m1 = self.up1(x_block1, size)
        x_m2 = self.up2(x_block2, size)
        x_m3 = self.up3(x_block3, size)
        x_m4 = self.up4(x_block4, size)

        x = self.bn(self.conv(torch.cat((x_m1, x_m2, x_m3, x_m4), 1)))
        x = F.relu(x)

        return x


class R(nn.Module):
    def __init__(self, block_channel):

        super(R, self).__init__()

        num_features = 64 + block_channel[3]//32
        self.conv0 = nn.Conv2d(num_features, num_features,
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn0 = nn.BatchNorm2d(num_features)

        self.conv1 = nn.Conv2d(num_features, num_features,
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features)

        self.conv2 = nn.Conv2d(
            num_features, 3, kernel_size=5, stride=1, padding=2, bias=True)

    def forward(self, x):
        x0 = self.conv0(x)
        x0 = self.bn0(x0)
        x0 = F.relu(x0)

        x1 = self.conv1(x0)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)

        x2 = self.conv2(x1)

        return x2


class R_plane(nn.Module):
    def __init__(self, block_channel):

        super(R_plane, self).__init__()

        # num_features = 64 + block_channel[3]//32
        num_features = 64
        self.conv0 = nn.Conv2d(num_features, num_features,
                               kernel_size=5, stride=1, padding=2, bias=True)
        self.bn0 = nn.BatchNorm2d(num_features)

        self.conv1 = nn.Conv2d(num_features, num_features,
                               kernel_size=5, stride=1, padding=2, bias=True)
        self.bn1 = nn.BatchNorm2d(num_features)

        self.conv2 = nn.Conv2d(
            num_features, 3, kernel_size=5, stride=1, padding=2, bias=True)

        self.conv0_m = nn.Conv2d(num_features, num_features,
                               kernel_size=5, stride=1, padding=2, bias=True)
        self.bn0_m = nn.BatchNorm2d(num_features)

        self.conv1_m = nn.Conv2d(num_features, num_features,
                               kernel_size=5, stride=1, padding=2, bias=True)
        self.bn1_m = nn.BatchNorm2d(num_features)

        self.conv2_m = nn.Conv2d(
            num_features, 6, kernel_size=5, stride=1, padding=2, bias=True)



    def forward(self, x, m):

        x0 = self.conv0(x)
        x0 = self.bn0(x0)
        x0 = F.relu(x0)

        x1 = self.conv1(x0)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)

        x2 = self.conv2(x1)


        m0 = self.conv0_m(m)
        m0 = self.bn0_m(m0)
        m0 = F.relu(m0)

        m1 = self.conv1_m(m0)
        m1 = self.bn1_m(m1)
        m1 = F.relu(m1)

        m2 = self.conv2_m(m1)
        m2 = F.sigmoid(m2)


        return x2, m2




class REG(nn.Module):
    def __init__(self, block_channel):

        super(REG, self).__init__()

        num_features = 64
        self.conv0 = nn.Conv2d(3, 64,
                               kernel_size=5, stride=2, padding=2, bias=True)
        self.bn0 = nn.BatchNorm2d(64)

        self.conv1 = nn.Conv2d(64, 128,
                               kernel_size=5, stride=2, padding=2, bias=True)
        self.bn1 = nn.BatchNorm2d(128)

        self.conv2 = nn.Conv2d(128, 256,
                               kernel_size=3, stride=2, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(256)

        self.conv3 = nn.Conv2d(256, 512,
                               kernel_size=3, stride=2, padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(512)

        self.conv4 = nn.Conv2d(512, 512,
                               kernel_size=3, stride=2, padding=1, bias=True)
        self.bn4 = nn.BatchNorm2d(512)

        self.conv5 = nn.Conv2d(512, 512,
                               kernel_size=3, stride=2, padding=1, bias=True)
        self.bn5 = nn.BatchNorm2d(512)

        self.conv6 = nn.Conv2d(512, 1024,
                               kernel_size=4, stride=1, padding=0, bias=True)
        # self.bn6 = nn.BatchNorm2d(1024)

        self.conv7 = nn.Conv2d(1024, 512,
                               kernel_size=1, stride=1, padding=0, bias=True)

        self.conv8 = nn.Conv2d(512, 27,
                               kernel_size=1, stride=1, padding=0, bias=True)


    def forward(self, x):
        x0 = self.conv0(x)
        x0 = self.bn0(x0)
        x0 = F.relu(x0)

        x1 = self.conv1(x0)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)

        x2 = self.conv2(x1)
        x2 = self.bn2(x2)
        x2 = F.relu(x2)

        x3 = self.conv3(x2)
        x3 = self.bn3(x3)
        x3 = F.relu(x3)

        x4 = self.conv4(x3)
        x4 = self.bn4(x4)
        x4 = F.relu(x4)

        x5 = self.conv5(x4)
        x5 = self.bn5(x5)
        x5 = F.relu(x5)

        x6 = self.conv6(x5)
        x6 = F.relu(x6)

        x7 = self.conv7(x6)
        x7 = F.relu(x7)

        x8 = self.conv8(x7)


        # x3 = self.conv3(x2)
        # x3 = F.relu(x3)
        #
        # x4 = self.conv4(x3)

        return x8



class GP(nn.Module):
    def __init__(self, block_channel):

        super(GP, self).__init__()


        self.meanpool = nn.AvgPool2d((8, 10))

        self.conv0 = nn.Conv2d(2048, 27,
                               kernel_size=1, stride=1, padding=0, bias=True)


    def forward(self, x):
        x0 = self.meanpool(x)
        x0 = self.conv0(x0)

        return x0


class R_decompose(nn.Module):
    def __init__(self, block_channel):

        super(R_decompose, self).__init__()

        num_features = 64 + block_channel[3]//32
        self.conv0 = nn.Conv2d(num_features, num_features,
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn0 = nn.BatchNorm2d(num_features)

        self.conv1 = nn.Conv2d(num_features, num_features,
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features)

        self.conv2 = nn.Conv2d(
            num_features, 1, kernel_size=5, stride=1, padding=2, bias=True)


        self.conv0_1 = nn.Conv2d(num_features, num_features,
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn0_1 = nn.BatchNorm2d(num_features)

        self.conv1_1 = nn.Conv2d(num_features, num_features,
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1_1 = nn.BatchNorm2d(num_features)

        self.conv2_1 = nn.Conv2d(
            num_features, 1, kernel_size=5, stride=1, padding=2, bias=True)


    def forward(self, x):
        x0 = self.conv0(x)
        x0 = self.bn0(x0)
        x0 = F.relu(x0)

        x1 = self.conv1(x0)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)

        x2 = self.conv2(x1)
        x2 = F.relu(x2)

        x0_1 = self.conv0_1(x)
        x0_1 = self.bn0_1(x0_1)
        x0_1 = F.relu(x0_1)

        x1_1 = self.conv1_1(x0_1)
        x1_1 = self.bn1_1(x1_1)
        x1_1 = F.relu(x1_1)

        x2_1 = self.conv2_1(x1_1)

        x3 = x2 + x2_1

        return x3, x2


class DECODE(nn.Module):

    def __init__(self, num_features = 2048):
        super(DECODE, self).__init__()

        self.conv1 = nn.Conv2d(2048, 2048,
                               kernel_size=7, stride=1, padding=0, bias=True)


        self.up0 = nn.ConvTranspose2d(2048, 64,
                               kernel_size=4, stride=1, padding=0, bias=True)

        self.up1 = nn.Conv2d(64, 64,
                               kernel_size=3, stride=1, padding=1, bias=True)

        self.up2 = nn.Conv2d(64, 64,
                               kernel_size=3, stride=1, padding=1, bias=True)

        self.up3 = nn.Conv2d(64, 64,
                               kernel_size=3, stride=1, padding=1, bias=True)

        self.up4 = nn.Conv2d(64, 1,
                               kernel_size=3, stride=1, padding=1, bias=True)

        self.up0_seg = nn.ConvTranspose2d(2048, 64,
                               kernel_size=4, stride=1, padding=0, bias=True)

        self.up1_seg = nn.Conv2d(64, 64,
                               kernel_size=3, stride=1, padding=1, bias=True)

        self.up2_seg = nn.Conv2d(64, 64,
                               kernel_size=3, stride=1, padding=1, bias=True)

        self.up3_seg = nn.Conv2d(64, 64,
                               kernel_size=3, stride=1, padding=1, bias=True)

        self.up4_seg = nn.Conv2d(64, 2,
                               kernel_size=3, stride=1, padding=1, bias=True)



    def forward(self, x_block4):
        x_f0 = F.relu(self.conv1(x_block4))

        x_d0 = F.relu(self.up0(x_f0))
        x_d1 = F.upsample(x_d0, size=[8,8], mode='bilinear')
        x_d1 = F.relu(self.up1(x_d1))
        x_d2 = F.upsample(x_d1, size=[16,16], mode='bilinear')
        x_d2 = F.relu(self.up2(x_d2))
        x_d3 = F.upsample(x_d2, size=[32,32], mode='bilinear')
        x_d3 = F.relu(self.up3(x_d3))
        x_d4 = F.upsample(x_d3, size=[64,64], mode='bilinear')
        x_d4 = self.up4(x_d4)

        x_d0_seg = F.relu(self.up0_seg(x_f0))
        x_d1_seg = F.upsample(x_d0_seg, size=[8,8], mode='bilinear')
        x_d1_seg = F.relu(self.up1_seg(x_d1_seg))
        x_d2_seg = F.upsample(x_d1_seg, size=[16,16], mode='bilinear')
        x_d2_seg = F.relu(self.up2_seg(x_d2_seg))
        x_d3_seg = F.upsample(x_d2_seg, size=[32,32], mode='bilinear')
        x_d3_seg = F.relu(self.up3_seg(x_d3_seg))
        x_d4_seg = F.upsample(x_d3_seg, size=[64,64], mode='bilinear')
        x_d4_seg = F.sigmoid(self.up4_seg(x_d4_seg))


        return x_d4, x_d4_seg
