from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
import time
import resnet_seg_ae as resnet
from LED2Net.Network import GlobalHeightStage


# Initialize and Reshape the Encoders
def initialize_encoder(model_name, num_classes, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None

    if model_name == "resnet18":
        """ Resnet18
        """
        # model_ft = models.resnet18(pretrained=use_pretrained)
        model_ft = resnet.resnet18(pretrained=use_pretrained, num_classes=1000)
        # set_parameter_requires_grad(model_ft)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == "resnet34":
        """ Resnet34
        """
        # model_ft = models.resnet34(pretrained=use_pretrained)
        model_ft = resnet.resnet34(pretrained=use_pretrained, num_classes=1000)
        # set_parameter_requires_grad(model_ft)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == "resnet50":
        """ Resnet50
        """
        # model_ft = models.resnet50(pretrained=use_pretrained)
        model_ft = resnet.resnet50(pretrained=use_pretrained, num_classes=1000)
        # set_parameter_requires_grad(model_ft)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == "resnet101":
        """ Resnet101
        """
        # model_ft = models.resnet101(pretrained=use_pretrained)
        model_ft = resnet.resnet101(pretrained=use_pretrained, num_classes=1000)
        # set_parameter_requires_grad(model_ft)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft


ENCODER_RESNET = [
    'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
    'resnext50_32x4d', 'resnext101_32x8d'
]
ENCODER_DENSENET = [
    'densenet121', 'densenet169', 'densenet161', 'densenet201'
]

class Resnet(nn.Module):
    def __init__(self, backbone='resnet50', pretrained=True):
        super(Resnet, self).__init__()
        assert backbone in ENCODER_RESNET
        self.encoder = getattr(models, backbone)(pretrained=pretrained)
        del self.encoder.fc, self.encoder.avgpool

    def forward(self, x):
        features = []
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)

        x = self.encoder.layer1(x);
        features.append(x)  # 1/4
        x = self.encoder.layer2(x);
        features.append(x)  # 1/8
        x = self.encoder.layer3(x);
        features.append(x)  # 1/16
        x = self.encoder.layer4(x);
        features.append(x)  # 1/32
        return features


# full model

class SegNet(nn.Module):
    x_mean = torch.FloatTensor(np.array([0.485, 0.456, 0.406])[None, :, None, None])
    x_std = torch.FloatTensor(np.array([0.229, 0.224, 0.225])[None, :, None, None])

    def __init__(self, encoder, backbone):
        super(SegNet, self).__init__()
        self.resnet = encoder
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.out_scale = 8
        self.step_cols = 4
        # resnet50
        # self.conv1 = nn.Conv2d(2048, 1024, kernel_size=3, stride=1, padding=1)
        # self.conv2 = nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=1)
        # self.conv3 = nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1)
        # self.conv4 = nn.Conv2d(512, 64, kernel_size=3, stride=1, padding=1)
        # self.conv5 = nn.Conv2d(128, 2, kernel_size=3, stride=1, padding=1)
        # #self.conv5 = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)
        #
        # self.conv11 = nn.Conv2d(2048, 1024, kernel_size=3, stride=1, padding=1)
        # self.conv22 = nn.Conv2d(2048+1024, 512, kernel_size=3, stride=1, padding=1)
        # self.conv33 = nn.Conv2d(1024+512, 256, kernel_size=3, stride=1, padding=1)
        # self.conv44 = nn.Conv2d(512+256, 64, kernel_size=3, stride=1, padding=1)
        # self.conv55 = nn.Conv2d(128+64, 2, kernel_size=3, stride=1, padding=1)

        # resnet18/34     128 256                                                   512  1024
        # self.conv1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)#512, 256
        # self.conv2 = nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1)#512, 128
        # self.conv3 = nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1)#256, 64
        # self.conv4 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)#128, 64
        # self.conv5 = nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1)#128, 2
        #
        # self.conv11 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)#512, 256
        # self.conv22 = nn.Conv2d(128 + 64, 32, kernel_size=3, stride=1, padding=1) #512 + 256, 128
        # self.conv33 = nn.Conv2d(64 + 32, 16, kernel_size=3, stride=1, padding=1)  ###256 + 128, 64
        # self.conv44 = nn.Conv2d(32 + 16, 64, kernel_size=3, stride=1, padding=1)#128 + 64, 64
        # self.conv55 = nn.Conv2d(32 + 16, 4, kernel_size=3, stride=1, padding=1)#128 + 64, 2

        # resnet18/34
        self.conv1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        # self.conv5 = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1)

        self.conv11 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.conv22 = nn.Conv2d(512 + 256, 128, kernel_size=3, stride=1, padding=1)
        self.conv33 = nn.Conv2d(256 + 128, 64, kernel_size=3, stride=1, padding=1)
        self.conv44 = nn.Conv2d(128 + 64, 64, kernel_size=3, stride=1, padding=1)
        self.conv55 = nn.Conv2d(128 + 64, 4, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def _prepare_x(self, x):
        x = x.clone()
        if self.x_mean.device != x.device:
            self.x_mean = self.x_mean.to(x.device)
            self.x_std = self.x_std.to(x.device)
        x[:, :3] = (x[:, :3] - self.x_mean) / self.x_std

        return x

    def forward(self, images):
        x4, x3, x2, x1, x0 = self.resnet(images)

        x3_ = self.relu(self.conv1(self.upsample(x4)))
        x3_c = torch.cat((x3_, x3), 1)
        x2_ = self.relu(self.conv2(self.upsample(x3_c)))
        x2_c = torch.cat((x2_, x2), 1)
        x1_ = self.relu(self.conv3(self.upsample(x2_c)))
        x1_c = torch.cat((x1_, x1), 1)
        x0_ = self.relu(self.conv4(self.upsample(x1_c)))
        x0_c = torch.cat((x0_, x0), 1)
        x0_cs = self.sigmoid(self.conv5(self.upsample(x0_c)))

        # cor
        x = self.relu(self.conv11(self.upsample(x4)))
        x = torch.cat((x, x3_c), 1)
        x = self.relu(self.conv22(self.upsample(x)))
        x = torch.cat((x, x2_c), 1)
        x = self.relu(self.conv33(self.upsample(x)))
        x = torch.cat((x, x1_c), 1)
        x = self.relu(self.conv44(self.upsample(x)))
        x = torch.cat((x, x0_c), 1)
        # x = self.sigmoid(self.conv55(self.upsample(x)))
        x = self.conv55(self.upsample(x))

        return x


# Set Model Parameters, requires_grad attribute
def set_parameter_requires_grad(model):
    for param in model.parameters():
        param.requires_grad = True


if __name__ == '__main__':
    model_ft = SegNet(encoder="resnet34", num_classes=1)
    x = torch.zeros(1, 3, 224, 224)
    outs, x = model_ft(x)
    print(outs.shape)
