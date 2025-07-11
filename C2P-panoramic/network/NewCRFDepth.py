import torch
import torch.nn as nn
import torch.nn.functional as F

from .swin_transformer import SwinTransformer
from .newcrf_layers import NewCRF
from .uper_crf_head import PSP

from matplotlib import pyplot as plt
import numpy as np

class ConvCompressH(nn.Module):
    ''' Reduce feature height by factor of two '''
    def __init__(self, in_c, out_c, ks=3):
        super(ConvCompressH, self).__init__()
        # assert ks % 2 == 1
        self.layers = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=ks, stride=(2, 1), padding=ks//2),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class GlobalHeightConv(nn.Module):
    def __init__(self, in_c, out_c):
        super(GlobalHeightConv, self).__init__()
        self.layer = nn.Sequential(
            ConvCompressH(in_c, in_c//2),
            ConvCompressH(in_c//2, in_c//2),
            ConvCompressH(in_c//2, in_c//4),
            ConvCompressH(in_c//4, out_c),
        )

    def forward(self, x, out_w):
        x = self.layer(x) # 4 24 8 256

        factor = out_w // x.shape[3]
        x = torch.cat([x[..., -1:], x, x[..., :1]], 3)
        x = F.interpolate(x, size=(x.shape[2], out_w + 2 * factor), mode='bilinear', align_corners=False)
        x = x[..., factor:-factor]
        return x


class GlobalHeightStage(nn.Module):
    def __init__(self, c1, c2, c3, c4, out_scale=8):
        ''' Process 4 blocks from encoder to single multiscale features '''
        super(GlobalHeightStage, self).__init__()
        self.cs = c1, c2, c3, c4
        self.out_scale = out_scale
        self.ghc_lst = nn.ModuleList([
            GlobalHeightConv(c1, c1//out_scale),
            GlobalHeightConv(c2, c2//out_scale),
            GlobalHeightConv(c3, c3//out_scale),
            GlobalHeightConv(c4, c4//out_scale),
        ])

    def forward(self, conv_list, out_w):
        assert len(conv_list) == 4
        bs = conv_list[0].shape[0]
        feature = torch.cat([
            f(x, out_w).reshape(bs, -1, out_w)
            for f, x, out_c in zip(self.ghc_lst, conv_list, self.cs)
        ], dim=1)
        return feature
class NewCRFDepth(nn.Module):
    """
    Depth network based on neural window FC-CRFs architecture.
    """
    def __init__(self, version=None, inv_depth=False, pretrained=None,
                    frozen_stages=-1, min_depth=0.1, max_depth=100.0, **kwargs):
        super().__init__()

        self.inv_depth = inv_depth
        self.with_auxiliary_head = False
        self.with_neck = False

        norm_cfg = dict(type='BN', requires_grad=True)
        # norm_cfg = dict(type='GN', requires_grad=True, num_groups=8)

        window_size = int(version[-2:])

        if version[:-2] == 'base':
            embed_dim = 128
            depths = [2, 2, 18, 2]
            num_heads = [4, 8, 16, 32]
            in_channels = [128, 256, 512, 1024]
        elif version[:-2] == 'large':
            embed_dim = 192
            depths = [2, 2, 18, 2]
            num_heads = [6, 12, 24, 48]
            in_channels = [192, 384, 768, 1536]
        elif version[:-2] == 'tiny':
            embed_dim = 96
            depths = [2, 2, 6, 2]
            num_heads = [3, 6, 12, 24]
            in_channels = [96, 192, 384, 768]

        backbone_cfg = dict(
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            ape=False,
            drop_path_rate=0.3,
            patch_norm=True,
            use_checkpoint=False,
            frozen_stages=frozen_stages
        )

        embed_dim = 512
        decoder_cfg = dict(
            in_channels=in_channels,
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=embed_dim,
            dropout_ratio=0.0,
            num_classes=32,
            norm_cfg=norm_cfg,
            align_corners=False
        )

        self.backbone = SwinTransformer(**backbone_cfg)
        v_dim = decoder_cfg['num_classes']*4
        win = 7
        crf_dims = [128, 256, 512, 1024]
        v_dims = [64, 128, 256, embed_dim]
        self.crf3 = NewCRF(input_dim=in_channels[3], embed_dim=crf_dims[3], window_size=win, v_dim=v_dims[3], num_heads=32)
        self.crf2 = NewCRF(input_dim=in_channels[2], embed_dim=crf_dims[2], window_size=win, v_dim=v_dims[2], num_heads=16)
        self.crf1 = NewCRF(input_dim=in_channels[1], embed_dim=crf_dims[1], window_size=win, v_dim=v_dims[1], num_heads=8)
        self.crf0 = NewCRF(input_dim=in_channels[0], embed_dim=crf_dims[0], window_size=win, v_dim=v_dims[0], num_heads=4)

        self.decoder = PSP(**decoder_cfg)
        self.disp_head1 = DispHead(input_dim=crf_dims[0])
        ###
        self.disp_head3 = DispHead3(input_dim=crf_dims[0])
        self.disp_head4 = DispHead4(input_dim=crf_dims[0])
        ###

        self.up_mode = 'bilinear'
        if self.up_mode == 'mask':
            self.mask_head = nn.Sequential(
                nn.Conv2d(crf_dims[0], 64, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 16*9, 1, padding=0))

        self.min_depth = min_depth
        self.max_depth = max_depth

        self.init_weights(pretrained=pretrained)
        self.out_scale = 8
        self.step_cols = 4
        self.rnn_hidden_size = 512

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 256, 512)
            c1, c2, c3, c4 = [b.shape[1] for b in self.backbone(dummy)]
            c_last = (c1 * 8 + c2 * 4 + c3 * 2 + c4 * 1) // 8  # self.out_scale
        self.reduce_height_module = GlobalHeightStage(c1, c2, c3, c4, self.out_scale)

        self.bi_rnn = nn.LSTM(input_size=288,  #c_last
                              hidden_size=self.rnn_hidden_size//2,  #self.rnn_hidden_size
                              num_layers=2,
                              dropout=0.5,
                              batch_first=False,
                              bidirectional=True)
        self.drop_out = nn.Dropout(0.5)
        self.linear = nn.Linear(in_features=self.rnn_hidden_size,
                                out_features=1)
        self.linear.bias.data[0 * self.step_cols:1 * self.step_cols].fill_(-1)
        self.linear.bias.data[1 * self.step_cols:2 * self.step_cols].fill_(-0.478)
        self.linear.bias.data[2 * self.step_cols:3 * self.step_cols].fill_(0.425)
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 1), stride=(1, 2),
                           padding=(0, 0))

        num_features = 256
        self.conv0 = nn.Conv2d(num_features, num_features,
                               kernel_size=3, stride=1, padding=1, bias=True)
        self.bn0 = nn.BatchNorm2d(num_features)

        self.conv1 = nn.Conv2d(num_features, num_features,
                               kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(num_features)

        self.conv2 = nn.Conv2d(num_features, num_features,
                               kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(num_features)

        self.conv3 = nn.Conv2d(num_features, num_features,
                               kernel_size=3, stride=1, padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(num_features)

        self.conv4 = nn.Conv2d(num_features, num_features,
                               kernel_size=3, stride=1, padding=1, bias=True)
        self.bn4 = nn.BatchNorm2d(num_features)

        self.conv5 = nn.Conv2d(num_features, num_features,
                               kernel_size=3, stride=1, padding=1, bias=True)
        self.bn5 = nn.BatchNorm2d(num_features)

        self.conv6 = nn.Conv2d(num_features, 1,
                               kernel_size=3, stride=1, padding=1, bias=True)

        num_feature = 256
        self.conv00 = nn.Conv2d(512, num_feature,
                                kernel_size=1, stride=1, padding=0, bias=True)
        self.bn00 = nn.BatchNorm2d(num_feature)


        self.conv11 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(8, 3), stride=(1, 1),  # 32, 64
                                padding=(0, 1))  # 16
        self.conv22 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(8, 5), stride=(1, 1),  # 32, 64
                                padding=(0, 0))  # 12
        self.conv33 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(8, 9), stride=(1, 1),  # 32, 64
                                padding=(0, 0))  # 8
        self.conv44 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(8, 3), stride=(1, 1),  # 32, 64
                                padding=(0, 3))  # 20
        self.conv55 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(8, 3), stride=(1, 1),  # 32, 64
                                padding=(0, 5))  # 24
        self.conv66 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(8, 1), stride=(1, 1),  # 32, 64
                                padding=(0, 8))  # 32
        self.conv77 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(32, 1), stride=(1, 1),  # 32, 64
                                padding=(0, 0))  # 64

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone and heads.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        print(f'== Load encoder backbone from: {pretrained}')
        self.backbone.init_weights(pretrained=pretrained)
        self.decoder.init_weights()
        if self.with_auxiliary_head:
            if isinstance(self.auxiliary_head, nn.ModuleList):
                for aux_head in self.auxiliary_head:
                    aux_head.init_weights()
            else:
                self.auxiliary_head.init_weights()

    def upsample_mask(self, disp, mask):
        """ Upsample disp [H/4, W/4, 1] -> [H, W, 1] using convex combination """
        N, _, H, W = disp.shape
        mask = mask.view(N, 1, 9, 4, 4, H, W)
        mask = torch.softmax(mask, dim=2)

        up_disp = F.unfold(disp, kernel_size=3, padding=1)
        up_disp = up_disp.view(N, 1, 9, 1, 1, H, W)

        up_disp = torch.sum(mask * up_disp, dim=2)
        up_disp = up_disp.permute(0, 1, 4, 2, 5, 3)
        return up_disp.reshape(N, 1, 4*H, 4*W)

    def forward(self, imgs):
        feats = self.backbone(imgs)  #4tuple#bz*192*64*128 || bz*384*32*64 || bz*768*16*32 || bz*1536*8*16
        # block1, block2,block3, block4 = feats

        ppm_out = self.decoder(feats)  #bz,512,15,20

        x_up = F.upsample(ppm_out, scale_factor=2, mode='bilinear')
        x_up = F.upsample(x_up, scale_factor=2, mode='bilinear')
        x_up = F.upsample(x_up, scale_factor=2, mode='bilinear')
        x_ = self.conv00(x_up)
        x_ = self.bn00(x_)
        x_ = F.relu(x_)
        x0 = F.relu(self.bn0(self.conv0(x_)))  # x: 8 16
        x1 = F.relu(self.bn1(self.conv1(x0)))
        x2 = F.relu(self.bn2(self.conv2(x1)))
        x3 = F.relu(self.bn3(self.conv3(x2)))
        x4 = F.relu(self.bn4(self.conv4(x3)))
        x5 = F.relu(self.bn5(self.conv5(x4)))
        x6 = self.conv6(x5)  # x6:2 1 8 16
        point = self.conv77(x6)  # 16

        e3 = self.crf3(feats[3], ppm_out)
        e3 = nn.PixelShuffle(2)(e3)
        e2 = self.crf2(feats[2], e3)
        e2 = nn.PixelShuffle(2)(e2)
        e1 = self.crf1(feats[1], e2)
        e1 = nn.PixelShuffle(2)(e1)
        e0 = self.crf0(feats[0], e1)

        e30 = self.crf3(feats[3], ppm_out)
        e30 = nn.PixelShuffle(2)(e30)
        e20 = self.crf2(feats[2], e30)
        e20 = nn.PixelShuffle(2)(e20)
        e10 = self.crf1(feats[1], e20)
        e10 = nn.PixelShuffle(2)(e10)
        e00 = self.crf0(feats[0], e10)

        e31 = self.crf3(feats[3], ppm_out)
        e31 = nn.PixelShuffle(2)(e31)
        e21 = self.crf2(feats[2], e31)
        e21 = nn.PixelShuffle(2)(e21)
        e11 = self.crf1(feats[1], e21)
        e11 = nn.PixelShuffle(2)(e11)
        e01 = self.crf0(feats[0], e11)

        d1 = self.disp_head1(e0, 4)  #disp_head1(e0, 4)
        depth = d1 * self.max_depth
        d3 = self.disp_head3(e00, 4)
        d4 = self.disp_head4(e01, 4)

        return depth, point, d3, d4


class DispHead(nn.Module):
    def __init__(self, input_dim=100):
        super(DispHead, self).__init__()
        # self.norm1 = nn.BatchNorm2d(input_dim)
        self.conv1 = nn.Conv2d(input_dim, 1, 3, padding=1) #input_dim
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, scale):
        x = self.sigmoid(self.conv1(x))
        # x = self.relu(x)
        if scale > 1:
            x = upsample(x, scale_factor=scale)
        return x

class DispHead3(nn.Module):
    def __init__(self, input_dim=100):
        super(DispHead3, self).__init__()
        # self.norm1 = nn.BatchNorm2d(input_dim)
        self.conv1 = nn.Conv2d(input_dim,66, 3, padding=1) #input_dim
        # self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, scale):
        x = self.sigmoid(self.conv1(x))
        # x = self.conv1(x)
        if scale > 1:
            x = upsample(x, scale_factor=scale)
        return x

class DispHead4(nn.Module):
    def __init__(self, input_dim=100):
        super(DispHead4, self).__init__()
        # self.norm1 = nn.BatchNorm2d(input_dim)
        self.conv11 = nn.Conv2d(input_dim, 1, 3, padding=1)#input_dim
        # self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        num_features = 128
        self.conv0 = nn.Conv2d(num_features, num_features,
                               kernel_size=3, stride=1, padding=1, bias=True)
        self.gn0 = nn.GroupNorm(8, num_features)
        self.bn0 = nn.BatchNorm2d(num_features)

        self.conv1 = nn.Conv2d(num_features, num_features,
                               kernel_size=3, stride=1, padding=1, bias=True)
        self.gn1 = nn.GroupNorm(8, num_features)
        self.bn1 = nn.BatchNorm2d(num_features)

        self.conv2 = nn.Conv2d(num_features, num_features,
                               kernel_size=3, stride=1, padding=1, bias=True)
        self.gn2 = nn.GroupNorm(8, num_features)
        self.bn2 = nn.BatchNorm2d(num_features)

        self.conv3 = nn.Conv2d(num_features, 1,
                               kernel_size=3, stride=1, padding=1, bias=True)


    def forward(self, x, scale):
        # x = self.relu(self.norm1(x))
        # x0 = F.relu(self.bn0(self.conv0(x)))
        # x1 = F.relu(self.bn1(self.conv1(x0)))
        # x2 = F.relu(self.bn2(self.conv2(x1)))
        x3 = self.sigmoid(self.conv11(x))
        if scale > 1:
            x3 = upsample(x3, scale_factor=scale)
        return x3


class DispUnpack(nn.Module):
    def __init__(self, input_dim=100, hidden_dim=128):
        super(DispUnpack, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 16, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.pixel_shuffle = nn.PixelShuffle(4)

    def forward(self, x, output_size):
        x = self.relu(self.conv1(x))
        x = self.sigmoid(self.conv2(x)) # [b, 16, h/4, w/4]
        # x = torch.reshape(x, [x.shape[0], 1, x.shape[2]*4, x.shape[3]*4])
        x = self.pixel_shuffle(x)

        return x


def upsample(x, scale_factor=2, mode="bilinear", align_corners=False):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=scale_factor, mode=mode, align_corners=align_corners)

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