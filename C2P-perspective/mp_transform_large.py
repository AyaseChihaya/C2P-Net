
import torch
import numpy as np
from PIL import Image, ImageOps
import collections
try:
    import accimage
except ImportError:
    accimage = None
import random
import scipy.ndimage as ndimage
import torchvision.transforms as transforms
import pdb
import matplotlib.pyplot as plt

def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


class RandomRotate(object):
    """Random rotation of the image from -angle to angle (in degrees)
    This is useful for dataAugmentation, especially for geometric problems such as FlowEstimation
    angle: max angle of the rotation
    interpolation order: Default: 2 (bilinear)
    reshape: Default: false. If set to true, image size will be set to keep every pixel in the image.
    diff_angle: Default: 0. Must stay less than 10 degrees, or linear approximation of flowmap will be off.
    """

    def __init__(self, angle, diff_angle=0, order=2, reshape=False):
        self.angle = angle
        self.reshape = reshape
        self.order = order

    def __call__(self, sample):
        lsun_image, lsun_seg, lsun_depth, lsun_raw_depth = sample['lsun_image'], sample['lsun_seg'], sample['lsun_depth'], sample['lsun_raw_depth']

        applied_angle = random.uniform(-self.angle, self.angle)
        angle1 = applied_angle
        angle1_rad = angle1 * np.pi / 180

        # lsun_image = transforms.RandomRotation(degrees, interpolation=InterpolationMode.NEAREST, expand=False, center=None, fill=0)

        lsun_image = ndimage.interpolation.rotate(
            lsun_image, angle1, reshape=self.reshape, order=self.order, mode='nearest')
        lsun_seg = ndimage.interpolation.rotate(
            lsun_seg, angle1, reshape=self.reshape, order=0, mode='nearest')
        lsun_depth = ndimage.interpolation.rotate(
            lsun_depth, angle1, reshape=self.reshape, order=self.order, mode='nearest')
        lsun_raw_depth = ndimage.interpolation.rotate(
            lsun_raw_depth, angle1, reshape=self.reshape, order=self.order, mode='nearest')

        lsun_image = Image.fromarray(lsun_image)
        lsun_seg = Image.fromarray(lsun_seg)
        lsun_depth = Image.fromarray(lsun_depth)
        lsun_raw_depth = Image.fromarray(lsun_raw_depth)


        return {'lsun_image': lsun_image, 'lsun_seg': lsun_seg, 'lsun_depth': lsun_depth, 'lsun_raw_depth': lsun_raw_depth}

class RandomHorizontalFlip(object):

    def __call__(self, sample):
        lsun_image, lsun_seg, lsun_depth, lsun_raw_depth = sample['lsun_image'], sample['lsun_seg'], sample['lsun_depth'], sample['lsun_raw_depth']

        if not _is_pil_image(lsun_image):
            raise TypeError(
                'img should be PIL Image. Got {}'.format(type(img)))
        if not _is_pil_image(lsun_depth):
            raise TypeError(
                'img should be PIL Image. Got {}'.format(type(depth)))

        if random.random() < 0.5:
            lsun_image = lsun_image.transpose(Image.FLIP_LEFT_RIGHT)
            lsun_seg = lsun_seg.transpose(Image.FLIP_LEFT_RIGHT)
            lsun_depth = lsun_depth.transpose(Image.FLIP_LEFT_RIGHT)
            lsun_raw_depth = lsun_raw_depth.transpose(Image.FLIP_LEFT_RIGHT)

        return {'lsun_image': lsun_image, 'lsun_seg': lsun_seg, 'lsun_depth': lsun_depth, 'lsun_raw_depth': lsun_raw_depth}


class Scale(object):
    """ Rescales the inputs and target arrays to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation order: Default: 2 (bilinear)
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        lsun_image, lsun_seg, lsun_depth, lsun_raw_depth = sample['lsun_image'], sample['lsun_seg'], sample['lsun_depth'], sample['lsun_raw_depth']

        lsun_image = self.changeScale(lsun_image, self.size)
        lsun_seg = self.changeScale(lsun_seg, self.size, Image.NEAREST)
        lsun_depth = self.changeScale(lsun_depth, self.size)
        lsun_raw_depth = self.changeScale(lsun_raw_depth, self.size, Image.NEAREST)

        return {'lsun_image': lsun_image, 'lsun_seg': lsun_seg, 'lsun_depth': lsun_depth, 'lsun_raw_depth': lsun_raw_depth}

    def changeScale(self, img, size, interpolation=Image.BILINEAR):

        if not _is_pil_image(img):
            raise TypeError(
                'img should be PIL Image. Got {}'.format(type(img)))
        if not (isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)):
            raise TypeError('Got inappropriate size arg: {}'.format(size))

        if isinstance(size, int):
            # w, h = img.size
            return img.resize((size, size), interpolation)
            # w, h = img.size
            # if (w <= h and w == size) or (h <= w and h == size):
            #     return img
            # if w < h:
            #     ow = size
            #     oh = int(size * h / w)
            #     return img.resize((ow, oh), interpolation)
            # else:
            #     oh = size
            #     ow = int(size * w / h)
            #     return img.resize((ow, oh), interpolation)
        else:
            return img.resize(size[::-1], interpolation)


class CenterCrop(object):
    def __init__(self, size_image, size_depth, is_test=False):
        self.size_image = size_image
        self.size_depth = size_depth
        self.is_test = is_test

    def __call__(self, sample):
        lsun_image, lsun_seg, lsun_depth, lsun_raw_depth = sample['lsun_image'], sample['lsun_seg'], sample['lsun_depth'], sample['lsun_raw_depth']

        lsun_image = self.centerCrop(lsun_image, self.size_image)
        lsun_seg = self.centerCrop(lsun_seg, self.size_image)
        lsun_depth = self.centerCrop(lsun_depth, self.size_image)
        lsun_raw_depth = self.centerCrop(lsun_raw_depth, self.size_image)

        ow, oh = self.size_depth
        lsun_depth = lsun_depth.resize((ow, oh))
        lsun_seg = lsun_seg.resize((ow, oh), Image.NEAREST)
        lsun_raw_depth = lsun_raw_depth.resize((ow, oh), Image.NEAREST)

        # lsun_image, lsun_seg, lsun_depth, lsun_raw_depth = self.randomCrop(lsun_image, lsun_seg, lsun_depth, lsun_raw_depth, self.size_image)
        #
        # owi, ohi = self.size_image
        # lsun_image = lsun_image.resize((owi, ohi))
        # ow, oh = self.size_depth
        # lsun_seg = lsun_seg.resize((ow, oh), Image.NEAREST)
        # lsun_depth = lsun_depth.resize((ow, oh))
        # lsun_raw_depth = lsun_raw_depth.resize((ow, oh))

        return {'lsun_image': lsun_image, 'lsun_seg': lsun_seg, 'lsun_depth': lsun_depth, 'lsun_raw_depth': lsun_raw_depth}

    def centerCrop(self, image, size):

        w1, h1 = image.size

        tw, th = size

        if w1 == tw and h1 == th:
            return image

        x1 = int(round((w1 - tw) / 2.))
        y1 = int(round((h1 - th) / 2.))

        image = image.crop((x1, y1, tw + x1, th + y1))

        return image

    def randomCrop(self, lsun_image, lsun_seg, lsun_depth, lsun_raw_depth, size):

        w1, h1 = lsun_image.size

        tw, th = size

        if w1 == tw and h1 == th:
            return lsun_image, lsun_seg, lsun_depth, lsun_raw_depth

        x1 = random.randint(0,(w1 - tw))
        y1 = random.randint(0,(h1 - th))

        lsun_image = lsun_image.crop((x1, y1, tw + x1, th + y1))
        lsun_seg = lsun_seg.crop((x1, y1, tw + x1, th + y1))
        lsun_depth = lsun_depth.crop((x1, y1, tw + x1, th + y1))
        lsun_raw_depth = lsun_raw_depth.crop((x1, y1, tw + x1, th + y1))

        return lsun_image, lsun_seg, lsun_depth, lsun_raw_depth




class ToTensor(object):
    """Convert a ``PIL.Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """
    def __init__(self,is_test=False):
        self.is_test = is_test

    def __call__(self, sample):
        lsun_image, lsun_seg, lsun_depth, lsun_raw_depth = sample['lsun_image'], sample['lsun_seg'], sample['lsun_depth'], sample['lsun_raw_depth']
        """
        Args:
            pic (PIL.Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        # ground truth depth of training samples is stored in 8-bit while test samples are saved in 16 bit

        transform = transforms.Compose([
            transforms.PILToTensor()
        ])

        lsun_image = transform(lsun_image).float() / 255.
        lsun_seg = transform(lsun_seg)
        lsun_depth = transform(lsun_depth).float() / 4000.
        lsun_raw_depth = transform(lsun_raw_depth).float() / 4000.

        # print(lsun_image.shape)
        # print(lsun_image)


        return {'lsun_image': lsun_image, 'lsun_seg': lsun_seg, 'lsun_depth': lsun_depth, 'lsun_raw_depth': lsun_raw_depth}

    # def transform(self, lsun_image, lsun_seg, lsun_depth, size):
    #     transform = transforms.Compose([
    #         transforms.PILToTensor()
    #     ])


# class ToTensor(object):
#     """Convert a ``PIL.Image`` or ``numpy.ndarray`` to tensor.
#     Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
#     [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
#     """
#     def __init__(self,is_test=False):
#         self.is_test = is_test
#
#     def __call__(self, sample):
#         lsun_image, lsun_seg, lsun_depth, lsun_raw_depth = sample['lsun_image'], sample['lsun_seg'], sample['lsun_depth'], sample['lsun_raw_depth']
#         """
#         Args:
#             pic (PIL.Image or numpy.ndarray): Image to be converted to tensor.
#         Returns:
#             Tensor: Converted image.
#         """
#         # ground truth depth of training samples is stored in 8-bit while test samples are saved in 16 bit
#
#         lsun_image = self.to_tensor(lsun_image)
#         lsun_seg = self.to_tensor(lsun_seg) * 255
#         lsun_depth = self.to_tensor(lsun_depth).float() / 4000.
#         lsun_raw_depth = self.to_tensor(lsun_raw_depth).float() / 4000.
#
#
#         return {'lsun_image': lsun_image, 'lsun_seg': lsun_seg, 'lsun_depth': lsun_depth, 'lsun_raw_depth': lsun_raw_depth}
#
#     def to_tensor(self, pic):
#         if not(_is_pil_image(pic) or _is_numpy_image(pic)):
#             raise TypeError(
#                 'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))
#
#         if isinstance(pic, np.ndarray):
#             img = torch.from_numpy(pic.transpose((2, 0, 1)))
#
#             return img.float().div(255)
#
#         if accimage is not None and isinstance(pic, accimage.Image):
#             nppic = np.zeros(
#                 [pic.channels, pic.height, pic.width], dtype=np.float32)
#             pic.copyto(nppic)
#             return torch.from_numpy(nppic)
#         #
#         # # handle PIL Image
#         if pic.mode == 'I':
#             img = torch.from_numpy(np.array(pic, np.int32))
#             # mg = torch.from_numpy(np.array(pic, np.int32, copy=False))
#         elif pic.mode == 'I;16':
#             img = torch.from_numpy(np.array(pic, np.int16))
#             # img = torch.from_numpy(np.array(pic, np.int16, copy=False))
#         else:
#             img = torch.ByteTensor(
#                 torch.ByteStorage.from_buffer(pic.tobytes()))
#         # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
#         if pic.mode == 'YCbCr':
#             nchannel = 3
#         elif pic.mode == 'I;16':
#             nchannel = 1
#         else:
#             nchannel = len(pic.mode)
#
#         img = img.view(pic.size[1], pic.size[0], nchannel)
#         # put it from HWC to CHW format
#         # yikes, this transpose takes 80% of the loading time/CPU
#         img = img.transpose(0, 1).transpose(0, 2).contiguous()
#         if isinstance(img, torch.ByteTensor):
#             return img.float().div(255)
#         else:
#             return img


class Lighting(object):

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, sample):
        lsun_image, lsun_seg, lsun_depth, lsun_raw_depth = sample['lsun_image'], sample['lsun_seg'], sample['lsun_depth'], sample['lsun_raw_depth']
        if self.alphastd == 0:
            return lsun_image

        lsun_alpha = lsun_image.new().resize_(3).normal_(0, self.alphastd)
        lsun_rgb = self.eigvec.type_as(lsun_image).clone()\
            .mul(lsun_alpha.view(1, 3).expand(3, 3))\
            .mul(self.eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()
        lsun_image = lsun_image.add(lsun_rgb.view(3, 1, 1).expand_as(lsun_image))

        return {'lsun_image': lsun_image, 'lsun_seg': lsun_seg, 'lsun_depth': lsun_depth, 'lsun_raw_depth': lsun_raw_depth}


class Grayscale(object):

    def __call__(self, img):
        gs = img.clone()
        # gs[0].mul_(0.299).add_(gs[1], 0.587).add_(gs[2], 0.114)
        gs[0].mul_(0.299).add_(0.587, gs[1]).add_(0.114, gs[2])
        gs[1].copy_(gs[0])
        gs[2].copy_(gs[0])
        return gs


class Saturation(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        alpha = random.uniform(-self.var, self.var)
        return img.lerp(gs, alpha)


class Brightness(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = img.new().resize_as_(img).zero_()
        alpha = random.uniform(-self.var, self.var)

        return img.lerp(gs, alpha)


class Contrast(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        gs.fill_(gs.mean())
        alpha = random.uniform(-self.var, self.var)
        return img.lerp(gs, alpha)


class RandomOrder(object):
    """ Composes several transforms together in random order.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        lsun_image, lsun_seg, lsun_depth, lsun_raw_depth = sample['lsun_image'], sample['lsun_seg'], sample['lsun_depth'], sample['lsun_raw_depth']

        if self.transforms is None:
            return {'lsun_image': lsun_image, 'lsun_seg': lsun_seg, 'lsun_depth': lsun_depth, 'lsun_raw_depth': lsun_raw_depth}
        order = torch.randperm(len(self.transforms))
        for i in order:
            lsun_image = self.transforms[i](lsun_image)

        return {'lsun_image': lsun_image, 'lsun_seg': lsun_seg, 'lsun_depth': lsun_depth, 'lsun_raw_depth': lsun_raw_depth}


class ColorJitter(RandomOrder):

    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4):
        self.transforms = []
        if brightness != 0:
            self.transforms.append(Brightness(brightness))
        if contrast != 0:
            self.transforms.append(Contrast(contrast))
        if saturation != 0:
            self.transforms.append(Saturation(saturation))


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        lsun_image, lsun_seg, lsun_depth, lsun_raw_depth = sample['lsun_image'], sample['lsun_seg'], sample['lsun_depth'], sample['lsun_raw_depth']

        lsun_image = self.normalize(lsun_image, self.mean, self.std)

        return {'lsun_image': lsun_image, 'lsun_seg': lsun_seg, 'lsun_depth': lsun_depth, 'lsun_raw_depth': lsun_raw_depth}

    def normalize(self, tensor, mean, std):
        """Normalize a tensor image with mean and standard deviation.
        See ``Normalize`` for more details.
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
            mean (sequence): Sequence of means for R, G, B channels respecitvely.
            std (sequence): Sequence of standard deviations for R, G, B channels
                respecitvely.
        Returns:
            Tensor: Normalized image.
        """

        # TODO: make efficient
        for t, m, s in zip(tensor, mean, std):
            t.sub_(m).div_(s)
        return tensor
