import os.path
from os.path import join

from matplotlib import pyplot as plt

from data.image_folder import make_dataset
from data.transforms import Sobel, to_norm_tensor, to_tensor, ReflectionSythesis_1, ReflectionSythesis_2
from PIL import Image
import random
import torch
import math

import torchvision.transforms as transforms
import torchvision.transforms.functional as F

import utils.util as util
import data.torchdata as torchdata


def __scale_width(img, target_width):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    h = math.ceil(h / 2.) * 2  # round up to even
    return img.resize((w, h), Image.BICUBIC)


def __scale_height(img, target_height):
    ow, oh = img.size
    if (oh == target_height):
        return img
    h = target_height
    w = int(target_height * ow / oh)
    w = math.ceil(w / 2.) * 2  # round up to even
    return img.resize((w, h), Image.BICUBIC)


def paired_data_transforms(img_1, img_2, mask):

    img_1_256 = F.resize(img_1, (256, 256))
    img_2_256 = F.resize(img_2, (256, 256))
    mask_256 = F.resize(mask, (256, 256))
    # img_1_256 = img_1
    # img_2_256 = img_2
    # mask_256 = mask
    # img_1_512 = img_1
    # img_2_512 = img_2

    # 添加旋转角度
    angle = random.randint(-10, 10)
    img_1_256 = F.rotate(img_1_256, angle)
    img_2_256 = F.rotate(img_2_256, angle)
    mask_256 = F.rotate(mask_256, angle)
    # img_1_512 = F.rotate(img_1_512, angle)
    # img_2_512 = F.rotate(img_2_512, angle)

    # 添加高斯模糊
    if random.random() < 0.5:
        kernel_size = (5, 5)
        sigma = (0.1, 3.0)
        img_1_256 = F.gaussian_blur(img_1_256, kernel_size, sigma)
        img_2_256 = F.gaussian_blur(img_2_256, kernel_size, sigma)
        # img_1_512 = F.gaussian_blur(img_1_512, kernel_size, sigma)
        # img_2_512 = F.gaussian_blur(img_2_512, kernel_size, sigma)

    # 添加亮度、对比度调整
    if random.random() < 0.5:
        brightness_factor = 0.5
        contrast_factor = 0.5
        saturation = 0.5
        img_1_256 = F.adjust_brightness(img_1_256, brightness_factor)
        img_1_256 = F.adjust_contrast(img_1_256, contrast_factor)
        img_1_256 = F.adjust_saturation(img_1_256, saturation)
        img_2_256 = F.adjust_brightness(img_2_256, brightness_factor)
        img_2_256 = F.adjust_contrast(img_2_256, contrast_factor)
        img_2_256 = F.adjust_saturation(img_2_256, saturation)
        # img_1_512 = F.adjust_brightness(img_1_512, brightness_factor)
        # img_1_512 = F.adjust_contrast(img_1_512, contrast_factor)
        # img_1_512 = F.adjust_saturation(img_1_512, saturation)
        # img_2_512 = F.adjust_brightness(img_2_512, brightness_factor)
        # img_2_512 = F.adjust_contrast(img_2_512, contrast_factor)
        # img_2_512 = F.adjust_saturation(img_2_512, saturation)

    if random.random() < 0.5:
        img_1_256 = F.hflip(img_1_256)
        img_2_256 = F.hflip(img_2_256)
        mask_256 = F.hflip(mask_256)
        # img_1_512 = F.hflip(img_1_512)
        # img_2_512 = F.hflip(img_2_512)

    if random.random() < 0.5:
        img_1_256 = F.vflip(img_1_256)
        img_2_256 = F.vflip(img_2_256)
        mask_256 = F.vflip(mask_256)
        # img_1_512 = F.vflip(img_1_512)
        # img_2_512 = F.vflip(img_2_512)

    return img_1_256, img_2_256, mask_256
    # return unaligned_transforms(img_1), unaligned_transforms(img_2)


BaseDataset = torchdata.Dataset


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size, shuffle, *args, **kwargs):
        super(DataLoader, self).__init__(dataset, batch_size, shuffle, *args, **kwargs)
        self.shuffle = shuffle

    def reset(self):
        if self.shuffle:
            print('Reset Dataset...')
            self.dataset.reset()


class CEILDataset(BaseDataset):
    def __init__(self, datadir, fns=None, size=None, enable_transforms=True, low_sigma=2, high_sigma=5, low_gamma=1.3,
                 high_gamma=1.3):
        super(CEILDataset, self).__init__()
        self.size = size
        self.datadir = datadir
        self.enable_transforms = enable_transforms

        sortkey = lambda key: os.path.split(key)[-1]
        self.paths = sorted(make_dataset(datadir, fns), key=sortkey)
        if size is not None:
            self.paths = self.paths[:size]

        self.syn_model = ReflectionSythesis_1(kernel_sizes=[11], low_sigma=low_sigma, high_sigma=high_sigma,
                                              low_gamma=low_gamma, high_gamma=high_gamma)
        self.reset(shuffle=False)

    def reset(self, shuffle=True):
        if shuffle:
            random.shuffle(self.paths)
        num_paths = len(self.paths) // 2
        self.B_paths = self.paths[0:num_paths]
        self.R_paths = self.paths[num_paths:2 * num_paths]

    def data_synthesis(self, t_img, r_img):
        if self.enable_transforms:
            t_img, r_img = paired_data_transforms(t_img, r_img)
        syn_model = self.syn_model
        t_img, r_img, m_img = syn_model(t_img, r_img)

        B = to_tensor(t_img)
        R = to_tensor(r_img)
        M = to_tensor(m_img)

        return B, R, M

    def __getitem__(self, index):
        index_B = index % len(self.B_paths)
        index_R = index % len(self.R_paths)

        B_path = self.B_paths[index_B]
        R_path = self.R_paths[index_R]

        t_img = Image.open(B_path).convert('RGB')
        r_img = Image.open(R_path).convert('RGB')

        B, R, M = self.data_synthesis(t_img, r_img)

        fn = os.path.basename(B_path)
        return {'input': M, 'target_t': B, 'target_r': R, 'fn': fn}

    def __len__(self):
        if self.size is not None:
            return min(max(len(self.B_paths), len(self.R_paths)), self.size)
        else:
            return max(len(self.B_paths), len(self.R_paths))


class CEILTestDataset(BaseDataset):
    def __init__(self, datadir, fns=None, size=None, enable_transforms=False, unaligned_transforms=False,
                 round_factor=1, flag=None):
        super(CEILTestDataset, self).__init__()
        self.size = size
        self.datadir = datadir
        self.fns = fns or os.listdir(join(datadir, 'blended'))
        self.enable_transforms = enable_transforms
        self.unaligned_transforms = unaligned_transforms
        self.round_factor = round_factor
        self.flag = flag

        if size is not None:
            self.fns = self.fns[:size]

    def __getitem__(self, index):
        fn = self.fns[index]

        t_img = Image.open(join(self.datadir, 'transmission_layer', fn)).convert('RGB')
        m_img = Image.open(join(self.datadir, 'blended', fn)).convert('RGB')

        if self.enable_transforms:
            t_img, m_img = paired_data_transforms(t_img, m_img, self.unaligned_transforms)

        B = to_tensor(t_img)
        M = to_tensor(m_img)

        dic = {'input': M, 'target_t': B, 'fn': fn, 'target_r': B}  # fake reflection gt
        if self.flag is not None:
            dic.update(self.flag)
        return dic

    def __len__(self):
        if self.size is not None:
            return min(len(self.fns), self.size)
        else:
            return len(self.fns)


class CEILTrainDataset(BaseDataset):
    def __init__(self, datadir, fns=None, size=None, enable_transforms=False, round_factor=1, flag=None):
        super(CEILTrainDataset, self).__init__()
        self.size = size
        self.datadir = datadir
        self.fns = fns or os.listdir(join(datadir, 'blended'))
        self.enable_transforms = enable_transforms
        self.round_factor = round_factor
        self.flag = flag

        if size is not None:
            self.fns = self.fns[:size]

    def __getitem__(self, index):
        fn = self.fns[index]
        fn_mask = fn.split(".")[0] + ".png"
        t_img_512 = Image.open(join(self.datadir, 'transmission_layer', fn)).convert('RGB')
        m_img_512 = Image.open(join(self.datadir, 'blended', fn)).convert('RGB')
        mask_512 = Image.open(join(self.datadir, 'mask', fn_mask)).convert('L')

        if self.enable_transforms:
            t_img, m_img, mask_256 = paired_data_transforms(t_img_512, m_img_512, mask_512)

        B = to_tensor(t_img)
        M = to_tensor(m_img)
        mask_256 = to_tensor(mask_256)

        # # 展示图像
        # image_np1 = B.permute(1, 2, 0).numpy()
        # image_np2 = M.permute(1, 2, 0).numpy()
        # fig, axes = plt.subplots(1, 2)
        # axes[0].imshow(image_np1)
        # axes[0].axis('off')
        # axes[1].imshow(image_np2)
        # axes[1].axis('off')
        # plt.tight_layout()
        # plt.show()

        # dic = {'input': M, 'target_t': B, 'fn': fn, 'target_r': B, 'mask': mask_512, 'input_512': M_512,
        #        'target_t_512': B_512}
        dic = {'input': M, 'target_t': B, 'fn': fn, 'target_r': B, 'mask': mask_256}
        if self.flag is not None:
            dic.update(self.flag)
        return dic

    def __len__(self):
        if self.size is not None:
            return min(len(self.fns), self.size)
        else:
            return len(self.fns)


class RealDataset(BaseDataset):
    def __init__(self, datadir, fns=None, size=None):
        super(RealDataset, self).__init__()
        self.size = size
        self.datadir = datadir
        self.fns = fns or os.listdir(join(datadir))

        if size is not None:
            self.fns = self.fns[:size]

    def __getitem__(self, index):
        fn = self.fns[index]
        B = -1

        m_img = Image.open(join(self.datadir, fn)).convert('RGB')

        M = to_tensor(m_img)
        data = {'input': M, 'target_t': B, 'fn': fn}
        return data

    def __len__(self):
        if self.size is not None:
            return min(len(self.fns), self.size)
        else:
            return len(self.fns)


class PairedCEILDataset(CEILDataset):
    def __init__(self, datadir, fns=None, size=None, enable_transforms=True, low_sigma=2, high_sigma=5):
        self.size = size
        self.datadir = datadir

        self.fns = fns or os.listdir(join(datadir, 'reflection_layer'))
        if size is not None:
            self.fns = self.fns[:size]

        self.syn_model = ReflectionSythesis_1(kernel_sizes=[11], low_sigma=low_sigma, high_sigma=high_sigma)
        self.enable_transforms = enable_transforms
        self.reset()

    def reset(self):
        return

    def __getitem__(self, index):
        fn = self.fns[index]
        B_path = join(self.datadir, 'transmission_layer', fn)
        R_path = join(self.datadir, 'reflection_layer', fn)

        t_img = Image.open(B_path).convert('RGB')
        r_img = Image.open(R_path).convert('RGB')

        B, R, M = self.data_synthesis(t_img, r_img)

        data = {'input': M, 'target_t': B, 'target_r': R, 'fn': fn}
        # return M, B
        return data

    def __len__(self):
        if self.size is not None:
            return min(len(self.fns), self.size)
        else:
            return len(self.fns)


class FusionDataset(BaseDataset):
    def __init__(self, datasets, fusion_ratios=None):
        self.datasets = datasets
        self.size = sum([len(dataset) for dataset in datasets])
        self.fusion_ratios = fusion_ratios or [1. / len(datasets)] * len(datasets)
        print('[i] using a fusion dataset: %d %s imgs fused with ratio %s' % (
            self.size, [len(dataset) for dataset in datasets], self.fusion_ratios))

    def reset(self):
        for dataset in self.datasets:
            dataset.reset()

    def __getitem__(self, index):
        residual = 1
        for i, ratio in enumerate(self.fusion_ratios):
            if random.random() < ratio / residual or i == len(self.fusion_ratios) - 1:
                dataset = self.datasets[i]
                return dataset[index % len(dataset)]
            residual -= ratio

    def __len__(self):
        return self.size


class RepeatedDataset(BaseDataset):
    def __init__(self, dataset, repeat=1):
        self.dataset = dataset
        self.size = len(dataset) * repeat
        # self.reset()

    def reset(self):
        self.dataset.reset()

    def __getitem__(self, index):
        dataset = self.dataset
        return dataset[index % len(dataset)]

    def __len__(self):
        return self.size
