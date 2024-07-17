# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

'''
Simple Baselines for Image Restoration

@article{chen2022simple,
  title={Simple Baselines for Image Restoration},
  author={Chen, Liangyu and Chu, Xiaojie and Zhang, Xiangyu and Sun, Jian},
  journal={arXiv preprint arXiv:2204.04676},
  year={2022}
}
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
# from .models.archs.arch_util import LayerNorm2d
import sys

sys.path.append('/ghome/zhuyr/Deref_RW/networks/')


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class NAFNet(nn.Module):

    def __init__(self, img_channel=3, width=32, middle_blk_num=1, enc_blk_nums=[1, 1, 1, 28],
                 dec_blk_nums=[1, 1, 1, 1], global_residual=False, drop_flag=False, drop_rate=0.4):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1,
                               groups=1,
                               bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=3, kernel_size=3, padding=1, stride=1, groups=1,
                                bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.global_residual = global_residual
        self.drop_flag = drop_flag

        if drop_flag:
            self.dropout = nn.Dropout2d(p=drop_rate)

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2 * chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)
        base_inp = inp[:, :3, :, :]
        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        if self.drop_flag:
            x = self.dropout(x)

        x = self.ending(x)
        if self.global_residual:
            # print(x.shape, inp.shape, base_inp.shape)
            x = x + base_inp
        else:
            x
        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


class NAFNet_wDetHead(nn.Module):

    def __init__(self, img_channel=3, width=32, middle_blk_num=1, enc_blk_nums=None,
                 dec_blk_nums=[1, 1, 1, 1], global_residual=True, drop_flag=True, drop_rate=0.4,
                 concat=False, merge_manner=0):
        super().__init__()

        if enc_blk_nums is None:
            enc_blk_nums = [1, 1, 1, 28]
        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1,
                               groups=1,
                               bias=True)
        # self.intro2 = nn.Conv2d(in_channels=img_channel, out_channels=width*2, kernel_size=3, padding=1, stride=1,
        #                        groups=1,
        #                        bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=3, kernel_size=3, padding=1, stride=1, groups=1,
                                bias=True)
        # self.conv1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=2,
        #                        groups=1,
        #                        bias=True)
        # self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=2,
        #                        groups=1,
        #                        bias=True)
        # self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=2,
        #                        groups=1,
        #                        bias=True)
        # self.merge_layers = [
        #     nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1, stride=1, groups=1, bias=True),
        #     nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, stride=1, groups=1, bias=True),
        #     nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1, stride=1, groups=1, bias=True),
        #     nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        # ]

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.global_residual = global_residual
        self.drop_flag = drop_flag
        self.concat = concat
        self.merge_manner = merge_manner

        self.conv_layers_d = [
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1, stride=1, groups=1, bias=True),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, stride=1, groups=1, bias=True),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1, stride=1, groups=1, bias=True),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1, stride=1, groups=1, bias=True)]

        self.a_pool_16 = nn.MaxPool2d(kernel_size=[16, 16], stride=16)
        self.a_pool_128 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        if drop_flag:
            self.dropout = nn.Dropout2d(p=drop_rate)
        self.G_pool = nn.MaxPool2d(kernel_size=[2, 2], stride=2)

        # --------------------------- Merge sparse & Img -------------------------------------------------------
        self.intro_Det = nn.Conv2d(in_channels=1, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                                   bias=True)
        self.DetEnc = nn.Sequential(*[NAFBlock(width) for _ in range(3)])
        if self.concat:
            self.Merge_conv = nn.Conv2d(in_channels=width * 2, out_channels=width, kernel_size=3, padding=1, stride=1,
                                        groups=1,
                                        bias=True)
        else:
            self.Merge_conv = nn.Conv2d(in_channels=width, out_channels=width, kernel_size=3, padding=1, stride=1,
                                        groups=1,
                                        bias=True)
        # ---------------------------  Merge sparse & Img -------------------------------------------------------

        chan = width
        if enc_blk_nums is None:
            enc_blk_nums = [1, 1, 1, 28]
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2 * chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )
        if dec_blk_nums is None:
            dec_blk_nums = [1, 1, 1, 1]
        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp, spare_ref):
        B, C, H, W = inp.shape
        # inp = self.check_image_size(inp)
        base_inp = inp
        # ves_s = inp * x2

        # # 找出在 spare_ref 中非零元素并且对应位置在 x2 中的值不为 1 的位置
        # non_zero_indices = (spare_ref != 0) & (x2 != 1)
        # x3 = torch.zeros_like(spare_ref)
        # x3[non_zero_indices] = spare_ref[non_zero_indices]
        # spare_ref = x3

        x = self.intro(inp)
        # ves = self.intro(ves_s)

        fea_sparse = self.intro_Det(spare_ref)
        fea_sparse = self.DetEnc(fea_sparse)

        if self.merge_manner == 0 and self.concat:
            x = torch.cat([x, fea_sparse], dim=1)
            x = self.Merge_conv(x)
        elif self.merge_manner == 1 and not self.concat:
            x = x + fea_sparse
            x = self.Merge_conv(x)
        elif self.merge_manner == 2 and not self.concat:
            x = x + fea_sparse * x
            x = self.Merge_conv(x)
        else:
            x = x
            print('Merge Flag Error!!!(No Merge Operation)')

        encs = []
        # x = x * x2.expand_as(x) + x

        for encoder, down, conv_layer in zip(self.encoders, self.downs, self.conv_layers_d):
            # x = torch.cat([x, ves], dim=1)
            # x = conv_layer(x.cpu())
            # x = x.cuda()
            x = encoder(x)
            encs.append(x)
            x = down(x)
            # ves = down(ves)

        x = self.middle_blks(x)
        # mask_16 = self.a_pool_16(x2)
        # x = x * mask_16.expand_as(x) + x

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        if self.drop_flag:
            x = self.dropout(x)

        x = self.ending(x)
        if self.global_residual:
            # print(x.shape, inp.shape, base_inp.shape)
            x = x + base_inp
        else:
            x

        x = x[:, :, :H, :W]

        # x = F.interpolate(x, size=(512, 512), mode='bilinear', align_corners=False)

        return x

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_channel = 3
    width = 32
    enc_blks = [1, 1, 1, 28]
    middle_blk_num = 1
    dec_blks = [1, 1, 1, 1]

    # net = NAFNet_wDetHead(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
    #                       enc_blk_nums=enc_blks, dec_blk_nums=dec_blks, global_residual=True,
    #                       concat=True, merge_manner=2)  # .cuda()
    net = NAFNet_wDetHead(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                          enc_blk_nums=enc_blks, dec_blk_nums=dec_blks, global_residual=True,
                          drop_flag=True, drop_rate=0.4, merge_manner=0, concat=True)
    net.to(device)
    input = torch.randn([4, 3, 256, 256]).cuda()  # .cuda()  inp_shape = (5, 3, 128, 128)
    spare = torch.randn([4, 1, 256, 256]).cuda()
    mask = torch.randn([4, 1, 256, 256]).cuda()
    print(net(input, spare, mask).size())
