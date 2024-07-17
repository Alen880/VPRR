# ------------------------------------------------------------------------
# Modified from NAFNet (https://github.com/megvii-research/NAFNet)
# ------------------------------------------------------------------------


import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt

from networks.archs.arch_util import LayerNorm2d
from networks.archs.local_arch import Local_Base
from networks.archs.Flow_arch import KernelPrior
from networks.archs.my_module import code_extra_mean_var
from networks.Lwnet import unet
from torchvision import models
from functools import partial

import numpy as np

nonlinearity = partial(F.relu, inplace=True)


class ResBlock(nn.Module):
    def __init__(self, ch):
        super(ResBlock, self).__init__()
        self.body = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1))

    def forward(self, input):
        res = self.body(input)
        output = res + input
        return output


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


class kernel_attention(nn.Module):
    def __init__(self, kernel_size, in_ch, out_ch):
        super(kernel_attention, self).__init__()

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )
        self.conv_kernel = nn.Sequential(
            nn.Conv2d(kernel_size * kernel_size, out_ch, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(out_ch * 2, out_ch, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, input, kernel):
        x = self.conv_1(input)
        kernel = self.conv_kernel(kernel)
        att = torch.cat([x, kernel], dim=1)
        att = self.conv_2(att)
        x = x * att
        output = x + input

        return output


class NAFBlock_kernel(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., kernel_size=21):
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

        self.kernel_atttion = kernel_attention(kernel_size, in_ch=c, out_ch=c)
        self.G_pool = nn.MaxPool2d(kernel_size=[2, 2], stride=2)

    def forward(self, inp, kernel):
        x = inp
        # kernel [B, 19*19, H, W]
        x = self.kernel_atttion(x, kernel)

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


def generate_k(model, code, n_row=1):
    model.eval()

    # unconditional model
    # for a random Gaussian vector, its l2norm is always close to 1.
    # therefore, in optimization, we can constrain the optimization space to be on the sphere with radius of 1

    u = code  # [B, 19*19]
    samples, _ = model.inverse(u)

    samples = model.post_process(samples)

    return samples


class DACblock(nn.Module):
    def __init__(self, channel):
        super(DACblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=5, padding=5)
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.conv1x1(self.dilate2(x)))
        dilate3_out = nonlinearity(self.conv1x1(self.dilate2(self.dilate1(x))))
        dilate4_out = nonlinearity(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out


class SPPblock(nn.Module):
    def __init__(self, in_channels):
        super(SPPblock, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=[3, 3], stride=3)
        self.pool3 = nn.MaxPool2d(kernel_size=[5, 5], stride=5)
        self.pool4 = nn.MaxPool2d(kernel_size=[6, 6], stride=6)

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1, padding=0)

    def forward(self, x):
        self.in_channels, h, w = x.size(1), x.size(2), x.size(3)
        self.layer1 = F.upsample(self.conv(self.pool1(x)), size=(h, w), mode='bilinear')
        self.layer2 = F.upsample(self.conv(self.pool2(x)), size=(h, w), mode='bilinear')
        self.layer3 = F.upsample(self.conv(self.pool3(x)), size=(h, w), mode='bilinear')
        self.layer4 = F.upsample(self.conv(self.pool4(x)), size=(h, w), mode='bilinear')

        out = torch.cat([self.layer1, self.layer2, self.layer3, self.layer4, x], 1)

        return out


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class wnet(torch.nn.Module):
    def __init__(self, n_classes=1, in_c=3, layers=(8, 16, 32), conv_bridge=True, shortcut=True, mode='test'):
        super(wnet, self).__init__()
        self.unet1 = unet(in_c=in_c, n_classes=n_classes, layers=layers, conv_bridge=conv_bridge, shortcut=shortcut)
        self.unet2 = unet(in_c=in_c + n_classes, n_classes=n_classes, layers=layers, conv_bridge=conv_bridge,
                          shortcut=shortcut)
        self.n_classes = n_classes
        self.mode = mode
        self.act = torch.sigmoid if n_classes == 1 else nn.Softmax(dim=0)

    def forward(self, x):
        x1 = self.unet1(x)
        x2 = self.unet2(torch.cat([x, x1], dim=1))
        x2 = self.act(x2)
        return x2


class UFPNet_code_uncertainty(nn.Module):
    def __init__(self, img_channel=3, width=64, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[],
                 drop_flag=True, drop_rate=0.4, kernel_size=19, n_classes=1, in_c=3,
                 layers=(8, 16, 32), conv_bridge=True, shortcut=True):
        super().__init__()

        self.kernel_size = kernel_size
        self.kernel_extra = code_extra_mean_var(kernel_size)

        self.flow = KernelPrior(n_blocks=5, input_size=19 ** 2, hidden_size=25, n_hidden=1, kernel_size=19)

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1,
                               groups=1,
                               bias=True)
        self.intro2 = nn.Conv2d(in_channels=1, out_channels=width, kernel_size=3, padding=1, stride=1,
                                groups=1,
                                bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1,
                                groups=1,
                                bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.kernel_down = nn.ModuleList()
        self.drop_flag = drop_flag
        self.intro_Det = nn.Conv2d(in_channels=1, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                                   bias=True)
        self.DetEnc = nn.Sequential(*[NAFBlock(width) for _ in range(3)])
        self.Merge_conv = nn.Conv2d(in_channels=width * 2, out_channels=width, kernel_size=3, padding=1, stride=1,
                                    groups=1,
                                    bias=True)
        self.conv_layers_d = [
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1, stride=1, groups=1, bias=True),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, stride=1, groups=1, bias=True),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1, stride=1, groups=1, bias=True),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1, stride=1, groups=1, bias=True)]
        self.conv_layers_u = [
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1, stride=1, groups=1, bias=True),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1, stride=1, groups=1, bias=True),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, stride=1, groups=1, bias=True),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1, stride=1, groups=1, bias=True)]
        if drop_flag:
            self.dropout = nn.Dropout2d(p=drop_rate)

        # self.create_ves = wnet(in_c=3, n_classes=1, layers=[8, 16, 32], conv_bridge=True, shortcut=True)
        # self.load_wnet_weights("ckpt/lwnet.pth")
        # self.freeze_wnet()
        # self.create_ves.eval()
        self.a_pool_16 = nn.MaxPool2d(kernel_size=[16, 16], stride=16)
        self.a_pool_128 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)

        chan = width
        for num in enc_blk_nums:
            if num == 1:
                self.encoders.append(
                    nn.Sequential(*[NAFBlock_kernel(chan, kernel_size=kernel_size) for _ in range(num)]))
            else:
                self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(
                nn.Conv2d(chan, 2 * chan, 2, 2)
            )
            self.kernel_down.append(nn.Conv2d(kernel_size * kernel_size, kernel_size * kernel_size, 2, 2))
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

        # self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.padder_size = 2 ** len(self.encoders)

    def load_wnet_weights(self, path):
        state_dict = torch.load(path)
        self.create_ves.load_state_dict(state_dict['model_state_dict'])

    def freeze_wnet(self):
        for param in self.create_ves.parameters():
            param.requires_grad = False

    def forward(self, inp, spare_ref):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        # ves_s = inp * x2

        # non_zero_indices = (spare_ref != 0) & (x2 != 1)
        # x3 = torch.zeros_like(spare_ref)
        # x3[non_zero_indices] = spare_ref[non_zero_indices]
        # spare_ref = x3

        with torch.no_grad():
            # kernel estimation: size [B, H*W, 19, 19]
            kernel_code, kernel_var = self.kernel_extra(inp)
            kernel_code = (kernel_code - torch.mean(kernel_code, dim=[2, 3], keepdim=True)) / torch.std(kernel_code,
                                                                                                        dim=[2, 3],
                                                                                                        keepdim=True)
            # code uncertainty
            sigma = kernel_var
            kernel_code_uncertain = kernel_code * torch.sqrt(1 - torch.square(sigma)) + torch.randn_like(
                kernel_code) * sigma

            kernel = generate_k(self.flow,
                                kernel_code_uncertain.reshape(kernel_code.shape[0] * kernel_code.shape[1], -1))
            kernel = kernel.reshape(kernel_code.shape[0], kernel_code.shape[1], self.kernel_size, self.kernel_size)
            kernel_blur = kernel

            kernel = kernel.permute(0, 2, 3, 1).reshape(B, self.kernel_size * self.kernel_size, inp.shape[2],
                                                        inp.shape[3])

        x = self.intro(inp)
        # ves = self.intro(ves_s)

        fea_sparse = self.intro_Det(spare_ref)
        fea_sparse = self.DetEnc(fea_sparse)
        x = torch.cat([x, fea_sparse], dim=1)
        x = self.Merge_conv(x)

        encs = []
        # mask_128 = self.a_pool_128(x2)

        for i, (encoder, down, kernel_down, conv_layer) in enumerate(zip(
                self.encoders, self.downs, self.kernel_down, self.conv_layers_d)):
            # x = torch.cat([x, ves], dim=1)
            # x = conv_layer(x.cpu())
            # x = x.cuda()
            # if i == 1:
            #     x = x * mask_128.expand_as(x) + x
            if len(encoder) == 1:
                # x = encoder[0](x)
                x = encoder[0](x, kernel)
                kernel = kernel_down(kernel)
            else:
                x = encoder(x)

            encs.append(x)
            x = down(x)
            # ves = down(ves)

        x = self.middle_blks(x)
        # mask_16 = self.a_pool_16(x2)
        #
        # x = x * mask_16.expand_as(x) + x

        for decoder, up, enc_skip, conv_layer2 in zip(self.decoders, self.ups, encs[::-1], self.conv_layers_u):
            x = up(x)
            x = x + enc_skip

            x = decoder(x)

        if self.drop_flag:
            x = self.dropout(x)

        # x = x * x2.expand_as(x) + ves_s

        x = self.ending(x)
        x = x + inp
        x = torch.clamp(x, 0.0, 1.0)

        # x = F.interpolate(x, size=(512, 512), mode='bilinear', align_corners=False)

        return x

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), mode='reflect')
        return x


class UFPNet_code_uncertainty_Local(Local_Base, UFPNet_code_uncertainty):
    def __init__(self, *args, train_size=(1, 3, 256, 256), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        UFPNet_code_uncertainty.__init__(self, *args, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)


if __name__ == '__main__':
    img_channel = 3
    width = 32
    enc_blks = [1, 1, 1, 28]
    middle_blk_num = 1
    dec_blks = [1, 1, 1, 1]

    # net = NAFNet_wDetHead(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
    #                       enc_blk_nums=enc_blks, dec_blk_nums=dec_blks, global_residual=True,
    #                       concat=True, merge_manner=2)  # .cuda()
    net = UFPNet_code_uncertainty(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                                  enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)
    input = torch.randn([4, 3, 256, 256])  # .cuda()  inp_shape = (5, 3, 128, 128)
    spare = torch.randn([4, 1, 256, 256])
    mask = torch.randn([4, 1, 256, 256])
    print(net(input, spare).size())
