import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from utils.netUtils import get_kernel

"""
This code borrow from Deep image prior
"""

# class ListModule(nn.Module):
#     def __init__(self, *args):
#         super(ListModule, self).__init__()
#         idx = 0
#         for module in args:
#             self.add_module(str(idx), module)
#             idx += 1
#
#     def __getitem__(self, idx):
#         if idx >= len(self._modules):
#             raise IndexError('index {} is out of range'.format(idx))
#         if idx < 0:
#             idx = len(self) + idx
#
#         it = iter(self._modules.values())
#         for i in range(idx):
#             next(it)
#         return next(it)
#
#     def __iter__(self):
#         return iter(self._modules.values())
#
#     def __len__(self):
#         return len(self._modules)
#
# class UNet(nn.Module):
#     '''
#         upsample_mode in ['deconv', 'nearest', 'bilinear']
#         pad in ['zero', 'replication', 'none']
#     '''
#     def __init__(self, in_channels=3, out_channels=3,
#                        feature_scale=4, more_layers=0, concat_x=False,
#                        upsample_mode='deconv', pad='zero', norm_layer=nn.InstanceNorm2d, need_sigmoid=True, need_bias=True):
#         super(UNet, self).__init__()
#
#         self.feature_scale = feature_scale
#         self.more_layers = more_layers
#         self.concat_x = concat_x
#
#         filters = [64, 128, 256, 512, 1024]
#         filters = [x // self.feature_scale for x in filters]
#
#         self.start = unetConv2(in_channels, filters[0] if concat_x is False else filters[0] - in_channels, norm_layer, need_bias, pad)
#
#         self.down1 = unetDown(filters[0], filters[1] if concat_x is False else filters[1] - in_channels, norm_layer, need_bias, pad)
#         self.down2 = unetDown(filters[1], filters[2] if concat_x is False else filters[2] - in_channels, norm_layer, need_bias, pad)
#         self.down3 = unetDown(filters[2], filters[3] if concat_x is False else filters[3] - in_channels, norm_layer, need_bias, pad)
#         self.down4 = unetDown(filters[3], filters[4] if concat_x is False else filters[4] - in_channels, norm_layer, need_bias, pad)
#
#         # more downsampling layers
#         if self.more_layers > 0:
#             self.more_downs = [
#                 unetDown(filters[4], filters[4] if concat_x is False else filters[4] - in_channels , norm_layer, need_bias, pad) for i in range(self.more_layers)]
#             self.more_ups = [unetUp(filters[4], upsample_mode, need_bias, pad, same_num_filt =True) for i in range(self.more_layers)]
#
#             self.more_downs = ListModule(*self.more_downs)
#             self.more_ups   = ListModule(*self.more_ups)
#
#         self.up4 = unetUp(filters[3], upsample_mode, need_bias, pad)
#         self.up3 = unetUp(filters[2], upsample_mode, need_bias, pad)
#         self.up2 = unetUp(filters[1], upsample_mode, need_bias, pad)
#         self.up1 = unetUp(filters[0], upsample_mode, need_bias, pad)
#
#         self.final = conv(filters[0], out_channels, 1, bias=need_bias, pad=pad)
#
#         if need_sigmoid:
#             self.final = nn.Sequential(self.final, nn.Sigmoid())
#
#     def forward(self, inputs):
#
#         # Downsample
#         downs = [inputs]
#         down = nn.AvgPool2d(2, 2)
#         for i in range(4 + self.more_layers):
#             downs.append(down(downs[-1]))
#
#         in64 = self.start(inputs)
#         if self.concat_x:
#             in64 = torch.cat([in64, downs[0]], 1)
#
#         down1 = self.down1(in64)
#         if self.concat_x:
#             down1 = torch.cat([down1, downs[1]], 1)
#
#         down2 = self.down2(down1)
#         if self.concat_x:
#             down2 = torch.cat([down2, downs[2]], 1)
#
#         down3 = self.down3(down2)
#         if self.concat_x:
#             down3 = torch.cat([down3, downs[3]], 1)
#
#         down4 = self.down4(down3)
#         if self.concat_x:
#             down4 = torch.cat([down4, downs[4]], 1)
#
#         if self.more_layers > 0:
#             prevs = [down4]
#             for kk, d in enumerate(self.more_downs):
#                 # print(prevs[-1].size())
#                 out = d(prevs[-1])
#                 if self.concat_x:
#                     out = torch.cat([out,  downs[kk + 5]], 1)
#
#                 prevs.append(out)
#
#             up_ = self.more_ups[-1](prevs[-1], prevs[-2])
#             for idx in range(self.more_layers - 1):
#                 l = self.more_ups[self.more - idx - 2]
#                 up_= l(up_, prevs[self.more - idx - 2])
#         else:
#             up_= down4
#
#         up4= self.up4(up_, down3)
#         up3= self.up3(up4, down2)
#         up2= self.up2(up3, down1)
#         up1= self.up1(up2, in64)
#
#         return self.final(up1)
#
#
#
# class unetConv2(nn.Module):
#     def __init__(self, in_size, out_size, norm_layer, need_bias, pad):
#         super(unetConv2, self).__init__()
#
#         if norm_layer is not None:
#             self.conv1= nn.Sequential(conv(in_size, out_size, 3, bias=need_bias, pad=pad),
#                                        norm_layer(out_size),
#                                        nn.ReLU(),)
#             self.conv2= nn.Sequential(conv(out_size, out_size, 3, bias=need_bias, pad=pad),
#                                        norm_layer(out_size),
#                                        nn.ReLU(),)
#         else:
#             self.conv1= nn.Sequential(conv(in_size, out_size, 3, bias=need_bias, pad=pad),
#                                        nn.ReLU(),)
#             self.conv2= nn.Sequential(conv(out_size, out_size, 3, bias=need_bias, pad=pad),
#                                        nn.ReLU(),)
#     def forward(self, inputs):
#         outputs= self.conv1(inputs)
#         outputs= self.conv2(outputs)
#         return outputs
#
#
# class unetDown(nn.Module):
#     def __init__(self, in_size, out_size, norm_layer, need_bias, pad):
#         super(unetDown, self).__init__()
#         self.conv= unetConv2(in_size, out_size, norm_layer, need_bias, pad)
#         self.down= nn.MaxPool2d(2, 2)
#
#     def forward(self, inputs):
#         outputs= self.down(inputs)
#         outputs= self.conv(outputs)
#         return outputs
#
#
# class unetUp(nn.Module):
#     def __init__(self, out_size, upsample_mode, need_bias, pad, same_num_filt=False):
#         super(unetUp, self).__init__()
#
#         num_filt = out_size if same_num_filt else out_size * 2
#         if upsample_mode == 'deconv':
#             self.up= nn.ConvTranspose2d(num_filt, out_size, 4, stride=2, padding=1)
#             self.conv= unetConv2(out_size * 2, out_size, None, need_bias, pad)
#         elif upsample_mode=='bilinear' or upsample_mode=='nearest':
#             self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode=upsample_mode),
#                                    conv(num_filt, out_size, 3, bias=need_bias, pad=pad))
#             self.conv= unetConv2(out_size * 2, out_size, None, need_bias, pad)
#         else:
#             assert False
#
#     def forward(self, inputs1, inputs2):
#         in1_up= self.up(inputs1)
#
#         if (inputs2.size(2) != in1_up.size(2)) or (inputs2.size(3) != in1_up.size(3)):
#             diff2 = (inputs2.size(2) - in1_up.size(2)) // 2
#             diff3 = (inputs2.size(3) - in1_up.size(3)) // 2
#             inputs2_ = inputs2[:, :, diff2 : diff2 + in1_up.size(2), diff3 : diff3 + in1_up.size(3)]
#         else:
#             inputs2_ = inputs2
#
#         output= self.conv(torch.cat([in1_up, inputs2_], 1))
#
#         return output
#
#
# def conv(in_f, out_f, kernel_size, stride=1, bias=True, pad='zero', downsample_mode='stride'):
#     downsampler = None
#     if stride != 1 and downsample_mode != 'stride':
#
#         if downsample_mode == 'avg':
#             downsampler = nn.AvgPool2d(stride, stride)
#         elif downsample_mode == 'max':
#             downsampler = nn.MaxPool2d(stride, stride)
#         elif downsample_mode in ['lanczos2', 'lanczos3']:
#             downsampler = Downsampler(n_planes=out_f, factor=stride, kernel_type=downsample_mode, phase=0.5,
#                                       preserve_size=True)
#         else:
#             assert False
#
#         stride = 1
#
#     padder = None
#     to_pad = int((kernel_size - 1) / 2)
#     if pad == 'reflection':
#         padder = nn.ReflectionPad2d(to_pad)
#         to_pad = 0
#
#     convolver = nn.Conv2d(in_f, out_f, kernel_size, stride, padding=to_pad, bias=bias)
#
#     layers = filter(lambda x: x is not None, [padder, convolver, downsampler])
#     return nn.Sequential(*layers)
#
#
# class Downsampler(nn.Module):
#     '''
#         http://www.realitypixels.com/turk/computergraphics/ResamplingFilters.pdf
#     '''
#
#     def __init__(self, n_planes, factor, kernel_type, phase=0, kernel_width=None, support=None, sigma=None,
#                  preserve_size=False):
#         super(Downsampler, self).__init__()
#
#         assert phase in [0, 0.5], 'phase should be 0 or 0.5'
#
#         if kernel_type == 'lanczos2':
#             support = 2
#             kernel_width = 4 * factor + 1
#             kernel_type_ = 'lanczos'
#
#         elif kernel_type == 'lanczos3':
#             support = 3
#             kernel_width = 6 * factor + 1
#             kernel_type_ = 'lanczos'
#
#         elif kernel_type == 'gauss12':
#             kernel_width = 7
#             sigma = 1 / 2
#             kernel_type_ = 'gauss'
#
#         elif kernel_type == 'gauss1sq2':
#             kernel_width = 9
#             sigma = 1. / np.sqrt(2)
#             kernel_type_ = 'gauss'
#
#         elif kernel_type in ['lanczos', 'gauss', 'box']:
#             kernel_type_ = kernel_type
#
#         else:
#             assert False, 'wrong name kernel'
#
#         # note that `kernel width` will be different to actual size for phase = 1/2
#         self.kernel = get_kernel(factor, kernel_type_, phase, kernel_width, support=support, sigma=sigma)
#
#         downsampler = nn.Conv2d(n_planes, n_planes, kernel_size=self.kernel.shape, stride=factor, padding=0)
#         downsampler.weight.data[:] = 0
#         downsampler.bias.data[:] = 0
#
#         kernel_torch = torch.from_numpy(self.kernel)
#         for i in range(n_planes):
#             downsampler.weight.data[i, i] = kernel_torch
#
#         self.downsampler_ = downsampler
#
#         if preserve_size:
#
#             if self.kernel.shape[0] % 2 == 1:
#                 pad = int((self.kernel.shape[0] - 1) / 2.)
#             else:
#                 pad = int((self.kernel.shape[0] - factor) / 2.)
#
#             self.padding = nn.ReplicationPad2d(pad)
#
#         self.preserve_size = preserve_size
#
#     def forward(self, input):
#         if self.preserve_size:
#             x = self.padding(input)
#         else:
#             x = input
#         self.x = x
#         return self.downsampler_(x)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

