import torch.nn as nn
import torch


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
    )


# adapted from https://github.com/usuyama/pytorch-unet/tree/master
class UNet(nn.Module):

    def __init__(self, n_channels, base_channel):
        super().__init__()

        self.dconv_down1 = double_conv(n_channels, base_channel)
        self.dconv_down2 = double_conv(base_channel, base_channel * 2)
        self.dconv_down3 = double_conv(base_channel * 2, base_channel * 4)
        self.dconv_down4 = double_conv(base_channel * 4, base_channel * 8)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.dconv_up3 = double_conv(base_channel * 12, base_channel * 4)
        self.dconv_up2 = double_conv(base_channel * 6, base_channel * 2)
        self.dconv_up1 = double_conv(base_channel * 3, base_channel)

        self.conv_last = nn.Conv2d(base_channel, n_channels, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)  # 256x256

        x = self.maxpool(conv1)  # 128x128
        conv2 = self.dconv_down2(x)

        x = self.maxpool(conv2)  # 64x64
        conv3 = self.dconv_down3(x)

        x = self.maxpool(conv3)  # 32x32
        bootle = self.dconv_down4(x)

        x = self.upsample(bootle)  # 64x64
        x = torch.cat([x, conv3], dim=1)
        up1 = self.dconv_up3(x)

        x = self.upsample(up1)  # 128x128
        x = torch.cat([x, conv2], dim=1)
        up2 = self.dconv_up2(x)

        x = self.upsample(up2)  # 256x256
        x = torch.cat([x, conv1], dim=1)
        up3 = self.dconv_up1(x)

        out = self.conv_last(up3)

        return out


class DoubleUnet(nn.Module):

    def __init__(self, n_channels, base_channel):
        super().__init__()

        self.unet1 = UNet(n_channels, base_channel)
        self.unet2 = UNet(n_channels, base_channel)

    def forward(self, x):
        out1 = self.unet1(x)
        out2 = self.unet2(out1)
        return out1, out2
