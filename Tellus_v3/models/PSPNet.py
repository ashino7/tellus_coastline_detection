import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F


class PSPNet(nn.Module):

    def __init__(self, n_class, width, height):
        super(PSPNet, self).__init__()
        self.n_class = n_class
        self.n_classes = n_class
        self.width = width
        self.height = height
        self.feature_conv = FeatureMap_convolution()
        self.feature_res_1 = ResidualBlockPSP(n_blocks=3, in_channels=128, mid_channels=64, out_channels=256,
                                              stride=1, dilation=1)
        self.feature_res_2 = ResidualBlockPSP(n_blocks=4, in_channels=256, mid_channels=128, out_channels=512,
                                              stride=2, dilation=1)
        self.feature_dilated_res_1 = ResidualBlockPSP(n_blocks=6, in_channels=512, mid_channels=256, out_channels=1024
                                                      ,stride=1, dilation=2)
        self.feature_dilated_res_2 = ResidualBlockPSP(n_blocks=3, in_channels=1024, mid_channels=512, out_channels=2048,
                                                      stride=1, dilation=4)
        self.pyramid_pooling = PyramidPooling(in_channels=2048, pool_sizes=[6, 3, 2, 1], height=30, width=40)
        self.decode_feature = DecodePSPFeature(height=self.height, width=self.width, n_class=self.n_class)
        self.aux = AuxilaryPSPlayers(in_channels=1024, height=self.height, width=self.width, n_class=self.n_class)

    def forward(self, x):
        x = self.feature_conv(x)
        x = self.feature_res_1(x)
        x = self.feature_res_2(x)
        x = self.feature_dilated_res_1(x)
        output_aux = self.aux(x)
        x = self.feature_dilated_res_2(x)
        x = self.pyramid_pooling(x)
        output = self.decode_feature(x)
        return output
        # return [output, output_aux]

class PSPModule(nn.Module):

    def __init__(self, channels, out_channels,  sizes=(1, 2, 3, 6)):
        super(PSPModule, self).__init__()
        self.stages = nn.ModuleList([self._make_stage(channels, size) for size in sizes])
        self.bottleneck = nn.Conv2d(channels * (len(sizes) + 1), out_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def _make_stage(self, channels, size):
        prior = nn.AdaptiveAvgPool2d((size, size))
        conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, x):
        priors = [F.upsample(input=stage(x), size=(x.shape[2], x.shape[3]), mode='bilinear') for stage in self.stages]
        priors += [x]
        cat = torch.cat(priors, dim=1)
        bottle = self.bottleneck(cat)
        x = self.relu(bottle)
        return x


class conv2DBatchNormRelu(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, bias):
        super(conv2DBatchNormRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.batchnorm(self.conv(x)))


class FeatureMap_convolution(nn.Module):

    def __init__(self):
        super(FeatureMap_convolution, self).__init__()
        self.cbnr_1 = conv2DBatchNormRelu(3, 64, 3, 2, 1, 1, False)
        self.cbnr_2 = conv2DBatchNormRelu(64, 64, 3, 1, 1, 1, False)
        self.cbnr_3 = conv2DBatchNormRelu(64, 128, 3, 1, 1, 1, False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.cbnr_1(x)
        x = self.cbnr_2(x)
        x = self.cbnr_3(x)
        return self.maxpool(x)

class ResidualBlockPSP(nn.Sequential):

    def __init__(self, n_blocks, in_channels, mid_channels, out_channels, stride, dilation):
        super(ResidualBlockPSP, self).__init__()

        self.add_module(
            "block1",
            bottleNeckPSP(in_channels, mid_channels, out_channels, stride, dilation)
        )

        # bottleNeckIdentifyPSP
        for i in range(n_blocks - 1):
            self.add_module(
                "block" + str(i+2),
                bottleNeckIdentifyPSP(out_channels, mid_channels, stride, dilation)
            )


class conv2DBatchNorm(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, bias):
        super(conv2DBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        return self.batchnorm(x)


class bottleNeckPSP(nn.Module):

    def __init__(self, in_channels, mid_channels, out_channels, stride, dilation):
        super(bottleNeckPSP, self).__init__()
        self.cbr_1 = conv2DBatchNormRelu(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)
        self.cbr_2 = conv2DBatchNormRelu(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.cb_3 = conv2DBatchNorm(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)

        # skip
        self.cb_residual = conv2DBatchNorm(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, dilation=1,
                                           bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.cb_3(self.cbr_2(self.cbr_1(x)))
        residual = self.cb_residual(x)
        return self.relu(conv + residual)


class bottleNeckIdentifyPSP(nn.Module):
    def __init__(self, in_channels, mid_channels, stride, dilation):
        super(bottleNeckIdentifyPSP, self).__init__()
        self.cbr_1 = conv2DBatchNormRelu(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)
        self.cbr_2 = conv2DBatchNormRelu(mid_channels, mid_channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation,
                                         bias=False)
        self.cb_3 = conv2DBatchNorm(mid_channels, in_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.cb_3(self.cbr_2(self.cbr_1(x)))
        residual = x
        return self.relu(conv + residual)


class PyramidPooling(nn.Module):
    def __init__(self, in_channels, pool_sizes, height, width):
        super(PyramidPooling, self).__init__()
        self.height = height
        self.width = width

        out_channels = int(in_channels / len(pool_sizes))

        self.avpool_1 = nn.AdaptiveAvgPool2d(output_size=pool_sizes[0])
        self.cbr_1 = conv2DBatchNormRelu(in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                                         dilation=1, bias=False)
        self.avpool_2 = nn.AdaptiveAvgPool2d(output_size=pool_sizes[1])
        self.cbr_2 = conv2DBatchNormRelu(in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                                         dilation=1, bias=False)

        self.avpool_3 = nn.AdaptiveAvgPool2d(output_size=pool_sizes[2])
        self.cbr_3 = conv2DBatchNormRelu(in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                                         dilation=1, bias=False)

        self.avpool_4 = nn.AdaptiveAvgPool2d(output_size=pool_sizes[3])
        self.cbr_4 = conv2DBatchNormRelu(in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                                         dilation=1, bias=False)

    def forward(self, x):
        out1 = self.cbr_1(self.avpool_1(x))
        out1 = F.interpolate(out1, size=(self.height, self.width), mode='bilinear', align_corners=True)

        out2 = self.cbr_2(self.avpool_2(x))
        out2 = F.interpolate(out2, size=(self.height, self.width), mode='bilinear', align_corners=True)

        out3 = self.cbr_3(self.avpool_3(x))
        out3 = F.interpolate(out3, size=(self.height, self.width), mode='bilinear', align_corners=True)

        out4 = self.cbr_4(self.avpool_4(x))
        out4 = F.interpolate(out4, size=(self.height, self.width), mode='bilinear', align_corners=True)

        output = torch.cat([x, out1, out2, out3, out4], dim=1)
        return output


class DecodePSPFeature(nn.Module):
    def __init__(self, height, width, n_class):
        super(DecodePSPFeature, self).__init__()
        self.height = height
        self.width = width
        self.cbr = conv2DBatchNormRelu(in_channels=4096, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1,
                                       bias=False)
        self.dropout = nn.Dropout2d(p=0.1)
        self.classification = nn.Conv2d(in_channels=512, out_channels=n_class, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.classification(self.dropout(self.cbr(x)))
        output = F.interpolate(x, size=(self.height, self.width), mode='bilinear', align_corners=True)

        return output


class AuxilaryPSPlayers(nn.Module):
    def __init__(self, in_channels, height, width, n_class):
        super(AuxilaryPSPlayers, self).__init__()
        self.height = height
        self.width = width

        self.cbr = conv2DBatchNormRelu(in_channels=in_channels, out_channels=256, kernel_size=3, stride=1,
                                       padding=1, dilation=1, bias=False)
        self.dropout = nn.Dropout2d(p=0.1)
        self.classification = nn.Conv2d(in_channels=256, out_channels=n_class, kernel_size=1, stride=1,
                                        padding=0)

    def forward(self, x):
        x = self.classification(self.dropout(self.cbr(x)))
        output = F.interpolate(x, size=(self.height, self.width), mode='bilinear', align_corners=True)

        return output