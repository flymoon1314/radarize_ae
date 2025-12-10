#!/usr/bin/env python3

import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class EfficientChannelAttention(nn.Module):  # Efficient Channel Attention module
    """
    ECA通道注意力机制
    """
    def __init__(self, c, b=1, gamma=2):
        """
        :param self: 说明
        :param c: 输入张量 x 的通道数
        :param b: 用于计算中间变量 t 的常数
        :param gamma: 一个缩放因子，用于计算卷积核的大小
        """
        super(EfficientChannelAttention, self).__init__()
        # 卷积核大小的计算
        t = int(abs((math.log(c, 2) + b) / gamma))
        k = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 核心就是这个1D卷积层
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k, padding=int(k / 2), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.conv1(x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        out = self.sigmoid(x)
        return out


class BasicBlock(nn.Module):  # 左侧的 residual block 结构（18-layer、34-layer）
    """
    ResNet 的一种基本残差块结构，通常用于 18 层、34 层 ResNet 等网络中。
    它包含两个卷积层、批归一化、激活函数，并使用了通道注意力模块(就是上面那个方法)，以及跳跃连接
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):  # 两层卷积 Conv2d + Shutcuts
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.channel = EfficientChannelAttention(
            planes
        )  # Efficient Channel Attention module

        self.shortcut = nn.Sequential()
        if (
            stride != 1 or in_planes != self.expansion * planes
        ):  # Shutcuts用于构建 Conv Block 和 Identity Block
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))  # 第一个卷积 + ReLU
        out = self.bn2(self.conv2(out))  # 第二个卷积
        ECA_out = self.channel(out)  # 通道注意力
        out = out * ECA_out  # 应用通道注意力
        out += self.shortcut(x)  # 跳跃连接（残差连接）
        out = F.relu(out)  # ReLU 激活
        return out


class ECAResNet18(nn.Module):
    """
    基于 ResNet-18 结构，并结合了 ECA 模块
    """
    def __init__(self, n_channels, n_outputs):
        super(ECAResNet18, self).__init__()
        self.in_planes = 64
        num_blocks = [2, 2, 2, 2]
        block = BasicBlock

        self.conv1 = nn.Conv2d(
            n_channels,
            64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )  # conv1
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)  # conv2_x
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)  # conv3_x
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)  # conv4_x
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)  # conv5_x
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, 32)
        self.fc = FcBlock(32, n_outputs)
        # 权重初始化
        weight_init(self)

    def _make_layer(self, block, planes, num_blocks, stride):
        """
        构建残差块层
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        x = self.fc(x)
        return x


class FcBlock(nn.Module):
    """
    全连接层的设置
    """
    def __init__(self, in_dim, out_dim, mid_dim=256, dropout=0.05):
        super(FcBlock, self).__init__()
        self.mid_dim = mid_dim
        self.in_dim = in_dim
        self.out_dim = out_dim

        # fc layers
        self.fcs = nn.Sequential(
            nn.Linear(self.in_dim, self.mid_dim),
            nn.ReLU(True),
            nn.Linear(self.mid_dim, self.out_dim),
        )

    def forward(self, x):
        # x = x.view(x.size(0), -1)
        x = self.fcs(x)
        return x


def weight_init(m):
    """
    参数初始化
    Usage:
        model = Model()
        model.apply(weight_init)
    """
    # 使用正态分布初始化1d卷积层和偏置（若有）权重
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    # 使用Xavier正态分布初始化2d和3d卷积层的权重，偏置依旧正态分布
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    # 转置卷积核上方一致
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    # 批归一化层使用正态分布初始化权重，均值为 1，标准差为 0.02，偏置被初始化为 0
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    # 线性层使用 Xavier 正态分布初始化权重，偏置使用正态分布初始化
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    # 循环神经网络层
    # 对于具有至少 2 个维度的参数（如权重矩阵），
    # 使用正交初始化
    # 对于一维的参数（如偏置），则使用正态分布初始化
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)


class ResNet34(nn.Module):
    def __init__(self, n_channels, n_outputs):
        super(ResNet34, self).__init__()

        self.resnet34 = models.resnet34(pretrained=True)
        self.resnet34.conv1 = nn.Conv2d(
            n_channels,
            64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )
        self.resnet34.fc = nn.Linear(512, n_outputs)

    def forward(self, x):
        return self.resnet34(x)


class ResNet18(nn.Module):
    """Model to predict x and y flow from radar heatmaps."""

    def __init__(self, n_channels, n_outputs):
        super(ResNet18, self).__init__()

        self.resnet18 = models.resnet18(pretrained=False)
        self.resnet18.conv1 = nn.Conv2d(
            n_channels,
            64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )
        self.resnet18.fc = nn.Linear(512, n_outputs)

        # self.resnet18.layer1[0].conv1 = nn.Conv2d(64, 64, kernel_size=(3, 3), dilation=2, padding=(2,2))
        # self.resnet18.layer1[0].conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), dilation=2, padding=(2,2))
        # self.resnet18.layer1[1].conv1 = nn.Conv2d(64, 64, kernel_size=(3, 3), dilation=2, padding=(2,2))
        # self.resnet18.layer1[1].conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), dilation=2, padding=(2,2))

        # self.resnet18.layer2[0].conv1 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=2, dilation=2, padding=(2,2))
        # self.resnet18.layer2[0].conv2 = nn.Conv2d(128, 128, kernel_size=(3, 3), dilation=2, padding=(2,2))
        # self.resnet18.layer2[1].conv1 = nn.Conv2d(128, 128, kernel_size=(3, 3), dilation=2, padding=(2,2))
        # self.resnet18.layer2[1].conv2 = nn.Conv2d(128, 128, kernel_size=(3, 3), dilation=2, padding=(2,2))

        # self.resnet18.layer3[0].conv1 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=2, dilation=2, padding=(2,2))
        # self.resnet18.layer3[0].conv2 = nn.Conv2d(256, 256, kernel_size=(3, 3), dilation=2, padding=(2,2))
        # self.resnet18.layer3[1].conv1 = nn.Conv2d(256, 256, kernel_size=(3, 3), dilation=2, padding=(2,2))
        # self.resnet18.layer3[1].conv2 = nn.Conv2d(256, 256, kernel_size=(3, 3), dilation=2, padding=(2,2))

        # self.resnet18.layer4[0].conv1 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=2, dilation=2, padding=(2,2))
        # self.resnet18.layer4[0].conv2 = nn.Conv2d(512, 512, kernel_size=(3, 3), dilation=2, padding=(2,2))
        # self.resnet18.layer4[1].conv1 = nn.Conv2d(512, 512, kernel_size=(3, 3), dilation=2, padding=(2,2))
        # self.resnet18.layer4[1].conv2 = nn.Conv2d(512, 512, kernel_size=(3, 3), dilation=2, padding=(2,2))

        # print(self.resnet18)

        weight_init(self)

    def forward(self, x):
        out = self.resnet18(x)
        return out


class ResNet50(nn.Module):
    """Model to predict x and y flow from radar heatmaps."""

    def __init__(self, n_channels, n_outputs):
        super(ResNet50, self).__init__()

        # CNN encoder for heatmaps
        self.resnet50 = models.resnet50(pretrained=True)
        self.resnet50.conv1 = nn.Conv2d(
            n_channels,
            64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )
        self.resnet50.fc = nn.Linear(2048, n_outputs)

    def forward(self, x):
        out = self.resnet50(x)
        return out


class ResNet18Nano(nn.Module):
    """Model to predict x and y flow from radar heatmaps."""

    def __init__(self, n_channels, n_outputs):
        super(ResNet18Nano, self).__init__()

        # CNN encoder for48eatmaps
        resnet18 = models.resnet._resnet(
            "resnet18",
            models.resnet.BasicBlock,
            [1, 1, 1, 1],
            pretrained=False,
            progress=False,
        )
        resnet18.conv1 = nn.Conv2d(
            n_channels,
            64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )
        self.enc = nn.Sequential(OrderedDict(list(resnet18.named_children())[:5]))
        self.avgpool = resnet18.avgpool
        self.fc = nn.Linear(64, n_outputs)

    def init_weights(self):
        for m in self.modules():
            m.apply(weight_init)

    def forward(self, x):
        x = self.enc(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class ResNet18Micro(nn.Module):
    """Model to predict x and y flow from radar heatmaps."""

    def __init__(self, n_channels, n_outputs):
        super(ResNet18Micro, self).__init__()

        # CNN encoder for48eatmaps
        resnet18 = models.resnet._resnet(
            "resnet18",
            models.resnet.BasicBlock,
            [1, 1, 1, 1],
            pretrained=False,
            progress=False,
        )
        resnet18.conv1 = nn.Conv2d(
            n_channels,
            64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )
        self.enc = nn.Sequential(OrderedDict(list(resnet18.named_children())[:6]))
        self.avgpool = resnet18.avgpool
        self.fc = nn.Linear(128, n_outputs)

    def forward(self, x):
        x = self.enc(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

