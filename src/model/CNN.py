# -*-coding: utf-8 -*-

import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x))


class CBAM(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, ratio)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


class CNN(nn.Module):
    """
    CBAM-CNN feature extractor.。
    """

    def __init__(self, num_classes: int = 5):
        super().__init__()
        in_channels = 1

        # Block 1
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=5, padding="same")
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding="same")
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.cbam1 = CBAM(64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(p=0.3)

        # Block 2
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, padding="same")
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(128, 64, kernel_size=5, padding="same")
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU(inplace=True)
        self.cbam2 = CBAM(64)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(p=0.3)

        # Block 3
        self.conv5 = nn.Conv2d(64, 32, kernel_size=3, padding="same")
        self.bn5 = nn.BatchNorm2d(32)
        self.relu5 = nn.ReLU(inplace=True)
        self.conv6 = nn.Conv2d(32, 32, kernel_size=3, padding="same")
        self.bn6 = nn.BatchNorm2d(32)
        self.relu6 = nn.ReLU(inplace=True)
        self.cbam3 = CBAM(32)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()

    def forward(self, x):
        # Block 1
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.cbam1(x)
        x = self.dropout1(self.maxpool1(x))
        # Block 2
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.relu4(self.bn4(self.conv4(x)))
        x = self.cbam2(x)
        x = self.dropout2(self.maxpool2(x))
        # Block 3
        x = self.relu5(self.bn5(self.conv5(x)))
        x = self.relu6(self.bn6(self.conv6(x)))
        x = self.cbam3(x)
        x = self.maxpool3(x)
        return self.flatten(x)