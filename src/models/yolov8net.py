# yolov8net.py
### Modified darknet2pytorch.py to support YOLOv8 ###

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))

class CSPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks):
        super(CSPBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.blocks = nn.Sequential(*[ConvBlock(out_channels, out_channels, kernel_size=3, stride=1, padding=1) for _ in range(num_blocks)])
        self.conv2 = ConvBlock(2 * out_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        residual = self.conv1(x)
        x = self.blocks(residual)
        x = torch.cat([residual, x], dim=1)
        return self.conv2(x)

class YOLOv8Backbone(nn.Module):
    def __init__(self, num_classes):
        super(YOLOv8Backbone, self).__init__()
        # Main backbone blocks of YOLOv8
        # 降低每层的输出通道数以减少显存占用
        self.conv1 = ConvBlock(3, 16, kernel_size=3, stride=1, padding=1)  # 原来是32
        self.csp1 = CSPBlock(16, 32, num_blocks=1)  # 原来是64
        self.csp2 = CSPBlock(32, 64, num_blocks=1)  # 原来是128，缩减 num_blocks
        self.csp3 = CSPBlock(64, 128, num_blocks=1)  # 原来是256，缩减 num_blocks
        self.csp4 = CSPBlock(128, 256, num_blocks=1)  # 原来是512，缩减 num_blocks

        self.head = nn.Sequential(
            ConvBlock(256, 256, kernel_size=3, stride=1, padding=1),  # 原来是512
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1)  # 输出类别数量
        )

    def forward(self, x, targets=None):
        # Forward through backbone
        x = self.conv1(x)
        x = self.csp1(x)
        x = self.csp2(x)
        x = self.csp3(x)
        x = self.csp4(x)
        outputs = self.head(x)

        # Here you can add the calculation for loss if targets are provided
        if self.training and targets is not None:
            # Calculate the loss here, assuming you have a loss function defined elsewhere
            # Example:
            loss = self.compute_loss(outputs, targets)
            return loss, outputs

        return outputs

    def compute_loss(self, outputs, targets):
        # Placeholder for the actual loss calculation
        # You need to implement this function based on your loss function
        # For example, you could use a YOLO loss that combines classification, localization, and confidence losses
        loss = torch.tensor(0.0)  # Replace with actual loss calculation
        return loss
