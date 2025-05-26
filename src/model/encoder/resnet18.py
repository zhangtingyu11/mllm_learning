import torch
import torch.nn as nn
import torch.nn.init as init
from .base_encoder import BaseEncoder
from torchvision.models.resnet import ResNet, BasicBlock

class ResNet18(BaseEncoder):
    def __init__(self, out_dim=128, remove_first_downsample=False):
        super().__init__()
        # 1. 初始化原始ResNet结构
        self.model = ResNet(
            block=BasicBlock,
            layers=[2, 2, 2, 2],
            num_classes=out_dim
        )

        if remove_first_downsample:
            self.model.conv1 = nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            )
            self.model.maxpool = nn.Identity()
        
        # 4. 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """自定义权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def get_fc_layer(self):
        return self.model.fc

    def forward(self, x):
        return self.model(x)
