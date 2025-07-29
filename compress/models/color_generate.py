import torch
import torch.nn as nn
from torchvision.ops import DeformConv2d
from compress.layers import conv3x3, subpel_conv3x3, Win_noShift_Attention
from .utils import deconv

class Generate(nn.Module):
    def __init__(self, N):
        super(Generate, self).__init__()
        self.group = 1
        self.kernel_size = (3, 3)
        self.conv5 = nn.Conv2d(in_channels=1, out_channels=N, kernel_size=3, stride=2, padding=1)
        self.conv6 = nn.Conv2d(in_channels=N, out_channels=N, kernel_size=3, stride=2, padding=1)
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=N, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=N, out_channels=N, kernel_size=3, stride=1, padding=1)
        self.att = Win_noShift_Attention(dim=N*2, num_heads=8, window_size=8, shift_size=4)
        self.conv3 = deconv(in_channels=N*2, out_channels=N, kernel_size=3, stride=2)
        self.conv4 = deconv(in_channels=N, out_channels=2, kernel_size=3, stride=2)

    def forward(self, Y, UV):
        def_features = self.conv6(self.conv5(Y))
        uv_feature = self.conv2(self.conv1(UV))
        feature = self.att(torch.concat((def_features, uv_feature), dim=1))
        out = self.conv4(self.conv3(feature))
        return out