import torch
import torch.nn as nn
from torchvision.ops import DeformConv2d
from compress.layers import conv3x3, subpel_conv3x3, Win_noShift_Attention
from .layers import ResidualBlockUpsample, ResidualBlock, ResidualBlockWithStride
from .utils import deconv

# class Generate(nn.Module):
#     def __init__(self, N):
#         super(Generate, self).__init__()
#         self.group = 1
#         self.kernel_size = (3, 3)
#         self.head1 = nn.Sequential(
#             nn.Conv2d(in_channels=1,
#                       out_channels=2 * self.kernel_size[0] * self.kernel_size[1] * self.group,
#                       kernel_size=(3, 3), stride=1, padding=1)
#             # nn.ReLU(inplace=True)
#         )
#         self.head2 = nn.Sequential(
#             nn.Conv2d(in_channels=64,
#                       out_channels=2 * self.kernel_size[0] * self.kernel_size[1] * self.group,
#                       kernel_size=(3, 3), stride=1, padding=1)
#             # nn.ReLU(inplace=True)
#         )
#         self.deform_conv1 = DeformConv2d(in_channels=1, out_channels=N, kernel_size=3, stride=1, padding=1)
#         self.deform_conv2 = DeformConv2d(in_channels=N, out_channels=N, kernel_size=3, stride=1, padding=1)
#         self.conv5 = nn.Conv2d(in_channels=N, out_channels=N, kernel_size=3, stride=2, padding=1)
#         self.conv6 = nn.Conv2d(in_channels=N, out_channels=N, kernel_size=3, stride=2, padding=1)
#         self.conv1 = nn.Conv2d(in_channels=2, out_channels=N, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(in_channels=N, out_channels=N, kernel_size=3, stride=1, padding=1)
#         self.att = Win_noShift_Attention(dim=N*2, num_heads=8, window_size=8, shift_size=4)
#         self.conv3 = deconv(in_channels=N*2, out_channels=N, kernel_size=3, stride=2)
#         self.conv4 = deconv(in_channels=N, out_channels=2, kernel_size=3, stride=2)

#     def forward(self, Y, UV):
#         offsets = self.head1(Y)
#         def_features = self.deform_conv1(Y, offsets)
#         def_features = self.conv5(def_features)
#         offsets = self.head2(def_features)
#         def_features = self.deform_conv2(def_features, offsets)
#         def_features = self.conv6(def_features)
#         uv_feature = self.conv2(self.conv1(UV))
#         feature = self.att(torch.concat((def_features, uv_feature), dim=1))
#         out = self.conv4(self.conv3(feature))
#         return out
    
# class Generate(nn.Module):
#     def __init__(self, N):
#         super(Generate, self).__init__()
#         self.group = 1
#         self.kernel_size = (3, 3)
#         self.head1 = nn.Sequential(
#             nn.Conv2d(in_channels=N*2,
#                       out_channels=2 * self.kernel_size[0] * self.kernel_size[1] * self.group,
#                       kernel_size=(3, 3), stride=1, padding=1)
            
#         )
#         self.head2 = nn.Sequential(
#             nn.Conv2d(in_channels=N*2,
#                       out_channels=2 * self.kernel_size[0] * self.kernel_size[1] * self.group,
#                       kernel_size=(3, 3), stride=1, padding=1)
            
#         )
#         self.Y_conv1 = nn.Sequential(
#             nn.Conv2d(in_channels=1, out_channels=N, kernel_size=3, stride=1, padding=1),
#             nn.Conv2d(in_channels=N, out_channels=N, kernel_size=3, stride=1, padding=1),
#         )
        
#         self.deform_conv1 = DeformConv2d(in_channels=N, out_channels=N, kernel_size=3, stride=1, padding=1, groups= self.group)
#         self.Y_down1 = nn.Conv2d(in_channels=N, out_channels=N, stride=2, kernel_size=2)
#         self.Y_conv2 = nn.Conv2d(in_channels=N, out_channels=N, kernel_size=3, stride=1, padding=1)
#         self.deform_conv2 = DeformConv2d(in_channels=N, out_channels=N, kernel_size=3, stride=1, padding=1, groups= self.group)
#         self.Y_down2 = nn.Conv2d(in_channels=N, out_channels=N, stride=2, kernel_size=2)
#         self.uv_conv1 = nn.Sequential(
#             nn.Conv2d(in_channels=2, out_channels=N, kernel_size=3, stride=1, padding=1),
#             nn.Conv2d(in_channels=N, out_channels=N, kernel_size=3, stride=1, padding=1)
#         )
        
#         self.uv_conv2 = nn.Conv2d(in_channels=N, out_channels=N, kernel_size=3, stride=1, padding=1)
#         self.pixel_shuffle1 = nn.Sequential(
#             deconv(in_channels=N, out_channels=N, stride=2),
#             deconv(in_channels=N, out_channels=N, stride=2)
#         )
#         self.pixel_shuffle2 = nn.Sequential(
#             deconv(in_channels=N, out_channels=N, stride=2)
#         )
#         self.att = Win_noShift_Attention(dim=N*2, num_heads=8, window_size=8, shift_size=4)
#         self.res = nn.Sequential(
#             nn.Conv2d(in_channels=N*2, out_channels=N*2, kernel_size=3, stride=1, padding=1),
#             nn.Conv2d(in_channels=N*2, out_channels=N*2, kernel_size=3, stride=1, padding=1),
#             deconv(in_channels=N*2, out_channels=N*2, stride=2),
#             nn.Conv2d(in_channels=N*2, out_channels=N*2, kernel_size=3, stride=1, padding=1),
#             deconv(in_channels=N*2, out_channels=N*2, stride=2),
#             nn.Conv2d(in_channels=N*2, out_channels=2, kernel_size=3, stride=1, padding=1),
#         )

#     def forward(self, Y, UV):
#         uv_feature1 = self.uv_conv1(UV)
#         uv_feature = self.uv_conv2(uv_feature1)
#         y_fea = self.Y_conv1(Y)
#         offsets = self.head1(torch.cat((y_fea,self.pixel_shuffle1(uv_feature1)), dim = 1))
#         def_features = self.Y_down1(self.deform_conv1(y_fea, offsets))
#         def_features = self.Y_conv2(def_features)
#         offsets = self.head2(torch.cat((def_features,self.pixel_shuffle2(uv_feature)), dim = 1))
#         def_features = self.Y_down2(self.deform_conv2(def_features, offsets))
#         feature = self.att(torch.concat((def_features, uv_feature), dim=1))
#         out = self.res(feature)
#         return out

# class Generate(nn.Module):
#     def __init__(self, N):
#         super(Generate, self).__init__()
#         self.group = 1
#         self.kernel_size = (3, 3)
#         self.head1 = nn.Sequential(
#             nn.Conv2d(in_channels=1,
#                       out_channels=2 * self.kernel_size[0] * self.kernel_size[1] * self.group,
#                       kernel_size=(3, 3), stride=1, padding=1)
#             # nn.ReLU(inplace=True)
#         )
#         self.head2 = nn.Sequential(
#             nn.Conv2d(in_channels=64,
#                       out_channels=2 * self.kernel_size[0] * self.kernel_size[1] * self.group,
#                       kernel_size=(3, 3), stride=1, padding=1)
#             # nn.ReLU(inplace=True)
#         )
#         self.deform_conv1 = DeformConv2d(in_channels=1, out_channels=N, kernel_size=3, stride=1, padding=1)
#         self.deform_conv2 = DeformConv2d(in_channels=N, out_channels=N, kernel_size=3, stride=1, padding=1)
#         self.conv5 = nn.Conv2d(in_channels=N, out_channels=N, kernel_size=3, stride=2, padding=1)
#         self.conv6 = nn.Conv2d(in_channels=N, out_channels=N, kernel_size=3, stride=2, padding=1)
#         self.conv1 = nn.Conv2d(in_channels=2, out_channels=N, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(in_channels=N, out_channels=N, kernel_size=3, stride=1, padding=1)
#         self.att = Win_noShift_Attention(dim=N*2, num_heads=8, window_size=8, shift_size=4)
#         # self.conv3 = deconv(in_channels=N*2, out_channels=N, kernel_size=3, stride=2)
#         self.conv3 = ResidualBlockUpsample(in_ch=N*2, out_ch=N*2)
#         # self.conv4 = deconv(in_channels=N, out_channels=2, kernel_size=3, stride=2)
#         self.conv4 = ResidualBlockUpsample(in_ch=N*2, out_ch=2)

#     def forward(self, Y, UV):
#         offsets = self.head1(Y)
#         def_features = self.deform_conv1(Y, offsets)
#         def_features = self.conv5(def_features)
#         offsets = self.head2(def_features)
#         def_features = self.deform_conv2(def_features, offsets)
#         def_features = self.conv6(def_features)
#         uv_feature = self.conv2(self.conv1(UV))
#         feature = self.att(torch.concat((def_features, uv_feature), dim=1))
#         out = self.conv4(self.conv3(feature))
#         return out

# class ConvFNN(nn.Module):
#     def __init__(self, N):
#         super(ConvFNN, self).__init__()
#         self.bn = nn.BatchNorm2d
#         self.conv1 = nn.Conv2d(in_channels=N, out_channels=N, kernel_size=1, stride=1)
#         self.deep
import torch.nn.functional as f

class EfficientAttention(nn.Module):

    def __init__(self, in_channels, key_channels, head_count, value_channels):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels

        self.keys = nn.Conv2d(in_channels, key_channels, 1)
        self.queries = nn.Conv2d(in_channels, key_channels, 1)
        self.values = nn.Conv2d(in_channels, value_channels, 1)
        self.reprojection = nn.Conv2d(value_channels, in_channels, 1)

    def forward(self, input_):
        n, _, h, w = input_.size()
        keys = self.keys(input_).reshape((n, self.key_channels, h * w))
        queries = self.queries(input_).reshape(n, self.key_channels, h * w)
        values = self.values(input_).reshape((n, self.value_channels, h * w))
        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count

        attended_values = []
        for i in range(self.head_count):
            key = f.softmax(keys[
                            :,
                            i * head_key_channels: (i + 1) * head_key_channels,
                            :
                            ], dim=2)
            query = f.softmax(queries[
                              :,
                              i * head_key_channels: (i + 1) * head_key_channels,
                              :
                              ], dim=1)
            value = values[
                    :,
                    i * head_value_channels: (i + 1) * head_value_channels,
                    :
                    ]
            context = key @ value.transpose(1, 2)
            attended_value = (
                    context.transpose(1, 2) @ query
            ).reshape(n, head_value_channels, h, w)
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)
        reprojected_value = self.reprojection(aggregated_values)
        attention = reprojected_value + input_

        return attention


class Generate(nn.Module):
    def __init__(self, N):
        super(Generate, self).__init__()
        self.group = 1
        self.kernel_size = (3, 3)
        self.head1 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=2 * self.kernel_size[0] * self.kernel_size[1] * self.group,
                      kernel_size=(4, 4), stride=4)
            # nn.ReLU(inplace=True)
        )
        self.head2 = nn.Sequential(
            nn.Conv2d(in_channels=N,
                      out_channels=2 * self.kernel_size[0] * self.kernel_size[1] * self.group,
                      kernel_size=(2, 2), stride=2, padding=0)
            # nn.ReLU(inplace=True)
        )
        self.deform_conv1 = DeformConv2d(in_channels=2, out_channels=N, kernel_size=3, stride=1, padding=1)
        # self.uv_res1 = ResidualBlock(in_ch=N, out_ch=N)
        self.uv_res1 = nn.Conv2d(in_channels=N, out_channels=N, stride=1, kernel_size=3, padding=1)
        self.deform_conv2 = DeformConv2d(in_channels=N, out_channels=N, kernel_size=3, stride=1, padding=1)
        # self.uv_res2 = ResidualBlock(in_ch=N, out_ch=N)
        self.uv_res2 = nn.Conv2d(in_channels=N, out_channels=N, stride=1, kernel_size=3, padding=1)
        self.y_res1 = ResidualBlockWithStride(in_ch=1, out_ch=N)
        self.y_res2 = ResidualBlockWithStride(in_ch=N, out_ch=N)
        self.att = EfficientAttention(N*2, N*2, 1, N*2)
        self.img_res = nn.Sequential(
            ResidualBlockUpsample(in_ch=N*2, out_ch=N),
            ResidualBlock(in_ch=N, out_ch=N),
            ResidualBlockUpsample(in_ch=N, out_ch=N),
            ResidualBlock(in_ch=N, out_ch=2)
        )

    def forward(self, Y, UV):
        y_offset = self.head1(Y)
        uv_deform_fea = self.deform_conv1(UV, y_offset)
        uv_fea = self.uv_res1(uv_deform_fea)
        y_fea = self.y_res1(Y)
        y_offset = self.head2(y_fea)
        uv_deform_fea = self.deform_conv2(uv_fea, y_offset)
        uv_fea = self.uv_res2(uv_deform_fea)
        y_fea = self.y_res2(y_fea)
        img_fea = self.att(torch.cat((y_fea, uv_fea), dim=1))
        out = self.img_res(img_fea)
        return out

