# import torch
# import torch.nn as nn
# from torchvision.ops import DeformConv2d
# from .layers import ResidualBlock, AttentionBlock, ResidualDenseBlock_out
# from compress.layers import conv3x3, Win_noShift_Attention
# from .utils import deconv
# import torch.nn.functional as f
#
# class INV_block(nn.Module):
#     def __init__(self, subnet_constructor=conv3x3, clamp=2.0, in_c=3, out_c=3):
#         super().__init__()
#
#         self.split_len1 = in_c
#         self.split_len2 = in_c
#         self.clamp = clamp
#         # ρ
#         self.r = subnet_constructor(in_c, out_c)
#         # η
#         self.y = subnet_constructor(in_c, out_c)
#         # φ
#         self.f = subnet_constructor(in_c, out_c)
#
#     def e(self, s):
#         return torch.exp(self.clamp * 2 * (torch.sigmoid(s) - 0.5))
#
#     def forward(self, x, rev=False):
#
#         x1, x2 = (x.narrow(1, 0, self.split_len1),
#                   x.narrow(1, self.split_len1, self.split_len2))
#
#         if not rev:
#
#             t2 = self.f(x2)
#             y1 = x1 + t2
#             s1, t1 = self.r(y1), self.y(y1)
#             y2 = self.e(s1) * x2 + t1
#
#         else:
#
#             s1, t1 = self.r(x1), self.y(x1)
#             y2 = (x2 - t1) / self.e(s1)
#             t2 = self.f(y2)
#             y1 = (x1 - t2)
#
#         return torch.cat((y1, y2), 1)
#
# # class Fusion_Model(nn.Module):
# #     def __init__(self, N):
# #         super().__init__()
# #         self.N = N
# #         self.inv_block1 = INV_block(in_c=self.N, out_c=self.N)
# #         self.inv_block2 = INV_block(in_c=self.N, out_c=self.N)
# #         self.inv_block3 = INV_block(in_c=self.N, out_c=self.N)
# #         self.inv_block4 = INV_block(in_c=self.N, out_c=self.N)
#
#
# #     def forward(self, x, rev=False):
# #         if not rev:
# #             inv_feature = self.inv_block1(x, rev)
# #             inv_feature = self.inv_block2(inv_feature, rev)
# #             inv_feature = self.inv_block3(inv_feature, rev)
# #             inv_feature = self.inv_block4(inv_feature, rev)
# #         else:
# #             inv_feature = self.inv_block4(x, rev)
# #             inv_feature = self.inv_block3(inv_feature, rev)
# #             inv_feature = self.inv_block2(inv_feature, rev)
# #             inv_feature = self.inv_block1(inv_feature, rev)
# #         return inv_feature
#
#
# class EfficientAttention(nn.Module):
#
#     def __init__(self, in_channels, key_channels, head_count, value_channels):
#         super().__init__()
#         self.in_channels = in_channels
#         self.key_channels = key_channels
#         self.head_count = head_count
#         self.value_channels = value_channels
#
#         self.keys = nn.Conv2d(in_channels, key_channels, 1)
#         self.queries = nn.Conv2d(in_channels, key_channels, 1)
#         self.values = nn.Conv2d(in_channels, value_channels, 1)
#         self.reprojection = nn.Conv2d(value_channels, in_channels, 1)
#
#     def forward(self, input_):
#         n, _, h, w = input_.size()
#         keys = self.keys(input_).reshape((n, self.key_channels, h * w))
#         queries = self.queries(input_).reshape(n, self.key_channels, h * w)
#         values = self.values(input_).reshape((n, self.value_channels, h * w))
#         head_key_channels = self.key_channels // self.head_count
#         head_value_channels = self.value_channels // self.head_count
#
#         attended_values = []
#         for i in range(self.head_count):
#             key = f.softmax(keys[
#                 :,
#                 i * head_key_channels: (i + 1) * head_key_channels,
#                 :
#             ], dim=2)
#             query = f.softmax(queries[
#                 :,
#                 i * head_key_channels: (i + 1) * head_key_channels,
#                 :
#             ], dim=1)
#             value = values[
#                 :,
#                 i * head_value_channels: (i + 1) * head_value_channels,
#                 :
#             ]
#             context = key @ value.transpose(1, 2)
#             attended_value = (
#                 context.transpose(1, 2) @ query
#             ).reshape(n, head_value_channels, h, w)
#             attended_values.append(attended_value)
#
#         aggregated_values = torch.cat(attended_values, dim=1)
#         reprojected_value = self.reprojection(aggregated_values)
#         attention = reprojected_value + input_
#
#         return attention
#
#
# # class spatial_alignment(nn.Module):
# #     def __init__(self, in_channel, out_channel):
# #         super(spatial_alignment, self).__init__()
# #         self.in_channel = in_channel
# #         self.out_channel = out_channel
# #         self.res1 = ResBlock(in_channel=self.in_channel, out_channel=self.out_channel, stride=1, kernel_size=3,
# #                                padding=1)
# #         self.res2 = ResBlock(in_channel=self.in_channel, out_channel=self.out_channel, stride=1, kernel_size=3,
# #                              padding=1)
# #         self.conv1 = conv3x3(in_ch=self.in_channel*2, out_ch=self.out_channel)
# #         self.conv2 = conv3x3(in_ch=self.in_channel*2, out_ch=self.out_channel)
# #         self.att1 = EfficientAttention(2*self.in_channel, 2*self.out_channel, 1, 2*self.out_channel)
# #         self.att2 = EfficientAttention(2*self.in_channel, 2*self.out_channel, 1, 2*self.out_channel)
#
#
#
# #     def forward(self, luma, chroma):
# #         luma = self.res1(luma)
# #         chroma = self.res2(chroma)
# #         luma1 = self.att1(torch.cat([luma, chroma], dim=1))
# #         chroma1 = self.att2(torch.cat([luma, chroma], dim=1))
# #         luma1 = self.conv1(luma1)
# #         chroma1 = self.conv2(chroma1)
# #         return luma1, chroma1
#
# # class Fusion_Model(nn.Module):
# #     # n=N/2
# #     def __init__(self, n):
# #         super().__init__()
# #         self.n = n
# #         self.att1 = AttentionBlock(self.n*2)
# #         self.att2 = AttentionBlock(self.n*2)
# #         self.att3 = AttentionBlock(self.n*2)
# #         self.res_y_1 = ResidualBlock(in_ch=self.n, out_ch=self.n)
# #         self.res_y_2 = ResidualBlock(in_ch=self.n*2, out_ch=self.n)
# #         self.res_uv_1 = ResidualBlock(in_ch=self.n, out_ch=self.n)
# #         self.res_uv_2 = ResidualBlock(in_ch=self.n*2, out_ch=self.n)
# #         self.sig = nn.Sigmoid()
# #         self.res_img_1 = ResidualBlock(in_ch=self.n*2, out_ch=self.n*2)
# #         self.res_img_2 = ResidualBlock(in_ch=self.n*2, out_ch=self.n*2)
#
# #     def forward(self, Y_f, UV_f):
# #         Y_f = self.res_y_1(Y_f)
# #         UV_f = self.res_uv_1(UV_f)
# #         Y_f_1 = self.res_y_2(self.att1(torch.cat((Y_f, UV_f), dim=1)))
# #         UV_f_1 = self.res_uv_2(self.att2(torch.cat((Y_f, UV_f), dim=1)))
# #         img_feature = torch.cat((Y_f_1, UV_f_1), dim=1)
# #         mask = self.sig(self.att3(img_feature))
# #         img_feature = self.res_img_2(self.res_img_1(img_feature)*mask+img_feature)
# #         return img_feature
#
# # class Split_Model(nn.Module):
# #     # n=N/2
# #     def __init__(self, n):
# #         super().__init__()
# #         self.n = n
# #         self.att1 = AttentionBlock(self.n*2)
# #         self.att2 = AttentionBlock(self.n*2)
# #         self.att3 = AttentionBlock(self.n*2)
# #         self.res_y_1 = ResidualBlock(in_ch=self.n, out_ch=self.n)
# #         self.res_y_2 = ResidualBlock(in_ch=self.n*2, out_ch=self.n)
# #         self.res_uv_1 = ResidualBlock(in_ch=self.n, out_ch=self.n)
# #         self.res_uv_2 = ResidualBlock(in_ch=self.n*2, out_ch=self.n)
# #         self.sig = nn.Sigmoid()
# #         self.res_img_1 = ResidualBlock(in_ch=self.n*2, out_ch=self.n*2)
# #         self.res_img_2 = ResidualBlock(in_ch=self.n*2, out_ch=self.n*2)
#
# #     def forward(self, x):
# #         mask = self.sig(self.att3(x))
# #         x = self.res_img_2(self.res_img_1(x)*mask+x)
# #         Y_f, UV_f = [torch.chunk(x, 2, dim=1)[i].squeeze(1) for i in range(2)]
# #         Y_f = self.res_y_1(Y_f)
# #         UV_f = self.res_uv_1(UV_f)
# #         Y_f_1 = self.res_y_2(self.att1(torch.cat((Y_f, UV_f), dim=1)))
# #         UV_f_1 = self.res_uv_2(self.att2(torch.cat((Y_f, UV_f), dim=1)))
# #         return Y_f_1, UV_f_1
#
# class Fusion_Model(nn.Module):
#     # n=N/2
#     def __init__(self, n):
#         super().__init__()
#         self.n = n
#         self.in_channel = n*2
#         self.att1 = EfficientAttention(self.in_channel, self.in_channel, 1, self.in_channel)
#         self.att2 = EfficientAttention(self.in_channel, self.in_channel, 1, self.in_channel)
#         self.att3 = ResidualBlock(in_ch=self.in_channel, out_ch=self.in_channel)
#         self.res_y_1 = ResidualBlock(in_ch=self.n, out_ch=self.n)
#         self.res_y_2 = ResidualBlock(in_ch=self.n*2, out_ch=self.n)
#         self.res_uv_1 = ResidualBlock(in_ch=self.n, out_ch=self.n)
#         self.res_uv_2 = ResidualBlock(in_ch=self.n*2, out_ch=self.n)
#         self.sig = nn.Sigmoid()
#         self.res_img_1 = ResidualBlock(in_ch=self.n*2, out_ch=self.n*2)
#         self.res_img_2 = ResidualBlock(in_ch=self.n*2, out_ch=self.n*2)
#
#     def forward(self, Y_f, UV_f):
#         Y_f = self.res_y_1(Y_f)
#         UV_f = self.res_uv_1(UV_f)
#         Y_f_1 = self.res_y_2(self.att1(torch.cat((Y_f, UV_f), dim=1)))
#         UV_f_1 = self.res_uv_2(self.att2(torch.cat((Y_f, UV_f), dim=1)))
#         img_feature = torch.cat((Y_f_1, UV_f_1), dim=1)
#         mask = self.sig(self.att3(img_feature))
#         img_feature = self.res_img_2(self.res_img_1(img_feature)*mask+img_feature)
#         return img_feature
#
# class Split_Model(nn.Module):
#     # n=N/2
#     def __init__(self, n):
#         super().__init__()
#         self.n = n
#         self.in_channel = n*2
#         self.att1 = EfficientAttention(self.in_channel, self.in_channel, 1, self.in_channel)
#         self.att2 = EfficientAttention(self.in_channel, self.in_channel, 1, self.in_channel)
#         self.att3 = ResidualBlock(in_ch=self.in_channel, out_ch=self.in_channel)
#         self.res_y_1 = ResidualBlock(in_ch=self.n, out_ch=self.n)
#         self.res_y_2 = ResidualBlock(in_ch=self.n*2, out_ch=self.n)
#         self.res_uv_1 = ResidualBlock(in_ch=self.n, out_ch=self.n)
#         self.res_uv_2 = ResidualBlock(in_ch=self.n*2, out_ch=self.n)
#         self.sig = nn.Sigmoid()
#         self.res_img_1 = ResidualBlock(in_ch=self.n*2, out_ch=self.n*2)
#         self.res_img_2 = ResidualBlock(in_ch=self.n*2, out_ch=self.n*2)
#
#     def forward(self, x):
#         mask = self.sig(self.att3(x))
#         x = self.res_img_2(self.res_img_1(x)*mask+x)
#         Y_f, UV_f = [torch.chunk(x, 2, dim=1)[i].squeeze(1) for i in range(2)]
#         Y_f = self.res_y_1(Y_f)
#         UV_f = self.res_uv_1(UV_f)
#         Y_f_1 = self.res_y_2(self.att1(torch.cat((Y_f, UV_f), dim=1)))
#         UV_f_1 = self.res_uv_2(self.att2(torch.cat((Y_f, UV_f), dim=1)))
#         return Y_f_1, UV_f_1


import math
import torch
import torch.nn as nn
from torchvision.ops import DeformConv2d
from .layers import ResidualBlock, AttentionBlock, ResidualDenseBlock_out
from compress.layers import conv3x3
import torch.nn.functional as f


class INV_block(nn.Module):
    def __init__(self, subnet_constructor=conv3x3, clamp=2.0, in_c=3, out_c=3):
        super().__init__()

        self.split_len1 = in_c
        self.split_len2 = in_c
        self.clamp = clamp
        # ρ
        self.r = subnet_constructor(in_c, out_c)
        # η
        self.y = subnet_constructor(in_c, out_c)
        # φ
        self.f = subnet_constructor(in_c, out_c)

    def e(self, s):
        return torch.exp(self.clamp * 2 * (torch.sigmoid(s) - 0.5))

    def forward(self, x, rev=False):

        x1, x2 = (x.narrow(1, 0, self.split_len1),
                  x.narrow(1, self.split_len1, self.split_len2))

        if not rev:

            t2 = self.f(x2)
            y1 = x1 + t2
            s1, t1 = self.r(y1), self.y(y1)
            y2 = self.e(s1) * x2 + t1

        else:

            s1, t1 = self.r(x1), self.y(x1)
            y2 = (x2 - t1) / self.e(s1)
            t2 = self.f(y2)
            y1 = (x1 - t2)

        return torch.cat((y1, y2), 1)


# class Fusion_Model(nn.Module):
#     def __init__(self, N):
#         super().__init__()
#         self.N = N
#         self.inv_block1 = INV_block(in_c=self.N, out_c=self.N)
#         self.inv_block2 = INV_block(in_c=self.N, out_c=self.N)
#         self.inv_block3 = INV_block(in_c=self.N, out_c=self.N)
#         self.inv_block4 = INV_block(in_c=self.N, out_c=self.N)


#     def forward(self, x, rev=False):
#         if not rev:
#             inv_feature = self.inv_block1(x, rev)
#             inv_feature = self.inv_block2(inv_feature, rev)
#             inv_feature = self.inv_block3(inv_feature, rev)
#             inv_feature = self.inv_block4(inv_feature, rev)
#         else:
#             inv_feature = self.inv_block4(x, rev)
#             inv_feature = self.inv_block3(inv_feature, rev)
#             inv_feature = self.inv_block2(inv_feature, rev)
#             inv_feature = self.inv_block1(inv_feature, rev)
#         return inv_feature


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


# class spatial_alignment(nn.Module):
#     def __init__(self, in_channel, out_channel):
#         super(spatial_alignment, self).__init__()
#         self.in_channel = in_channel
#         self.out_channel = out_channel
#         self.res1 = ResBlock(in_channel=self.in_channel, out_channel=self.out_channel, stride=1, kernel_size=3,
#                                padding=1)
#         self.res2 = ResBlock(in_channel=self.in_channel, out_channel=self.out_channel, stride=1, kernel_size=3,
#                              padding=1)
#         self.conv1 = conv3x3(in_ch=self.in_channel*2, out_ch=self.out_channel)
#         self.conv2 = conv3x3(in_ch=self.in_channel*2, out_ch=self.out_channel)
#         self.att1 = EfficientAttention(2*self.in_channel, 2*self.out_channel, 1, 2*self.out_channel)
#         self.att2 = EfficientAttention(2*self.in_channel, 2*self.out_channel, 1, 2*self.out_channel)


#     def forward(self, luma, chroma):
#         luma = self.res1(luma)
#         chroma = self.res2(chroma)
#         luma1 = self.att1(torch.cat([luma, chroma], dim=1))
#         chroma1 = self.att2(torch.cat([luma, chroma], dim=1))
#         luma1 = self.conv1(luma1)
#         chroma1 = self.conv2(chroma1)
#         return luma1, chroma1

# class Fusion_Model(nn.Module):
#     # n=N/2
#     def __init__(self, n):
#         super().__init__()
#         self.n = n
#         self.att1 = AttentionBlock(self.n*2)
#         self.att2 = AttentionBlock(self.n*2)
#         self.att3 = AttentionBlock(self.n*2)
#         self.res_y_1 = ResidualBlock(in_ch=self.n, out_ch=self.n)
#         self.res_y_2 = ResidualBlock(in_ch=self.n*2, out_ch=self.n)
#         self.res_uv_1 = ResidualBlock(in_ch=self.n, out_ch=self.n)
#         self.res_uv_2 = ResidualBlock(in_ch=self.n*2, out_ch=self.n)
#         self.sig = nn.Sigmoid()
#         self.res_img_1 = ResidualBlock(in_ch=self.n*2, out_ch=self.n*2)
#         self.res_img_2 = ResidualBlock(in_ch=self.n*2, out_ch=self.n*2)

#     def forward(self, Y_f, UV_f):
#         Y_f = self.res_y_1(Y_f)
#         UV_f = self.res_uv_1(UV_f)
#         Y_f_1 = self.res_y_2(self.att1(torch.cat((Y_f, UV_f), dim=1)))
#         UV_f_1 = self.res_uv_2(self.att2(torch.cat((Y_f, UV_f), dim=1)))
#         img_feature = torch.cat((Y_f_1, UV_f_1), dim=1)
#         mask = self.sig(self.att3(img_feature))
#         img_feature = self.res_img_2(self.res_img_1(img_feature)*mask+img_feature)
#         return img_feature

# class Split_Model(nn.Module):
#     # n=N/2
#     def __init__(self, n):
#         super().__init__()
#         self.n = n
#         self.att1 = AttentionBlock(self.n*2)
#         self.att2 = AttentionBlock(self.n*2)
#         self.att3 = AttentionBlock(self.n*2)
#         self.res_y_1 = ResidualBlock(in_ch=self.n, out_ch=self.n)
#         self.res_y_2 = ResidualBlock(in_ch=self.n*2, out_ch=self.n)
#         self.res_uv_1 = ResidualBlock(in_ch=self.n, out_ch=self.n)
#         self.res_uv_2 = ResidualBlock(in_ch=self.n*2, out_ch=self.n)
#         self.sig = nn.Sigmoid()
#         self.res_img_1 = ResidualBlock(in_ch=self.n*2, out_ch=self.n*2)
#         self.res_img_2 = ResidualBlock(in_ch=self.n*2, out_ch=self.n*2)

#     def forward(self, x):
#         mask = self.sig(self.att3(x))
#         x = self.res_img_2(self.res_img_1(x)*mask+x)
#         Y_f, UV_f = [torch.chunk(x, 2, dim=1)[i].squeeze(1) for i in range(2)]
#         Y_f = self.res_y_1(Y_f)
#         UV_f = self.res_uv_1(UV_f)
#         Y_f_1 = self.res_y_2(self.att1(torch.cat((Y_f, UV_f), dim=1)))
#         UV_f_1 = self.res_uv_2(self.att2(torch.cat((Y_f, UV_f), dim=1)))
#         return Y_f_1, UV_f_1


def kernel_size(in_channel):
    """Compute kernel size for one dimension convolution in eca-net"""
    k = int((math.log2(in_channel) + 1) // 2)  # parameters from ECA-net
    if k % 2 == 0:
        return k + 1
    else:
        return k


class TFAM(nn.Module):
    """Fuse two feature into one feature."""

    def __init__(self, in_channel):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.k = kernel_size(in_channel)
        self.channel_conv1 = nn.Conv1d(4, 1, kernel_size=self.k, padding=self.k // 2)
        self.channel_conv2 = nn.Conv1d(4, 1, kernel_size=self.k, padding=self.k // 2)
        self.spatial_conv1 = nn.Conv2d(4, 1, kernel_size=7, padding=3)
        self.spatial_conv2 = nn.Conv2d(4, 1, kernel_size=7, padding=3)
        self.softmax = nn.Softmax(0)

    def forward(self, t1, t2):
        # channel part
        t1_channel_avg_pool = self.avg_pool(t1)  # b,c,1,1
        t1_channel_max_pool = self.max_pool(t1)  # b,c,1,1
        t2_channel_avg_pool = self.avg_pool(t2)  # b,c,1,1
        t2_channel_max_pool = self.max_pool(t2)  # b,c,1,1

        channel_pool = torch.cat([t1_channel_avg_pool, t1_channel_max_pool,
                                  t2_channel_avg_pool, t2_channel_max_pool],
                                 dim=2).squeeze(-1).transpose(1, 2)  # b,4,c
        t1_channel_attention = self.channel_conv1(channel_pool)  # b,1,c
        t2_channel_attention = self.channel_conv2(channel_pool)  # b,1,c
        channel_stack = torch.stack([t1_channel_attention, t2_channel_attention],
                                    dim=0)  # 2,b,1,c
        channel_stack = self.softmax(channel_stack).transpose(-1, -2).unsqueeze(-1)  # 2,b,c,1,1

        # spatial part
        t1_spatial_avg_pool = torch.mean(t1, dim=1, keepdim=True)  # b,1,h,w
        t1_spatial_max_pool = torch.max(t1, dim=1, keepdim=True)[0]  # b,1,h,w
        t2_spatial_avg_pool = torch.mean(t2, dim=1, keepdim=True)  # b,1,h,w
        t2_spatial_max_pool = torch.max(t2, dim=1, keepdim=True)[0]  # b,1,h,w
        spatial_pool = torch.cat([t1_spatial_avg_pool, t1_spatial_max_pool,
                                  t2_spatial_avg_pool, t2_spatial_max_pool], dim=1)  # b,4,h,w
        t1_spatial_attention = self.spatial_conv1(spatial_pool)  # b,1,h,w
        t2_spatial_attention = self.spatial_conv2(spatial_pool)  # b,1,h,w
        spatial_stack = torch.stack([t1_spatial_attention, t2_spatial_attention], dim=0)  # 2,b,1,h,w
        spatial_stack = self.softmax(spatial_stack)  # 2,b,1,h,w

        # fusion part, add 1 means residual add
        stack_attention = channel_stack + spatial_stack + 1  # 2,b,c,h,w
        fuse = stack_attention[0] * t1 + stack_attention[1] * t2  # b,c,h,w

        return fuse


class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAMLayer, self).__init__()

        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # shared MLP
        self.mlp = nn.Sequential(
            # Conv2d比Linear方便操作
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )

        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x


class Fusion_Model(nn.Module):
    # n=N/2
    def __init__(self, n):
        super().__init__()
        self.n = n
        self.in_channel = n * 2
        self.att1 = EfficientAttention(self.in_channel, self.in_channel, 1, self.in_channel)
        self.att2 = EfficientAttention(self.in_channel, self.in_channel, 1, self.in_channel)
        self.res_y_1 = ResidualBlock(in_ch=self.n, out_ch=self.n)
        self.res_y_2 = ResidualBlock(in_ch=self.n * 2, out_ch=self.n)
        self.res_uv_1 = ResidualBlock(in_ch=self.n, out_ch=self.n)
        self.res_uv_2 = ResidualBlock(in_ch=self.n * 2, out_ch=self.n)
        self.fuse = CBAMLayer(channel=self.n * 2)

    def forward(self, Y_f, UV_f):
        Y_f = self.res_y_1(Y_f)
        UV_f = self.res_uv_1(UV_f)
        Y_f_1 = self.res_y_2(self.att1(torch.cat((Y_f, UV_f), dim=1)))
        UV_f_1 = self.res_uv_2(self.att2(torch.cat((Y_f, UV_f), dim=1)))
        img_feature = self.fuse(torch.cat((Y_f_1, UV_f_1), dim=1))
        return img_feature


class Split_Model(nn.Module):
    # n=N/2
    def __init__(self, n):
        super().__init__()
        self.n = n
        self.in_channel = n * 2
        self.att1 = EfficientAttention(self.in_channel, self.in_channel, 1, self.in_channel)
        self.att2 = EfficientAttention(self.in_channel, self.in_channel, 1, self.in_channel)
        self.res_y_1 = ResidualBlock(in_ch=self.n, out_ch=self.n)
        self.res_y_2 = ResidualBlock(in_ch=self.n * 2, out_ch=self.n)
        self.res_uv_1 = ResidualBlock(in_ch=self.n, out_ch=self.n)
        self.res_uv_2 = ResidualBlock(in_ch=self.n * 2, out_ch=self.n)
        self.fuse = CBAMLayer(channel=self.n * 2)

    def forward(self, x):
        img_feature = self.fuse(x)
        Y_f, UV_f = [torch.chunk(img_feature, 2, dim=1)[i].squeeze(1) for i in range(2)]
        Y_f = self.res_y_1(Y_f)
        UV_f = self.res_uv_1(UV_f)
        Y_f_1 = self.res_y_2(self.att1(torch.cat((Y_f, UV_f), dim=1)))
        UV_f_1 = self.res_uv_2(self.att2(torch.cat((Y_f, UV_f), dim=1)))
        return Y_f_1, UV_f_1