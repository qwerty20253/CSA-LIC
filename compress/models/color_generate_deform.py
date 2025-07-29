import torch
import torch.nn as nn
from torchvision.ops import DeformConv2d
from compress.layers import conv3x3, subpel_conv3x3, Win_noShift_Attention
from .layers import ResidualBlockUpsample, ResidualBlock
from .utils import deconv
from .res import resnet101


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
    
# class Generate(nn.Module):
#     def __init__(self, N):
#         super(Generate, self).__init__()
#         self.group = 1
#         self.kernel_size = (3, 3)
#         self.head1 = nn.Sequential(
#             nn.Conv2d(in_channels=N,
#                       out_channels=2 * self.kernel_size[0] * self.kernel_size[1] * self.group,
#                       kernel_size=(3, 3), stride=1, padding=1)
#             # nn.ReLU(inplace=True)
#         )
#         self.head2 = nn.Sequential(
#             nn.Conv2d(in_channels=N,
#                       out_channels=2 * self.kernel_size[0] * self.kernel_size[1] * self.group,
#                       kernel_size=(3, 3), stride=1, padding=1)
#             # nn.ReLU(inplace=True)
#         )
#         self.uv_conv = nn.Conv2d(in_channels=2, out_channels=N, kernel_size=3, stride=1, padding=1)
#         self.deform_conv1 = DeformConv2d(in_channels=N, out_channels=N, kernel_size=3, stride=1, padding=1)
#         self.uv_conv1 = nn.Conv2d(in_channels=N, out_channels=N, kernel_size=3, stride=1, padding=1)
#         self.deform_conv2 = DeformConv2d(in_channels=N, out_channels=N, kernel_size=3, stride=1, padding=1)
#         self.uv_conv2 = nn.Conv2d(in_channels=N, out_channels=N, kernel_size=3, stride=1, padding=1)
#         self.y_conv1 = ResidualBlock(in_ch=N, out_ch=N)
#         self.y_conv2 = ResidualBlock(in_ch=N, out_ch=N)
#         self.y_down1 = nn.Conv2d(in_channels=1, out_channels=N, stride=2, kernel_size=2)
#         self.y_down2 = nn.Conv2d(in_channels=N, out_channels=N, stride=2, kernel_size=2)
#         self.att = Win_noShift_Attention(dim=N*2, num_heads=8, window_size=8, shift_size=4)
#         self.conv3 = ResidualBlockUpsample(in_ch=N*2, out_ch=N*2)
#         self.conv4 = ResidualBlockUpsample(in_ch=N*2, out_ch=2)

#     def forward(self, Y, UV):
#         Y_feature = self.y_down2(self.y_down1(Y))
#         Y_feature1 = self.y_conv1(Y_feature)
#         Y_feature2 = self.y_conv2(Y_feature1)
#         offsets = self.head1(Y_feature1)
#         UV_features = self.uv_conv(UV)
#         UV_features = self.uv_conv1(self.deform_conv1(UV_features, offsets))
#         offsets = self.head2(Y_feature2)
#         UV_features = self.uv_conv2(self.deform_conv2(UV_features, offsets))
#         feature = self.att(torch.concat((Y_feature2, UV_features), dim=1))
#         out = self.conv4(self.conv3(feature))
#         return out
    

class Generate(nn.Module):
    def __init__(self, N):
        super(Generate, self).__init__()
        self.pretrained_res = resnet101()

        self.group = 1
        self.kernel_size = (3, 3)
        self.head1 = nn.Sequential(
            nn.Conv2d(in_channels=N,
                      out_channels=2 * self.kernel_size[0] * self.kernel_size[1] * self.group,
                      kernel_size=(3, 3), stride=1, padding=1)
            # nn.ReLU(inplace=True)
        )
        self.head2 = nn.Sequential(
            nn.Conv2d(in_channels=N,
                      out_channels=2 * self.kernel_size[0] * self.kernel_size[1] * self.group,
                      kernel_size=(3, 3), stride=1, padding=1)
            # nn.ReLU(inplace=True)
        )
        self.uv_conv = nn.Conv2d(in_channels=2, out_channels=N, kernel_size=3, stride=1, padding=1)
        self.deform_conv1 = DeformConv2d(in_channels=N, out_channels=N, kernel_size=3, stride=1, padding=1)
        self.uv_conv1 = nn.Conv2d(in_channels=N, out_channels=N, kernel_size=3, stride=1, padding=1)
        self.deform_conv2 = DeformConv2d(in_channels=N, out_channels=N, kernel_size=3, stride=1, padding=1)
        self.uv_conv2 = nn.Conv2d(in_channels=N, out_channels=N, kernel_size=3, stride=1, padding=1)
        self.y_conv1 = ResidualBlock(in_ch=N, out_ch=N)
        self.y_conv2 = ResidualBlock(in_ch=N, out_ch=N)
        self.y_down1 = nn.Conv2d(in_channels=1, out_channels=N, stride=2, kernel_size=2)
        self.y_down2 = nn.Conv2d(in_channels=N, out_channels=N, stride=2, kernel_size=2)
        self.att = Win_noShift_Attention(dim=N*2, num_heads=8, window_size=8, shift_size=4)
        self.conv3 = ResidualBlockUpsample(in_ch=N*2, out_ch=N*2)
        self.conv4 = ResidualBlockUpsample(in_ch=N*2, out_ch=2)

    def forward(self, Y, UV):
        Y_feature = self.y_down2(self.y_down1(Y))
        Y_feature1 = self.y_conv1(Y_feature)
        Y_feature2 = self.y_conv2(Y_feature1)
        offsets = self.head1(Y_feature1)
        UV_features = self.uv_conv(UV)
        UV_features = self.uv_conv1(self.deform_conv1(UV_features, offsets))
        offsets = self.head2(Y_feature2)
        UV_features = self.uv_conv2(self.deform_conv2(UV_features, offsets))
        feature = self.att(torch.concat((Y_feature2, UV_features), dim=1))
        out = self.conv4(self.conv3(feature))
        return out