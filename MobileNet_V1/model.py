# import torch
# import torch.nn as nn
#
#
# def conv3x3(in_planes, out_planes, stride=1, padding=1):
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, bias=False)
#
#
# # why no bias: 如果卷积层之后是BN层，那么可以不用偏置参数，可以节省内存
# def conv1x1(in_planes, out_planes):
#     return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)
#
#
# class DPBlock(nn.Module):
#
#     def __init__(self, in_planes, out_planes, stride=1):
#         super(DPBlock, self).__init__()  # 调用基类__init__函数初始化
#         self.conv1 = conv3x3(in_planes, out_planes, stride)
#         self.bn1 = nn.BatchNorm2d(out_planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv1x1(in_planes, out_planes)
#         self.bn2 = nn.BatchNorm2d(out_planes)
#
#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#         print(x.shape)
#         print(out.shape)
#
#         return out
#
#
# class MobileNetV1Net(nn.Module):
#
#     def __init__(self, num_classes=5):
#         super(MobileNetV1Net, self).__init__()
#         block = DPBlock
#         self.model = nn.Sequential(
#             # 第一层卷积，
#             conv3x3(3, 32, 2),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             block(32, 64, 1),
#             block(64, 128, 2),
#             block(128, 128, 1),
#             block(128, 256, 2),
#             block(256, 256, 1),
#             block(256, 512, 2),
#             block(512, 512, 1),
#             block(512, 512, 1),
#             block(512, 512, 1),
#             block(512, 512, 1),
#             block(512, 512, 1),
#             block(512,  1024, 2),
#             block(1024, 1024, 2),
#             nn.AvgPool2d(7))
#         self.fc = nn.Linear(1024, num_classes)
#
#     def forward(self, x):
#         x = self.model(x)
#         x = x.view(-1, 1024)  # reshape
#         out = self.fc(x)
#
#         return out
#
#
# mobileNetV1 = MobileNetV1Net(num_classes=5)
#
# print(mobileNetV1)
# x = torch.randn(3, 3, 224, 224)
# y = mobileNetV1(x)
# print(y.shape)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torch.autograd import Variable


class Conv(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super(Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class ConvDW(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super(ConvDW, self).__init__()

        self.convDW = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=stride, padding=1, groups=in_channel, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.convDW(x)


class MobileNet_V1(nn.Module):
    def __init__(self, num_classes=5):
        super(MobileNet_V1, self).__init__()
        self.feature = nn.Sequential(
            Conv(3, 32, 2),
            ConvDW(32, 64, 1),
            ConvDW(64, 128, 2),
            ConvDW(128, 128, 1),
            ConvDW(128, 256, 2),
            ConvDW(256, 256, 1),
            ConvDW(256, 512, 2),
            ConvDW(512, 512, 1),
            ConvDW(512, 512, 1),
            ConvDW(512, 512, 1),
            ConvDW(512, 512, 1),
            ConvDW(512, 512, 1),
            ConvDW(512, 1024, 2),
            ConvDW(1024, 1024, 1),
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):

        out = self.feature(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
net = MobileNet_V1()
net.cuda()
data_input = Variable(torch.randn(1, 3, 224, 224)).to(device)

print(data_input.size())
net(data_input)
print(summary(net, (3, 224, 224)))
