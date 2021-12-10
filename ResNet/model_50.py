# import torch
# import torch.nn as nn
# from torch.nn import functional as F
# from torchsummary import summary
# from torch.autograd import Variable
#
#
# class ResNet50BasicBlock(nn.Module):
#     def __init__(self, in_channel, outs, kernerl_size, stride, padding):
#         super(ResNet50BasicBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_channel, outs[0], kernel_size=kernerl_size[0], stride=stride[0], padding=padding[0])
#         self.bn1 = nn.BatchNorm2d(outs[0])
#         self.conv2 = nn.Conv2d(outs[0], outs[1], kernel_size=kernerl_size[1], stride=stride[0], padding=padding[1])
#         self.bn2 = nn.BatchNorm2d(outs[1])
#         self.conv3 = nn.Conv2d(outs[1], outs[2], kernel_size=kernerl_size[2], stride=stride[0], padding=padding[2])
#         self.bn3 = nn.BatchNorm2d(outs[2])
#
#     def forward(self, x):
#         out = self.conv1(x)
#         out = F.relu(self.bn1(out))
#
#         out = self.conv2(out)
#         out = F.relu(self.bn2(out))
#
#         out = self.conv3(out)
#         out = self.bn3(out)
#
#         return F.relu(out + x)
#
#
# class ResNet50BottleNeck(nn.Module):
#     def __init__(self, in_channel, outs, kernel_size, stride, padding):
#         super(ResNet50BottleNeck, self).__init__()
#         # out1, out2, out3 = outs
#         # print(outs)
#         self.conv1 = nn.Conv2d(in_channel, outs[0], kernel_size=kernel_size[0], stride=stride[0], padding=padding[0])
#         self.bn1 = nn.BatchNorm2d(outs[0])
#         self.conv2 = nn.Conv2d(outs[0], outs[1], kernel_size=kernel_size[1], stride=stride[1], padding=padding[1])
#         self.bn2 = nn.BatchNorm2d(outs[1])
#         self.conv3 = nn.Conv2d(outs[1], outs[2], kernel_size=kernel_size[2], stride=stride[2], padding=padding[2])
#         self.bn3 = nn.BatchNorm2d(outs[2])
#
#         self.extra = nn.Sequential(nn.Conv2d(in_channel, outs[2], kernel_size=1, stride=stride[3], padding=0),
#             nn.BatchNorm2d(outs[2]))
#
#     def forward(self, x):
#         x_shortcut = self.extra(x)
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = F.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = F.relu(out)
#
#         out = self.conv3(out)
#         out = self.bn3(out)
#         return F.relu(x_shortcut + out)
#
#
# class ResNet50(nn.Module):
#     def __init__(self):
#         super(ResNet50, self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#
#         self.layer1 = nn.Sequential(
#             ResNet50BottleNeck(64, outs=[64, 64, 256], kernel_size=[1, 3, 1], stride=[1, 1, 1, 1], padding=[0, 1, 0]),
#             ResNet50BasicBlock(256, outs=[64, 64, 256], kernerl_size=[1, 3, 1], stride=[1, 1, 1, 1], padding=[0, 1, 0]),
#             ResNet50BasicBlock(256, outs=[64, 64, 256], kernerl_size=[1, 3, 1], stride=[1, 1, 1, 1],
#                                padding=[0, 1, 0]), )
#
#         self.layer2 = nn.Sequential(
#             ResNet50BottleNeck(256, outs=[128, 128, 512], kernel_size=[1, 3, 1], stride=[1, 2, 1, 2],
#                                padding=[0, 1, 0]),
#             ResNet50BasicBlock(512, outs=[128, 128, 512], kernerl_size=[1, 3, 1], stride=[1, 1, 1, 1],
#                                padding=[0, 1, 0]),
#             ResNet50BasicBlock(512, outs=[128, 128, 512], kernerl_size=[1, 3, 1], stride=[1, 1, 1, 1],
#                                padding=[0, 1, 0]),
#             ResNet50BottleNeck(512, outs=[128, 128, 512], kernel_size=[1, 3, 1], stride=[1, 1, 1, 1],
#                                padding=[0, 1, 0]))
#
#         self.layer3 = nn.Sequential(
#             ResNet50BottleNeck(512, outs=[256, 256, 1024], kernel_size=[1, 3, 1], stride=[1, 2, 1, 2],
#                                padding=[0, 1, 0]),
#             ResNet50BasicBlock(1024, outs=[256, 256, 1024], kernerl_size=[1, 3, 1], stride=[1, 1, 1, 1],
#                                padding=[0, 1, 0]),
#             ResNet50BasicBlock(1024, outs=[256, 256, 1024], kernerl_size=[1, 3, 1], stride=[1, 1, 1, 1],
#                                padding=[0, 1, 0]),
#             ResNet50BottleNeck(1024, outs=[256, 256, 1024], kernel_size=[1, 3, 1], stride=[1, 1, 1, 1],
#                                padding=[0, 1, 0]),
#             ResNet50BottleNeck(1024, outs=[256, 256, 1024], kernel_size=[1, 3, 1], stride=[1, 1, 1, 1],
#                                padding=[0, 1, 0]),
#             ResNet50BottleNeck(1024, outs=[256, 256, 1024], kernel_size=[1, 3, 1], stride=[1, 1, 1, 1],
#                                padding=[0, 1, 0]))
#
#         self.layer4 = nn.Sequential(
#             ResNet50BottleNeck(1024, outs=[512, 512, 2048], kernel_size=[1, 3, 1], stride=[1, 2, 1, 2],
#                                padding=[0, 1, 0]),
#             ResNet50BottleNeck(2048, outs=[512, 512, 2048], kernel_size=[1, 3, 1], stride=[1, 1, 1, 1],
#                                padding=[0, 1, 0]),
#             ResNet50BottleNeck(2048, outs=[512, 512, 2048], kernel_size=[1, 3, 1], stride=[1, 1, 1, 1],
#                                padding=[0, 1, 0]))
#
#         self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
#
#         self.fc = nn.Linear(2048, 10)
#
#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.maxpool(out)
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
#         out = self.avgpool(out)
#         out = out.reshape(x.shape[0], -1)
#         out = self.fc(out)
#         return out
#
#
# if __name__ == '__main__':
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     print(device)
#     net = ResNet50()
#     net.cuda()
#     data_input = Variable(torch.randn(1, 3, 224, 224)).to(device)
#
#     print(data_input.size())
#     net(data_input)
#     print(summary(net, (3, 224, 224)))

import torch
import torch.nn as nn
from torch.nn import functional as F
from torchsummary import summary
from torch.autograd import Variable


class ResNet50BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super(ResNet50BasicBlock, self).__init__()
        # in_channel = 64, out_channel = 64
        self.conv1 = nn.Conv2d(in_channel, out_channel[0], kernel_size=kernel_size[0], stride=stride[0], padding=padding[0])
        self.bn1 = nn.BatchNorm2d(out_channel[0])
        # in: 64, out： 64
        self.conv2 = nn.Conv2d(out_channel[0], out_channel[1], kernel_size=kernel_size[1], stride=stride[0], padding=padding[1])
        self.bn2 = nn.BatchNorm2d(out_channel[1])
        # in: 64, out： 256
        self.conv3 = nn.Conv2d(out_channel[1], out_channel[2], kernel_size=kernel_size[2], stride=stride[2], padding=padding[2])
        self.bn3 = nn.BatchNorm2d(out_channel[2])

    def forward(self, x):
        # print(x.shape)
        out = self.conv1(x)
        out = F.relu(self.bn1(out))

        out = self.conv2(out)
        out = F.relu(self.bn2(out))

        out = self.conv3(out)
        out = F.relu(self.bn3(out))

        return F.relu(x + out)


class ResNet50BottleNeck(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super(ResNet50BottleNeck, self).__init__()

        self.conv1 = nn.Conv2d(in_channel, out_channel[0], kernel_size=kernel_size[0], stride=stride[0], padding=padding[0])
        self.bn1 = nn.BatchNorm2d(out_channel[0])
        self.conv2 = nn.Conv2d(out_channel[0], out_channel[1], kernel_size=kernel_size[1], stride=stride[1], padding=padding[1])
        self.bn2 = nn.BatchNorm2d(out_channel[1])
        self.conv3 = nn.Conv2d(out_channel[1], out_channel[2], kernel_size=kernel_size[2], stride=stride[2], padding=padding[2])
        self.bn3 = nn.BatchNorm2d(out_channel[2])

        self.extra = nn.Sequential(
            nn.Conv2d(in_channel, out_channel[2], kernel_size=1, stride=stride[3], padding=0),
            nn.BatchNorm2d(out_channel[2])
        )

    def forward(self, x):
        x_shortcut = self.extra(x)
        # print(x.shape)
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        # print(out.shape)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        # print(out.shape)

        out = self.conv3(out)
        out = self.bn3(out)
        # print(out.shape)

        return F.relu(x_shortcut + out)


class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential(
            ResNet50BottleNeck(64, out_channel=[64, 64, 256], kernel_size=[1, 3, 1], stride=[1, 1, 1, 1], padding=[0, 1, 0]),
            ResNet50BasicBlock(256, out_channel=[64, 64, 256], kernel_size=[1, 3, 1], stride=[1, 1, 1, 1], padding=[0, 1, 0]),
            ResNet50BasicBlock(256, out_channel=[64, 64, 256], kernel_size=[1, 3, 1], stride=[1, 1, 1, 1], padding=[0, 1, 0])
        )
        self.layer2 = nn.Sequential(
            ResNet50BottleNeck(256, out_channel=[128, 128, 512], kernel_size=[1, 3, 1], stride=[1, 2, 1, 2], padding=[0, 1, 0]),
            ResNet50BasicBlock(512, out_channel=[128, 128, 512], kernel_size=[1, 3, 1], stride=[1, 1, 1, 1], padding=[0, 1, 0]),
            ResNet50BasicBlock(512, out_channel=[128, 128, 512], kernel_size=[1, 3, 1], stride=[1, 1, 1, 1], padding=[0, 1, 0]),
            ResNet50BottleNeck(512, out_channel=[128, 128, 512], kernel_size=[1, 3, 1], stride=[1, 1, 1, 1], padding=[0, 1, 0])
        )
        self.layer3 = nn.Sequential(
            ResNet50BottleNeck(512, out_channel=[256, 256, 1024], kernel_size=[1, 3, 1], stride=[1, 2, 1, 2], padding=[0, 1, 0]),
            ResNet50BasicBlock(1024, out_channel=[256, 256, 1024], kernel_size=[1, 3, 1], stride=[1, 1, 1, 1], padding=[0, 1, 0]),
            ResNet50BasicBlock(1024, out_channel=[256, 256, 1024], kernel_size=[1, 3, 1], stride=[1, 1, 1, 1], padding=[0, 1, 0]),
            ResNet50BottleNeck(1024, out_channel=[256, 256, 1024], kernel_size=[1, 3, 1], stride=[1, 1, 1, 1], padding=[0, 1, 0]),
            ResNet50BottleNeck(1024, out_channel=[256, 256, 1024], kernel_size=[1, 3, 1], stride=[1, 1, 1, 1], padding=[0, 1, 0]),
            ResNet50BottleNeck(1024, out_channel=[256, 256, 1024], kernel_size=[1, 3, 1], stride=[1, 1, 1, 1], padding=[0, 1, 0])
        )

        self.layer4 = nn.Sequential(
            ResNet50BottleNeck(1024, out_channel=[512, 512, 2048], kernel_size=[1, 3, 1], stride=[1, 2, 1, 2], padding=[0, 1, 0]),
            ResNet50BottleNeck(2048, out_channel=[512, 512, 2048], kernel_size=[1, 3, 1], stride=[1, 1, 1, 1], padding=[0, 1, 0]),
            ResNet50BottleNeck(2048, out_channel=[512, 512, 2048], kernel_size=[1, 3, 1], stride=[1, 1, 1, 1], padding=[0, 1, 0])
        )

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.fc = nn.Linear(2048, 10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.reshape(x.shape[0], -1)
        out = self.fc(out)
        return out


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)
# net = ResNet50()
# net.cuda()
# data_input = Variable(torch.randn(1, 3, 224, 224)).to(device)
#
# print(data_input.size())
# net(data_input)
# print(summary(net, (3, 224, 224)))
