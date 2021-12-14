import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torch.autograd import Variable


class SE(nn.Module):

    def __init__(self, in_chnls, ratio):
        super(SE, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.compress = nn.Conv2d(in_chnls, in_chnls//ratio, 1, 1, 0)
        self.excitation = nn.Conv2d(in_chnls//ratio, in_chnls, 1, 1, 0)

    def forward(self, x):
        out = self.squeeze(x)
        out = self.compress(out)
        out = F.relu(out)
        out = self.excitation(out)
        return F.sigmoid(out)


"""cardinality是内部路径的数量"""
"""4d表示depth即每一个分组的通道数为4（所以第一个基本结构输入通道数为128）"""
# class ResNeXt_Block(nn.Module):
#
#     def __init__(self, in_channel, cardinality, group_depth, stride, is_se=False):
#         super(ResNeXt_Block, self).__init__()
#         self.is_se = is_se
#         self.group_channel = cardinality * group_depth
#
#         self.conv1 = nn.Conv2d(in_channel, self.group_channel, kernel_size=1, stride=1, padding=0)
#         self.conv2 = nn.Conv2d(self.group_channel, self.group_channel, kernel_size=3, stride=stride, padding=1, groups=cardinality)
#         self.conv3 = nn.Conv2d(self.group_channel, self.group_channel*2, kernel_size=1, stride=1, padding=1)
#         self.bn = nn.BatchNorm2d(self.group_channel*2)
#
#         if self.is_se:
#             self.se = SE(self.group_channel*2, 16)
#
#         self.short_cut = nn.Sequential(
#             nn.Conv2d(in_channel, self.group_channel*2, kernel_size=1, stride=stride, padding=0, bias=False),
#             nn.BatchNorm2d(self.group_channel*2)
#         )
#
#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.conv2(out)
#         out = self.conv3(out)
#         out = self.bn(out)
#
#         if self.is_se:
#             coefficient = self.se(out)
#             out *= coefficient
#         out += self.short_cut(x)
#         out = F.relu(out)
#
#         return out

class ResNeXt_Block(nn.Module):
    """
    ResNeXt block with group convolutions
    """

    def __init__(self, in_chnls, cardinality, group_depth, stride, is_se=False):
        super(ResNeXt_Block, self).__init__()
        self.is_se = is_se
        self.group_chnls = cardinality * group_depth
        self.conv1 = nn.Conv2d(in_chnls, self.group_chnls, 1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(self.group_chnls, self.group_chnls, 3, stride=stride, padding=1, groups=cardinality)
        self.conv3 = nn.Conv2d(self.group_chnls, self.group_chnls*2, 1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(self.group_chnls*2)
        if self.is_se:
            self.se = SE(self.group_chnls*2, 16)
        self.short_cut = nn.Sequential(
            nn.Conv2d(in_chnls, self.group_chnls*2, 1, stride, 0, bias=False),
            nn.BatchNorm2d(self.group_chnls*2)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.bn(self.conv3(out))
        if self.is_se:
            coefficient = self.se(out)
            out *= coefficient
        out += self.short_cut(x)
        return F.relu(out)


class ResNeXt(nn.Module):
    """
    ResNeXt builder
    """

    def __init__(self, layers: object, cardinality, group_depth, num_classes, is_se=False) -> object:
        super(ResNeXt, self).__init__()
        self.is_se = is_se
        self.cardinality = cardinality
        self.channels = 64
        self.conv1 = nn.Conv2d(3, self.channels, 7, stride=2, padding=3)
        d1 = group_depth
        self.conv2 = self.___make_layers(d1, layers[0], stride=1)
        d2 = d1 * 2
        self.conv3 = self.___make_layers(d2, layers[1], stride=2)
        d3 = d2 * 2
        self.conv4 = self.___make_layers(d3, layers[2], stride=2)
        d4 = d3 * 2
        self.conv5 = self.___make_layers(d4, layers[3], stride=2)
        self.fc = nn.Linear(self.channels, num_classes)   # 224x224 input size

    def ___make_layers(self, d, blocks, stride):
        strides = [stride] + [1] * (blocks-1)
        layers = []
        for stride in strides:
            layers.append(ResNeXt_Block(self.channels, self.cardinality, d, stride, self.is_se))
            self.channels = self.cardinality*d*2
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = F.max_pool2d(out, 3, 2, 1)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = F.avg_pool2d(out, 7)
        out = out.view(out.size(0), -1)
        out = F.softmax(self.fc(out), dim=1)
        return out


def resNeXt50_32x4d_SE(num_classes=5):
    return ResNeXt([3, 4, 6, 3], 32, 4, num_classes, is_se=True)



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
net = resNeXt50_32x4d_SE()
net.cuda()
data_input = Variable(torch.randn(1, 3, 224, 224)).to(device)

print(data_input.size())
net(data_input)
print(summary(net, (3, 224, 224)))