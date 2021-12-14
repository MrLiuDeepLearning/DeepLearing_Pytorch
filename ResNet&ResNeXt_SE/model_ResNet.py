import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torch.autograd import Variable


"""
Sequeeze:通过全局平均池化实现全局信息的获取
Exitation:先通过全连接层Linear(channel, channel/ration)将特征压缩到channel/ration
然后使用ReLU进行非线性操作
再使用全连接层Linear(channel/ration, channel)将特征还原
"""


class SE(nn.Module):

    def __init__(self, in_channel, ratio):
        super(SE, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.compress = nn.Conv2d(in_channel, in_channel // ratio, 1, 1, 0)
        self.excitation = nn.Conv2d(in_channel // ratio, in_channel, 1, 1, 0)

    def forward(self, x):
        out = self.squeeze(x)
        out = self.compress(out)
        out = F.relu(out)
        out = self.excitation(out)
        out = torch.sigmoid(out)

        return out


class BasicBlock(nn.Module):
    message = "basic"

    def __init__(self, in_channel, out_channel, stride, is_SE=False):
        super(BasicBlock, self).__init__()
        self.is_se = is_SE

        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn = nn.BatchNorm2d(out_channel)

        if self.is_se:
            self.se = SE(out_channel, ratio=16)

        self.short_cut = nn.Sequential()
        if stride != 1:  # stride=2
            self.short_cut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_channel))

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn(out)

        if self.is_se:
            coefficient = self.se(out)
            '''将self.se(out)与上一层的卷积特征逐空间位置相乘'''
            out *= coefficient
        out += self.short_cut(x)
        out = F.relu(out)

        return out


class BottleNeck(nn.Module):
    message = "bottleneck"

    def __init__(self, in_channel, out_channel, stride, is_se=False):
        super(BottleNeck, self).__init__()
        self.is_se = is_se

        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv3 = nn.Conv2d(out_channel, out_channel * 4, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel * 4)

        if self.is_se:
            self.se = SE(out_channel * 4, ratio=16)

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channel, out_channel * 4, kernel_size=1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(out_channel * 4))

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv3(out)
        out = self.bn2(out)

        if self.is_se:
            coefficient = self.se(out)
            out *= coefficient
        out += self.shortcut(x)
        out = F.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block: object, groups: object, num_classes, is_se=False) -> object:
        super(ResNet, self).__init__()
        self.channels = 64
        self.block = block
        self.is_se = is_se

        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(3, 2, 1)
        self.conv2_x = self._make_conv_x(channels=64, blocks=groups[0], strides=1, index=2)
        self.conv3_x = self._make_conv_x(channels=128, blocks=groups[1], strides=2, index=3)
        self.conv4_x = self._make_conv_x(channels=256, blocks=groups[2], strides=2, index=4)
        self.conv5_x = self._make_conv_x(channels=512, blocks=groups[3], strides=2, index=5)
        self.pool2 = nn.AvgPool2d(7)
        patches = 512 if self.block.message == "basic" else 512 * 4
        self.fc = nn.Linear(patches, num_classes)  # for 224 * 224 input size

    def _make_conv_x(self, channels, blocks, strides, index):
        list_strides = [strides] + [1] * (blocks - 1)  # In conv_x groups, the first strides is 2, the others are ones.
        conv_x = nn.Sequential()
        for i in range(len(list_strides)):
            layer_name = str("block_%d_%d" % (index, i))  # when use add_module, the name should be difference.
            conv_x.add_module(layer_name, self.block(self.channel, channels, list_strides[i], self.is_se))
            self.channels = channels if self.block.message == "basic" else channels * 4
        return conv_x

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.bn(out))
        out = self.pool1(out)
        out = self.conv2_x(out)
        out = self.conv3_x(out)
        out = self.conv4_x(out)
        out = self.conv5_x(out)
        out = self.pool2(out)
        out = out.view(out.size(0), -1)
        out = F.softmax(self.fc(out))
        return out

    def _make_conv_x(self, channels, blocks, strides, index):
        """
        making convolutional group
        :param channels: output channels of the conv-group
        :param blocks: number of blocks in the conv-group
        :param strides: strides
        :return: conv-group
        """
        list_strides = [strides] + [1] * (blocks - 1)  # In conv_x groups, the first strides is 2, the others are ones.
        conv_x = nn.Sequential()
        for i in range(len(list_strides)):
            layer_name = str("block_%d_%d" % (index, i))  # when use add_module, the name should be difference.
            conv_x.add_module(layer_name, self.block(self.channels, channels, list_strides[i], self.is_se))
            self.channels = channels if self.block.message == "basic" else channels * 4
        return conv_x

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.bn(out))
        out = self.pool1(out)
        out = self.conv2_x(out)
        out = self.conv3_x(out)
        out = self.conv4_x(out)
        out = self.conv5_x(out)
        out = self.pool2(out)
        out = out.view(out.size(0), -1)
        out = F.softmax(self.fc(out), dim=1)
        return out


def ResNet_50_SE(num_classes=5):
    return ResNet(block=BottleNeck, groups=[3, 4, 6, 3], num_classes=num_classes, is_se=True)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
net = ResNet_50_SE()
net.cuda()
data_input = Variable(torch.randn(1, 3, 224, 224)).to(device)

print(data_input.size())
net(data_input)
print(summary(net, (3, 224, 224)))
