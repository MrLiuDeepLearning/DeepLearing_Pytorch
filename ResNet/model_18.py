import torch
from torch import nn
from torch.nn import functional as F
from torchsummary import summary
from torch.autograd import Variable


class ResNetBasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super(ResNetBasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.bn1(out))

        out = self.conv2(out)
        out = self.bn2(out)
        return F.relu(x + out)


class ResNetDownBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super(ResNetDownBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride[0], padding=1)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride[1], padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.extra = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride[0], padding=0),
            nn.BatchNorm2d(out_channel)
        )

    def forward(self, x):
        extra_x = self.extra(x)
        out = self.conv1(x)
        out = F.relu(self.bn1(out))

        out = self.conv2(out)

        out = self.bn2(out)

        return F.relu(extra_x + out)


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential(
            ResNetBasicBlock(64, 64, 1),
            ResNetBasicBlock(64, 64, 1)
        )
        self.layer2 = nn.Sequential(
            ResNetDownBlock(64, 128, [2, 1]),
            ResNetBasicBlock(128, 128, 1)
        )
        self.layer3 = nn.Sequential(
            ResNetDownBlock(128, 256, [2, 1]),
            ResNetBasicBlock(256, 256, 1)
        )
        self.layer4 = nn.Sequential(
            ResNetDownBlock(256, 512, [2, 1]),
            ResNetBasicBlock(512, 512, 1)
        )

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.reshape(x.shape[0], -1)
        out = self.fc(out)
        return out

#
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)
# net = ResNet18()
# net.cuda()
# data_input = Variable(torch.randn(1, 3, 224, 224)).to(device)
#
# print(data_input.size())
# net(data_input)
# print(summary(net, (3, 224, 224)))
