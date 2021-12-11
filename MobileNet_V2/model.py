import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torch.autograd import Variable


class Head(nn.Module):
    def __init__(self):
        super(Head, self).__init__()
        self.conv_32 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn_32 = nn.BatchNorm2d(32)

    def forward(self, x):
        out = F.relu6(self.bn_32(self.conv_32(x)))

        return out


class Tail(nn.Module):
    def __init__(self, num_classes):
        super(Tail, self).__init__()
        self.conv_1280 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_1280 = nn.BatchNorm2d(1280)

        self.conv_end = nn.Conv2d(1280, num_classes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_end = nn.BatchNorm2d(num_classes)

    def forward(self, x):
        out = F.relu6(self.bn_1280(self.conv_1280(x)))
        out = F.avg_pool2d(out, kernel_size=7)
        out = self.conv_end(out)
        out = out.squeeze(dim=2)
        out = out.squeeze(dim=2)

        return out


class BottleNeck(nn.Module):
    def __init__(self, in_channel, expansion, out_channel, repeat_times, stride):
        super(BottleNeck, self).__init__()
        inner_channels = expansion * in_channel
        '''根据论文分析得知，invertedResidual总共有三种基本结构'''
        """第一种是1X1的卷积+ReLU6()函数"""
        self.conv1 = nn.Conv2d(in_channel, inner_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(inner_channels)
        """第二种是3X3的卷积+ReLU6()函数"""
        '''第一种情况：在invertedResidual第一层的时候，stride=2'''
        self.conv2_with_stride = nn.Conv2d(inner_channels, inner_channels, kernel_size=3, stride=stride, padding=1, groups=inner_channels, bias=False)
        '''第二种情况：在invertedResidual其它层的时候，stride=1'''
        self.conv2_no_stride = nn.Conv2d(inner_channels, inner_channels, kernel_size=3, stride=1, padding=1, groups=inner_channels, bias=False)

        """第三种是1X1的卷积+Linear()函数"""
        self.conv3 = nn.Conv2d(inner_channels, out_channel, kernel_size=1, stride=1, padding=0, groups=1, bias=False)


        """当invertedResidual重复出现时，以上所有的结构输入输出通道都发生了变化，不能再使用了，需要重新定义"""
        self.conv1_inner = nn.Conv2d(out_channel, expansion * out_channel, kernel_size=1, stride=1, padding=0,
                                     bias=False)

        self.conv2_inner_with_stride = nn.Conv2d(out_channel * expansion, expansion * out_channel, kernel_size=3,
                                                 stride=stride, padding=1, groups=out_channel, bias=False)
        self.conv2_inner_no_stride = nn.Conv2d(out_channel * expansion, expansion * out_channel, kernel_size=3,
                                               stride=1, padding=1, groups=out_channel, bias=False)

        self.conv3_inner = nn.Conv2d(out_channel * expansion, out_channel, kernel_size=1, stride=1, padding=0, groups=1,
                                     bias=False)

        self.bn_inner = nn.BatchNorm2d(expansion * out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.n = repeat_times

    def forward(self, x):
        # out = F.relu6(self.bn1(self.conv1(x)))
        # print("(x):{}".format(x.shape))
        out = self.conv1(x)
        # print("self.conv1(x):{}".format(x.shape))
        out = self.bn1(out)
        # print("self.bn1(x):{}".format(out.shape))
        out = F.relu6(out)
        # print("F.relu6(x):{}".format(out.shape))
        #
        out = self.conv2_with_stride(out)
        out = self.bn1(out)
        out = F.relu6(out)

        out = self.conv3(out)
        out = self.bn2(out)

        count = 2
        while count <= self.n:
            temp = out
            '''1X1 Conv'''
            out = self.conv1_inner(out)
            out = self.bn_inner(out)
            out = F.relu6(out)

            '''3X3 Conv'''
            out = self.conv2_inner_no_stride(out)
            out = self.bn_inner(out)
            out = F.relu6(out)

            '''1X1 Conv'''
            out = self.conv3_inner(out)
            out = self.bn2(out)

            out = out + temp
            count += 1
        return out


class MobileNet_V2(nn.Module):
    '''输入通道，倍增系数，输出通道，模块重复次数，步距'''
    param = [[32, 1, 16, 1, 1],
             [16, 6, 24, 2, 2],
             [24, 6, 32, 3, 2],
             [32, 6, 64, 4, 1],
             [64, 6, 96, 3, 2],
             [96, 6, 160, 3, 2],
             [160, 6, 320, 1, 1]]

    def __init__(self, num_classes):
        super(MobileNet_V2, self).__init__()
        self.layers = self._make_layers(num_classes=num_classes)

    def _make_layers(self, num_classes):
        layer = []
        layer.append(Head())

        for i in range(len(self.param)):
            layer.append(
                BottleNeck(self.param[i][0], self.param[i][1], self.param[i][2], self.param[i][3], self.param[i][4]))

        layer.append(Tail(num_classes))
        # self.fc = nn.Linear(1280, num_classes)
        return nn.Sequential(*layer)

    def forward(self, x):

        out = self.layers(x)

        return out


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
net = MobileNet_V2(num_classes=5)
net.cuda()
data_input = Variable(torch.randn(1, 3, 224, 224)).to(device)

# print(data_input.size())
net(data_input)
print(summary(net, (3, 224, 224)))
