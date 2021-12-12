# import torch
# import torch.nn as nn
# from torch.nn import functional as F
#
#
# # DenseBlock中的非线性组合函数 采用的是BN+ReLU+3x3 Conv的结构
# # 所有DenseBlock中各个层卷积之后均输出k个特征图，即得到的特征图的channel数为k，或者说采用k个卷积核
# # 在DenseNet称为growth rate，这是一个超参数。一般情况下使用较小的k=12
# # 假定输入层的特征图的channel数为k0,那么l层的channel数为k0 + k(l-1)
# # 每层仅有k个是自己独有的
#
# # DenseBlock是包含很多层的模块，每个层的特征图大小相同，层与层之间采用密集连接方式。
# # Transition模块是连接两个相邻的DenseBlock，并且通过Pooling使特征图大小降低
# # 对于Transition层，它主要是连接两个相邻的DenseBlock，并且降低特征图大小。
# # Transition层包括一个1x1的卷积和2x2的AvgPooling，结构为BN+ReLU+1x1 Conv+2x2 AvgPooling。
# # 另外，Transition层可以起到压缩模型的作用。假定Transition的上接DenseBlock得到的特征图channels数为theta(压缩系数)*M
#
#
# # 由于后面层的输入会非常大，DenseBlock内部可以采用bottleneck层来减少计算量，主要是原有的结构中增加1x1 Conv
# # 即BN+ReLU+1x1 Conv+BN+ReLU+3x3 Conv，称为DenseNet-B结构。其中1x1 Conv得到4k个特征图，作用是降低特征数量，从而提升计算效率
#
# class BN_Conv2d(nn.Module):
#     def __init__(self, in_channel, out_channel, kernel_size, stride, padding, dilation, groups=1, bias=False):
#         super(BN_Conv2d, self).__init__()
#         self.Conv_BN = nn.Sequential(
#             nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
#                       groups=groups, bias=bias),
#             nn.BatchNorm2d(out_channel)
#         )
#
#     def forward(self, x):
#         return F.relu(self.Conv_BN)
#
#
# class DenseBlock(nn.Module):
#     # 这里没有out_channnel,是因为输出的通道数为grouth_rate
#     def __init__(self, in_channel, num_layers, growth_rate):
#         super(DenseBlock, self).__init__()
#         self.in_channel = in_channel
#         self.growth_rate = growth_rate
#         self.num_layers = num_layers
#         self.layers = self._make_layers()
#
#     def _make_layers(self):
#         layer_list = []
#
#         for i in range(self.num_layers):
#             layer_list.append(
#                 nn.Sequential(
#                     BN_Conv2d(self.in_channel + i*self.growth_rate, 4*self.growth_rate, 1, 1, 0),
#                     BN_Conv2d(4*self.growth_rate, self.growth_rate, 3, 1, 1),
#                 )
#             )
#
#         return layer_list
import torch
import torch.nn as nn
from collections import OrderedDict



class DenseLayer(nn.Sequential):
    def __init__(self, in_channel, growth_rate, bn_size):
        super(DenseLayer, self).__init__()
        self.add_module('bn1', nn.BatchNorm2d(in_channel))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv1x1', nn.Conv2d(in_channel, growth_rate*bn_size, kernel_size=1, stride=1, padding=0, bias=False))

        self.add_module('bn2', nn.BatchNorm2d(growth_rate*bn_size))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv3x3', nn.Conv2d(growth_rate*bn_size, growth_rate, kernel_size=3, stride=1, padding=1, bias=False))

    def forward(self, input):
        output = super(DenseLayer, self).forward(input)
        return torch.cat([input, output], 1)


class DenseBlock(nn.Sequential):
    def __init__(self, num_layers, in_channel, bn_size, growth_rate):
        super(DenseBlock, self).__init__()
        for i in range(num_layers):
            self.add_module('denselayer%d' % (i+1), DenseLayer(in_channel + growth_rate*i, growth_rate, bn_size))


class Transition(nn.Sequential):
    def __init__(self, in_channle, out_channel):
        super(Transition, self).__init__()
        self.add_module('bn', nn.BatchNorm2d(in_channle))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(in_channle, out_channel, kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


'''DenseBlock_BC是包含1x1卷积层和缩小系数的block'''
'''DenseNet_100'''
class DenseNet_BC(nn.Module):

    def __init__(self, growth_rate=12, block_config=(6, 12, 24, 16), bn_size=4, theta=0.5, num_classes=10):
        super(DenseNet_BC, self).__init__()
        num_init_features = 2 * growth_rate

        if num_classes ==10:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False))
            ]))
        else:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
                ('norm0', nn.BatchNorm2d(num_init_features)),
                ('relu0', nn.ReLU(inplace=True)),
                ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            ]))

        num_features = num_init_features

        for i, num_layers in enumerate(block_config):
            self.features.add_module('denseblock%d' % (i+1), DenseBlock(num_layers, num_features, bn_size, growth_rate))
            num_features = num_features + growth_rate * num_layers
            if i != len(block_config)-1:
                self.features.add_module('transition%d' % (i+1), Transition(num_features, int(num_features * theta)))
                num_features = int(num_features * theta)

        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        self.features.add_module('relu5', nn.ReLU(inplace=True))
        self.features.add_module('avg_pool', nn.AdaptiveAvgPool2d((1, 1)))

        self.classifier = nn.Linear(num_features, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x ):
        features = self.features(x)
        out = features.view(features.size(0), -1)
        out = self.classifier(out)
        return out