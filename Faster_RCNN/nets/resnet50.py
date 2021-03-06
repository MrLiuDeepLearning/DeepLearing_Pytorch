import math
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url
from torchsummary import summary
from torch.autograd import Variable
import torch


# class BottleNeck(nn.Module):
#     expansion = 4
#
#     def __init__(self, in_channel, out_channel, stride=1, downsample=None):
#         super(BottleNeck, self).__init__()
#         self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, padding=0, bias=False)
#         self.bn1 = nn.BatchNorm2d(out_channel)
#
#         self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_channel)
#
#         self.conv3 = nn.Conv2d(out_channel, out_channel * 4, kernel_size=1, stride=1, padding=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(out_channel * 4)
#
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x):
#         residual = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#
#         out = self.conv3(out)
#         out = self.bn3(out)
#
#         if self.downsample is not None:
#             residual = self.downsample(x)
#
#         out += residual
#         out = self.relu(out)
#
#         return out
class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)

        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.conv3 = nn.Conv2d(out_channel, out_channel * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel * 4)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

# class ResNet(nn.Module):
#     def __init__(self, block, layers, num_classe=1000):
#         # -----------------------------------#
#         # ?????????????????????????????????600,600,3
#         # -----------------------------------#
#         self.in_channel = 64
#         super(ResNet, self).__init__()
#
#         # 600,600,3 -> 300,300,64
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#
#         # 300,300,64 -> 150,150,64
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)
#
#         # 150,150,64 -> 150,150,256
#         self.layer1 = self._make_layers(block, 64, layers[0])
#         # 150,150,256 -> 75,75,512
#         self.layer2 = self._make_layers(block, 128, layers[1], stride=2)
#         # 75,75,512 -> 38,38,1024 ???????????????????????????38,38,1024??????????????????
#         self.layer3 = self._make_layers(block, 256, layers[2], stride=2)
#         # self.layer4?????????classifier?????????
#         self.layer4 = self._make_layers(block, 512, layers[3], stride=2)
#
#         self.avgpool = nn.AvgPool2d(7)
#         self.fc = nn.Linear(512*block.expansion, num_classe)
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#
#     def _make_layers(self, block, out_channel, blocks, stride=1):
#         downsample = None
#         # ----------------------------------------------------------------#
#         # ?????????????????????????????????????????????,???????????????????????????downsample
#         # ----------------------------------------------------------------#
#         if stride != 1 or self.in_channel != out_channel * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.in_channel, out_channel * block.expansion, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(out_channel*block.expansion)
#             )
#         layers = []
#         layers.append(block(self.in_channel, out_channel, stride, downsample))
#         self.in_channel = out_channel * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(self.in_channel, out_channel))
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#
#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#         return x

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        # -----------------------------------#
        #   ??????????????????????????????600,600,3
        # -----------------------------------#
        self.in_channel = 64
        super(ResNet, self).__init__()

        # 600,600,3 -> 300,300,64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # 300,300,64 -> 150,150,64
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)

        # 150,150,64 -> 150,150,256
        self.layer1 = self._make_layer(block, 64, layers[0])
        # 150,150,256 -> 75,75,512
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # 75,75,512 -> 38,38,1024 ???????????????????????????38,38,1024??????????????????
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # self.layer4?????????classifier?????????
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, out_channel, blocks, stride=1):
        downsample = None
        # -------------------------------------------------------------------#
        #   ??????????????????????????????????????????????????????????????????????????????downsample
        # -------------------------------------------------------------------#
        if stride != 1 or self.in_channel != out_channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, out_channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel * block.expansion), )
        layers = []
        layers.append(block(self.in_channel, out_channel, stride, downsample))
        self.in_channel = out_channel * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_channel, out_channel))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# def resnet50(pretrained=False):
#     model = ResNet(BottleNeck, [3, 4, 6, 3])
#     if pretrained:
#         state_dict = load_state_dict_from_url("https://download.pytorch.org/models/resnet50-19c8e357.pth",
#                                               model_dir="./model_data")
#         model.load_state_dict(state_dict)
#     # ----------------------------------------------------------------------------#
#     #   ??????????????????????????????conv1???model.layer3?????????????????????38,38,1024????????????
#     # ----------------------------------------------------------------------------#
#     features = list([model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2, model.layer3])
#     # ----------------------------------------------------------------------------#
#     #   ????????????????????????model.layer4???model.avgpool
#     # ----------------------------------------------------------------------------#
#     classifier = list([model.layer4, model.avgpool])
#
#     features = nn.Sequential(*features)
#     classifier = nn.Sequential(*classifier)
#     return features, classifier


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
net = ResNet(BottleNeck, [3, 4, 6, 3])
net.cuda()
data_input = Variable(torch.randn(1, 3, 600, 600)).to(device)

print(data_input.size())
net(data_input)
print(summary(net, (3, 600, 600)))
