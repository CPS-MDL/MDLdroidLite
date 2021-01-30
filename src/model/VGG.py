'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
from torchsummary import summary
from ptflops import get_model_complexity_info

__all__ = ['VGG']

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG11_10': [8, 'M', 16, 'M', 32, 32, 'M', 64, 64, 'M', 64, 64, 'M'],
    'VGG11_25': [16, 'M', 32, 'M', 64, 64, 'M', 128, 128, 'M', 128, 128, 'M'],
    'VGG11_25_1': [16, 'M', 32, 'M', 64, 64, 'M', 128, 128, 'M', 256, 256, 'M'],
    'VGG11_25_2': [16, 'M', 32, 'M', 128, 128, 'M', 128, 128, 'M', 256, 256, 'M'],
    'VGG11_40': [25, 'M', 51, 'M', 102, 102, 'M', 204, 204, 'M', 204, 204, 'M'],
    'VGG11_50': [32, 'M', 64, 'M', 128, 128, 'M', 256, 256, 'M', 256, 256, 'M'],
    'VGG11_60': [38, 'M', 76, 'M', 153, 153, 'M', 307, 307, 'M', 307, 307, 'M'],
    'VGG11_70': [44, 'M', 89, 'M', 179, 179, 'M', 358, 358, 'M', 358, 358, 'M'],
    'VGG11_s1': [63, 'M', 44, 'M', 51, 51, 'M', 86, 86, 'M', 86, 86, 'M'],
    'VGG11_bn_s1': [17, 'M', 30, 'M', 52, 52, 'M', 86, 86, 'M', 86, 86, 'M'],
    'seed': [6, 'M', 12, 'M', 25, 25, 'M', 50, 50, 'M', 50, 50, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        # self.classifier = nn.Sequential(nn.Linear(512, 10))
        self.classifier = nn.Sequential(nn.Linear(cfg[vgg_name][-2], 10))

    def forward(self, x):
        out = self.features(x)
        out1 = out.view(out.size(0), -1)
        out = self.classifier(out1)
        return out1, out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class VGG_BN(nn.Module):
    def __init__(self, vgg_name):
        super(VGG_BN, self).__init__()
        if isinstance(vgg_name, list):
            self.features = self._make_layers(vgg_name)
            self.classifier = nn.Sequential(nn.Linear(vgg_name[-2], 10))
        else:
            self.features = self._make_layers(cfg[vgg_name])
        # self.classifier = nn.Sequential(nn.Linear(512, 10))
            self.classifier = nn.Sequential(nn.Linear(cfg[vgg_name][-2], 10))

    def forward(self, x):
        out = self.features(x)
        out1 = out.view(out.size(0), -1)
        out = self.classifier(out1)
        return out1, out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


if __name__ == "__main__":
    net = VGG('VGG11')
    summary = summary(net, (3, 34, 34))

    # macs, params = get_model_complexity_info(net, (3, 34, 34), as_strings=True,
    #                                          print_per_layer_stat=True, verbose=True)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    #
    # print('abcd')
