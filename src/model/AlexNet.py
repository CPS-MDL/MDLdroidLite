import torch
from torch import nn
from torchsummary import summary


class AlexNetSTD(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNetSTD, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.classifier(x)
        return x


class AlexNet(nn.Module):
    """
    kwargs: in1,in2,in3,in4,in5
            out1,out2,out3,out4,out5
            kernel_size,
            stride,
            padding,

    """

    def __init__(self, num_classes=6, num_channel=3,
                 out1=64, out2=192, out3=384, out4=256, out5=256, f1=4096, f2=4096, kernel_size=(1, 3), stride=1,
                 padding=0):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(num_channel, out1, kernel_size=(1, 11), stride=(1, 4), padding=(0, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2)),
            nn.Conv2d(out1, out2, kernel_size=(1, 5), padding=(0, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2)),
            nn.Conv2d(out2, out3, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(out3, out4, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(out4, out5, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2)),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1 * 3))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(out5 * 3 * 3, f1),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(f1, f2),
            nn.ReLU(inplace=True),
            nn.Linear(f2, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x


if __name__ == '__main__':
    kwargs = {'num_classes': 8, 'num_channel': 8,
              'out1': 8, 'out2': 8, 'out3': 16, 'out4': 16, 'out5': 16, 'f1': 300, 'f2': 300}
    model = AlexNet(**kwargs)
    summary(model, (3, 1, 150))
