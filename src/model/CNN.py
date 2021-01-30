import torch.nn as nn
import torch.nn.functional as F
import torch
from ptflops import get_model_complexity_info
from torchsummary import summary
from utils import device


class Net(nn.Module):
    """LeNet 5 layer"""

    def __init__(self, num_classes=6, num_channel=3,
                 out1=64, out2=192, k_size=(3, 3), c_stride=(1, 1), p_kernel=(2, 2), fc1=9216, fc2=128):
        super(Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=num_channel, out_channels=out1, kernel_size=k_size, stride=c_stride),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(p_kernel),
            nn.Conv2d(in_channels=out1, out_channels=out2, kernel_size=k_size, stride=c_stride),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(p_kernel)
        )
        self.classifier = nn.Sequential(
            nn.Linear(fc1, fc2),
            nn.ReLU(inplace=True),
            nn.Linear(fc2, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        out1 = x.view(x.size(0), -1)
        x = self.classifier(out1)
        out2 = F.softmax(x, dim=1)
        return out1, out2


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 20, kernel_size=5, stride=1)
        self.bn1 = nn.BatchNorm2d(20)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(20, 50, kernel_size=5, stride=1)
        self.bn2 = nn.BatchNorm2d(50)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(800, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = self.maxpool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.maxpool2(self.relu2(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x


class LeNet5_layers(nn.Module):
    def __init__(self):
        super(LeNet5_layers, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(3, 5, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            Flatten(),
            nn.Linear(80, 50),
            nn.ReLU(inplace=True),
            nn.Linear(50, 10)
        )

    def forward(self, x):
        x = self.layers(x)
        x = F.softmax(x, dim=1)
        return x


class LeNet5_standard(nn.Module):
    def __init__(self, out_channel=2, out_channel2=5, fc_out=50):
        super(LeNet5_standard, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(1, out_channel, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(out_channel, out_channel2, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            Flatten(),
            nn.Linear(out_channel2 * 16, fc_out),
            nn.ReLU(inplace=True),
            nn.Linear(fc_out, 10)
        )

    def forward(self, x):
        x = self.layers(x)
        x = F.softmax(x, dim=1)
        return x


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class LeNet5(nn.Module):
    def __init__(self, in_channel, out1_channel, out2_channel, fc, out_classes, kernel_size, flatten_factor, padding=[0,0]):
        super(LeNet5, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channel, out1_channel, kernel_size=(1, kernel_size), stride=1, padding=(0, padding[0])),
            # nn.BatchNorm2d(out1_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1, 2)),
            nn.Conv2d(out1_channel, out2_channel, kernel_size=(1, kernel_size), stride=1, padding=(0, padding[1])),
            # nn.BatchNorm2d(out2_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1, 2)),
        )

        self.classifier = nn.Sequential(
            nn.Linear(flatten_factor * out2_channel, fc),
            nn.ReLU(inplace=True),
            nn.Linear(fc, out_classes),
        )

    def forward(self, x):
        x = self.features(x)
        out1 = x.view(x.size(0), -1)
        x = self.classifier(out1)
        out2 = F.softmax(x, dim=1)
        return out1, out2


class LeNet5_GROW_BN(nn.Module):
    def __init__(self, out1, out2, fc1):
        super(LeNet5_GROW_BN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, out1, kernel_size=5, stride=1),
            nn.BatchNorm2d(out1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(out1, out2, kernel_size=5, stride=1),
            nn.BatchNorm2d(out2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(out2 * 16, fc1),
            nn.ReLU(inplace=True),
            nn.Linear(fc1, 10),
        )

    def forward(self, x):
        x = self.features(x)
        out1 = x.view(x.size(0), -1)
        x = self.classifier(out1)
        out2 = F.softmax(x, dim=1)
        return out1, out2


class LeNet5_GROW_BN_STD(nn.Module):
    def __init__(self):
        super(LeNet5_GROW_BN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5, stride=1),
            nn.BatchNorm2d(10),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(10, 15, kernel_size=5, stride=1),
            nn.BatchNorm2d(15),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(240, 50),
            nn.ReLU(inplace=True),
            nn.Linear(50, 10),
        )

    def forward(self, x):
        x = self.features(x)
        out1 = x.view(x.size(0), -1)
        x = self.classifier(out1)
        out2 = F.softmax(x, dim=1)
        return out1, out2


class LeNet5_GROW_Gate(nn.Module):
    def __init__(self):
        super(LeNet5_GROW_Gate, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(2, 5, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(5 * 16, 10),
            nn.ReLU(inplace=True),
            nn.Linear(10, 10),
        )

    def forward(self, x):
        x = self.features(x)
        out1 = x.view(x.size(0), -1)
        x = self.classifier(out1)
        out2 = F.softmax(x, dim=1)
        return out1, out2


class LeNet5_GROW(nn.Module):
    def __init__(self):
        super(LeNet5_GROW, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(2, 5, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(5 * 16, 10),
            nn.ReLU(inplace=True),
            nn.Linear(10, 10),
        )

    def forward(self, x):
        x = self.features(x)
        out1 = x.view(x.size(0), -1)
        x = self.classifier(out1)
        out2 = F.softmax(x, dim=1)
        return out1, out2


class LeNet5_GROW_P(nn.Module):
    def __init__(self, out1=2, out2=5, fc1=10):
        super(LeNet5_GROW_P, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, out1, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(out1, out2, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(out2 * 16, fc1),
            nn.ReLU(inplace=True),
            nn.Linear(fc1, 10),
        )

    def forward(self, x):
        x = self.features(x)
        out1 = x.view(x.size(0), -1)
        x = self.classifier(out1)
        out2 = F.softmax(x, dim=1)
        return out1, out2


class LeNet5_P(nn.Module):
    def __init__(self, out1=2, out2=5, fc1=10):
        super(LeNet5_P, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, out1, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(out1, out2, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(out2 * 16, fc1),
            nn.ReLU(inplace=True),
            nn.Linear(fc1, 10),
        )

    def forward(self, x):
        x = self.features(x)
        out1 = x.view(x.size(0), -1)
        x = self.classifier(out1)
        out2 = F.softmax(x, dim=1)
        return out2


class LeNet5_GROW_STD(nn.Module):
    def __init__(self):
        super(LeNet5_GROW_STD, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(20, 50, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(50 * 16, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 10),
        )

    def forward(self, x):
        x = self.features(x)
        out1 = x.view(x.size(0), -1)
        x = self.classifier(out1)
        out2 = F.softmax(x, dim=1)
        return out1, out2


class LeNet5_GROW1(nn.Module):
    def __init__(self, in_channel, out1, out2, fc1, out_channel):
        super(LeNet5_GROW1, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out1, kernel_size=5, stride=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(out1, out2, kernel_size=5, stride=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(out2 * 16, fc1)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(fc1, out_channel)

    def forward(self, x):
        ac1 = self.conv1(x)
        ac2 = self.conv2(self.maxpool1(self.relu1(ac1)))
        ac = self.maxpool2(self.relu2(ac2))
        x = ac
        x = x.view(x.size(0), -1)
        ac3 = self.fc1(x)
        acac = self.relu3(ac3)
        ac4 = self.fc2(acac)
        out = F.softmax(ac4, dim=1)
        return ac1, ac2, ac3, ac4, out


class LeNet5_GROW2(nn.Module):
    def __init__(self, in_channel, out1, out2, fc1, out_channel):
        super(LeNet5_GROW2, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out1, kernel_size=5, stride=1)
        self.bn1 = nn.BatchNorm2d(out1)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(out1, out2, kernel_size=5, stride=1)
        self.bn2 = nn.BatchNorm2d(out2)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(out2 * 16, fc1)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(fc1, out_channel)

    def forward(self, x):
        ac1 = self.maxpool1(self.relu1(self.bn1(self.conv1(x))))
        ac2 = self.maxpool2(self.relu2(self.bn2(self.conv2(ac1))))
        x = ac2
        x = x.view(x.size(0), -1)
        ac3 = self.fc1(x)
        ac4 = self.fc2(ac3)
        out = F.softmax(ac4, dim=1)
        return ac1, ac2, ac3, ac4, out


if __name__ == '__main__':
    # kwargs = {'num_classes': 6,
    #           'num_channel': 8,
    #           'out1': 2,
    #           'out2': 5,
    #           'k_size': (1, 5),  # 3->576, 7->288
    #           'c_stride': (1, 1),
    #           'p_kernel': (1, 2),
    #           'fc1': 10,
    #           'fc2': 10}
    para_model = {'out1': 20, 'out2': 50, 'fc1': 500}
    # para_model = {'out1': 20, 'out2': 25, 'fc1': 36}
    # para_model = {'out1': 24, 'out2': 39, 'fc1': 43}
    # para_model = {'out1': 11, 'out2': 29, 'fc1': 25}
    # para_model = {'out1': 12, 'out2': 39, 'fc1': 42}

    # para_model = {'out1': 30, 'out2': 35, 'fc1': 35}
    # para_model = {'out1': 23, 'out2': 33, 'fc1': 38}

    model = LeNet5_GROW_P(**para_model)

    #
    # input = torch.randn(1, 1, 28, 28)
    # macs, params = profile(model, inputs=(input,))
    # macs, params = clever_format([macs, params], "%.3f")
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    # first one
    macs, params = get_model_complexity_info(model, (1, 28, 28), as_strings=True,
                                             print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    # summary(model, (1, 28, 28))
    # standard training
    # tasks = [i / 10 for i in range(1, 10)]
