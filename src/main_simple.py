import torch
from torch import nn
from main import generate_data_loader
from utils import test, test1
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import numpy as np
import os
from utils import write_log, weights_init
from random import randint
from main_subject import obtain_ff
from model.CNN import LeNet5
from model.mobileNet import NetS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# torch.manual_seed(1234)
# np.random.seed(1234)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False


class LeNet5_GROW(nn.Module):
    def __init__(self):
        super(LeNet5_GROW, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=(5, 5), stride=1),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(2, 5, kernel_size=(5, 5), stride=1),
            nn.BatchNorm2d(5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(5 * 16, 10),
            nn.ReLU(inplace=True),
            nn.Linear(10, 10),
        )

    def forward(self, x):
        # x = x.view(x.size(0), -1)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        out = F.softmax(x, dim=1)
        return out


if __name__ == '__main__':
    path = "/Users/zber/ProgramDev/exp_pyTorch/results/model"

    dataset = 'Har'
    if dataset == "Har":
        param = {
            'lr': 0.0005,
            'epoch': 20,
            'batch_size': 32,
            'is_bn': False,
        }
        input_shape = (9, 1, 128)
        width = 128

        kwargs = {
            'in_channel': 9,
            'out1_channel': 3,
            'out2_channel': 6,
            'out3_channel': 12,
            'out4_channel': 25,
            'out_classes': 6,
            'kernel_size': 14,
            'avg_factor': 2
        }
        path_to_subject = "/Users/zber/ProgramDev/data_process_jupyter/Har/data_gen/subject_data"

        # ff = obtain_ff(kwargs, width)
        # kwargs['flatten_factor'] = ff

    for n in range(20):
        seed = randint(0, 10000)
        torch.manual_seed(seed)  # torch.manual_seed(1535) 1535

        model = NetS(**kwargs)
        # model = NetS()
        # model.apply(weights_init)
        model = model.to(device)

        # model = LeNet5_GROW()
        p = os.path.join(path, 'model{}.ckpt'.format(n))
        torch.save(model.state_dict(), p)
        # layers = find_layers(model)
        # trainloader, testloader = generate_data_loader(batch_size=100, dataset='MNIST')
        trainloader, testloader = generate_data_loader(batch_size=32, dataset=dataset)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
        criterion = nn.CrossEntropyLoss()
        for e in range(1):
            for i, (inputs, target) in enumerate(trainloader):
                model.train()
                inputs = inputs.to(device)
                target = target.to(device)
                optimizer.zero_grad()
                output = model(inputs)
                # _, output = model(inputs)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
        _, acc = test(model, testloader, criterion, is_two_out=False)
        # if acc > 70:
        p = os.path.join(path, 'seed.txt')
        print('seed:{}, acc:{}\n'.format(seed, acc))
        write_log('seed:{}, acc:{}\n'.format(seed, acc), p)
        # pred_list, cor_list = test1(model, testloader, criterion, is_two=True)
        #     _, acc = test(model, testloader, criterion)
        #     if acc < 88:
        #         print('acc: {}, remove model{}\n'.format(acc, n))
        #         os.remove(p)
        # pred_list = np.reshape(pred_list, (-1))
        # cor_list = np.reshape(cor_list, (-1))
        # print(confusion_matrix(cor_list, pred_list))
