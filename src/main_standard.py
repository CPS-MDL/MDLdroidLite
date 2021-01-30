import torch
from torch import nn
from main import generate_data_loader
from utils import test, test1
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import numpy as np
import os
from utils import write_log

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# torch.manual_seed(1234)
# np.random.seed(1234)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False


class LeNet5_GROW(nn.Module):
    def __init__(self, out1=2, out2=5, fc1=10):
        super(LeNet5_GROW, self).__init__()

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


if __name__ == '__main__':
    path = "/Users/zber/ProgramDev/exp_pyTorch/results/std_test"
    if not os.path.exists(path):
        os.makedirs(path)

    para_list = [
        {'out1': 2, 'out2': 5, 'fc1': 10},
        {'out1': 4, 'out2': 10, 'fc1': 20},
        {'out1': 6, 'out2': 11, 'fc1': 30},
        {'out1': 8, 'out2': 14, 'fc1': 40},
        {'out1': 10, 'out2': 17, 'fc1': 50},
    ]

    e_list = [0, 3, 6, 9, 10]

    for n in range(1):
        # layers = find_layers(model)
        trainloader, testloader = generate_data_loader(batch_size=100, dataset='MNIST')

        criterion = nn.CrossEntropyLoss()
        for e, para in zip(e_list, para_list):
            p_to_model = os.path.join(path, 'model{}.ckpt'.format(e))
            # torch.manual_seed(1535)  # torch.manual_seed(1535)
            model = LeNet5_GROW(**para)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
            for e_index in range(e):
                for i, (inputs, target) in enumerate(trainloader):
                    model.train()
                    inputs = inputs.to(device)
                    target = target.to(device)
                    optimizer.zero_grad()
                    output = model(inputs)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()

            p_to_log = os.path.join(path, 'log{}.txt'.format(e))
            _, acc = test(model, testloader, criterion, to_log=p_to_log)
            pred_list, cor_list = test1(model, testloader, criterion)
            pred_list = np.reshape(pred_list, (-1))
            cor_list = np.reshape(cor_list, (-1))
            write_log(str(confusion_matrix(cor_list, pred_list))+'\n', p_to_log)
            print(confusion_matrix(cor_list, pred_list))

            # a,b bound calculate
            layer_i = 0
            for seq in model.children():
                for module in seq.children():
                    classname = module.__class__.__name__
                    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
                        layer_i = layer_i + 1
                        write_log('Layer{}:\n'.format(layer_i), p_to_log)

                        tensor = module.weight
                        weight_min = torch.min(tensor).tolist()
                        weight_max = torch.max(tensor).tolist()
                        weight_sd = torch.std(tensor).tolist()
                        weight_mean = torch.mean(tensor).tolist()

                        tensor_bias = module.bias
                        bias_min = torch.min(tensor_bias).tolist()
                        bias_max = torch.max(tensor_bias).tolist()
                        bias_sd = torch.std(tensor_bias).tolist()
                        bias_mean = torch.mean(tensor_bias).tolist()

                        str1 = 'Weight: {:1.5f}|{:1.5f}|{:1.5f}|{:1.5f}|\n'.format(weight_min, weight_max, weight_sd,
                                                                                 weight_mean)
                        write_log(str1, p_to_log)
                        str2 = 'Bias: {:1.5f}|{:1.5f}|{:1.5f}|{:1.5f}|\n'.format(bias_min, bias_max, bias_sd, bias_mean)
                        write_log(str2, p_to_log)

            # save model
            torch.save(model.state_dict(), p_to_model)

        # if acc > 85:

        # write_log('seed:{}, acc:{}\n'.format(seed, acc), p)

        #     _, acc = test(model, testloader, criterion)
        #     if acc < 88:
        #         print('acc: {}, remove model{}\n'.format(acc, n))
        #         os.remove(p)
