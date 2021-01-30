import torch
from torchsummary import summary
import numpy as np
import torch.nn as nn
from model.CNN import LeNet5_GROW, LeNet, LeNet5, LeNet5_layers
import time
import data_loader
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
from utils import save_model, write_log, train, test, weights_init, dir_path, count_parameters
import torch.optim as optim

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

if __name__ == '__main__':
    model = LeNet5_layers()
    for para in list(model.parameters()):
        a = para
        print('para')


    # model = LeNet5_layers()
    # for i, layers in enumerate(model.layers):
    #     weights = layers.weight.data.numpy()
    #     result = np.mean(weights, 2)
    #     print(result)
    #
    # summary(model, (1, 24, 24))

    # for name, p in model.named_parameters():
    #     print('a')
    #
    # for x, module in model._modules.items():
    #     print('a')

    # for x, module in model._modules.items():
    #     print('a')
    # for i, _ in enumerate(model.classifier):
    #     print('b')
