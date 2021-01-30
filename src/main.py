import torch
import torch.nn as nn
from model import mobileNetv1, CNN, TCN, LSTM
from model.CNN import LeNet5
from model.AlexNet_pytorch import AlexNet
from model.AlexNet import AlexNetSTD
from model.VGG import VGG
import time
import numpy as np
import data_loader
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
from utils import save_model, write_log, train, test, weights_init, dir_path, count_parameters
import torch.optim as optim
from grow.grow_multilayer import grow_layers
from data_loader import generate_data_loader

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# device = torch.device('cpu')

def percentage(list):
    per_list = []
    sum = np.sum(list)
    for i in list:
        per = i / sum
        per_list.append(per)
    return per_list


def generate_mode(kwargs, model_name='CNN'):
    model = None
    if model_name == 'mobileNet':
        model = mobileNetv1.Net()
    if model_name == 'LSTM':
        num_channels = 8
        num_classes = 8
        model = LSTM.RNN(input_size=num_channels, num_classes=num_classes, **kwargs)
    if model_name == 'CNN':
        model = CNN.Net(**kwargs)
        # model.apply(weights_init)  # Customer the model weights
    if model_name == 'TCN':
        model = TCN.TCN(**kwargs)
    if model_name == 'AlexNet':
        model = AlexNet()
    if 'VGG' in model_name:
        model = VGG(model_name)
    return model

#
# def generate_data_loader(batch_size, dataset='MNIST', is_main=True):
#     if dataset == 'MNIST':
#         transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.1307,), (0.3081,)),
#         ])
#         if is_main:
#             root_dir = '../data'
#         else:
#             root_dir = '../../data'
#         trainset = torchvision.datasets.MNIST(root=root_dir, train=True,
#                                               download=True, transform=transform)
#         trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
#                                                   shuffle=True, num_workers=0)
#         testset = torchvision.datasets.MNIST(root=root_dir, train=False,
#                                              download=True, transform=transform)
#         testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
#                                                  shuffle=False, num_workers=0)
#     elif dataset == 'CIFAR10':
#         transform = transforms.Compose(
#             [transforms.ToTensor(),
#              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#
#         trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
#                                                 download=True, transform=transform)
#         trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
#                                                   shuffle=True, num_workers=0)
#         testset = torchvision.datasets.CIFAR10(root='../data', train=False,
#                                                download=True, transform=transform)
#         testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
#                                                  shuffle=False, num_workers=0)
#     elif dataset == 'EMG':
#         num_channels = 8
#         width = 20
#         Mode = '2D'
#         path_to_x_train = '/Users/zber/ProgramDev/exp_pyTorch/data/EMG/W20_overlap_x_train.npy'
#         path_to_y_train = '/Users/zber/ProgramDev/exp_pyTorch/data/EMG/W20_overlap_y_train.npy'
#         path_to_x_test = '/Users/zber/ProgramDev/exp_pyTorch/data/EMG/W20_overlap_x_test.npy'
#         path_to_y_test = '/Users/zber/ProgramDev/exp_pyTorch/data/EMG/W20_overlap_y_test.npy'
#
#         trainset = data_loader.Data(x=path_to_x_train, y=path_to_y_train, num_channels=num_channels, width=width,
#                                     Mode=Mode)
#         testset = data_loader.Data(x=path_to_x_test, y=path_to_y_test, num_channels=num_channels, width=width,
#                                    Mode=Mode)
#         trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
#         testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)
#     elif dataset == 'Har':
#         num_channels = 9
#         width = 128
#         Mode = '2D'
#         # path_to_x_train = '/Users/zber/Desktop/Desktop2/1_Data/data/Har/normal_train_X.npy'
#         # path_to_y_train = '/Users/zber/Desktop/Desktop2/1_Data/data/Har/normal_train_y.npy'
#         # path_to_x_test = '/Users/zber/Desktop/Desktop2/1_Data/data/Har/normal_test_X.npy'
#         # path_to_y_test = '/Users/zber/Desktop/Desktop2/1_Data/data/Har/normal_test_y.npy'
#
#         path_to_x_train = '/Users/zber/ProgramDev/data_process_jupyter/Har/data_gen/data_npy/train_X.npy'
#         path_to_y_train = '/Users/zber/ProgramDev/data_process_jupyter/Har/data_gen/data_npy/train_y.npy'
#         path_to_x_test = '/Users/zber/ProgramDev/data_process_jupyter/Har/data_gen/data_npy/test_X.npy'
#         path_to_y_test = '/Users/zber/ProgramDev/data_process_jupyter/Har/data_gen/data_npy/test_y.npy'
#
#         trainset = data_loader.Data(x=path_to_x_train, y=path_to_y_train, num_channels=num_channels, width=width,
#                                     Mode=Mode)
#         testset = data_loader.Data(x=path_to_x_test, y=path_to_y_test, num_channels=num_channels, width=width,
#                                    Mode=Mode)
#         trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
#         testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)
#
#     return trainloader, testloader


def train_test(model, trainloader, testloader, learning_rate, epoch, dic_path, str_channel, print_grad=False,
               is_two=False):

    grow_epoch = []  #
    # initialize critierion
    criterion = nn.CrossEntropyLoss()

    # initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

    # start training
    start = time.time()

    # loss , acc, time
    test_loss = []
    test_acc = []
    batch_loss = []
    train_loss_mean = []
    train_loss_std = []
    time_in_epoch = []
    layers_l2 = []
    layers_h_score = []

    for e in range(epoch):
        epoch_start = time.time()
        # train
        if print_grad:
            train_loss, grads, inputs, target = train(trainloader, model, criterion, optimizer,
                                                      epoch=(e + 1), to_log=dic_path['path_to_log'],
                                                      print_grad=True, two_output=is_two)  #
            if e in grow_epoch:
                feature_out, output = model(inputs)
                loss = criterion(output, target)
                print('batch_loss before growing:{}'.format(loss))
                print('test loss and acc before growing:')
                test(model, test_loader=testloader, criterion=criterion, to_log=dic_path['path_to_log'],
                     is_two_out=is_two)
                model = grow_layers(model, [2, 3, 6])
                feature_out, output = model(inputs)
                loss = criterion(output, target)
                print('batch_loss after growing:{}'.format(loss))
                print('test loss and acc after growing:')
                test(model, test_loader=testloader, criterion=criterion, to_log=dic_path['path_to_log'],
                     is_two_out=is_two)
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            # h score
            per_l2 = [percentage(grad) for grad in grads]
            train_loss = np.reshape(train_loss, (1, -1))
            per_l2 = np.transpose(per_l2, (1, 0))
            per_loss_list = per_l2 * train_loss + train_loss
            for gradients in per_loss_list:
                layers_h_score.append(gradients.tolist())
            # l2
            grads = np.transpose(grads, (1, 0))
            for gradients in grads:
                layers_l2.append(gradients.tolist())

        else:
            train_loss = train(trainloader, model, criterion, optimizer, epoch=(e + 1), to_log=dic_path['path_to_log'])

        batch_loss.append(train_loss)
        train_loss_mean.append(np.mean(batch_loss))
        train_loss_std.append(np.std(batch_loss))

        # test after train
        loss, acc = test(model, test_loader=testloader, criterion=criterion, to_log=dic_path['path_to_log'],
                         is_two_out=is_two)
        test_loss.append(loss)
        test_acc.append(acc)
        time_in_epoch.append(time.time() - epoch_start)

        # save model for each epoch
        path_to_model_epoch = os.path.join(dic_path['res_dir'], ('model_E{}_{}.pkl'.format(e, str_channel)))
        save_model(model=model, path_to_model=path_to_model_epoch, mode='entire')

    # total time
    total_time = time.time() - start

    # save final model
    save_model(model=model, path_to_model=(dic_path['path_to_model']).format(str=str_channel))

    # the size of parameters
    para_str = 'The number of parameters in the model is : {}\n'.format(count_parameters(model))

    # total time
    time_str = 'The total training time is {}\n'.format(total_time)

    # output
    print(time_str)
    print(para_str)

    # save log
    write_log(time_str, dic_path['path_to_log'])
    write_log(para_str, dic_path['path_to_log'])
    write_log('train time in each epoch:{}\n'.format(time_in_epoch), dic_path['path_to_log'])
    write_log('Loss:{}\n'.format(test_loss), dic_path['path_to_log'])
    write_log('Accuracy:{}\n'.format(test_acc), dic_path['path_to_log'])
    write_log('Train_loss_mean:{}\n'.format(train_loss_mean), dic_path['path_to_log'])
    write_log('Train_loss_std:{}\n'.format(train_loss_std), dic_path['path_to_log'])
    for index, item in enumerate(batch_loss):
        str = '\n{}_Epoch_{} = {};\n'.format(str_channel, index + 1, item)
        write_log(str, dic_path['path_to_log'])

    if print_grad:
        layers_h_score = np.reshape(layers_h_score, (epoch, -1, len(layers_h_score[0])))
        layers_l2 = np.reshape(layers_l2, (epoch, -1, len(layers_l2[0])))
        for index, layer in enumerate(layers_h_score):
            mean_gradients = [np.mean(np.asarray(grad)) for grad in layer]
            str = '\nlayer{}_h_score_mean = {};\n'.format(index, mean_gradients)
            write_log(str, dic_path['path_to_log'])
        for index, layer in enumerate(layers_l2):
            mean_gradients = [np.mean(np.asarray(grad)) for grad in layer]
            str = '\nlayer{}_l2_mean = {};\n'.format(index, mean_gradients)
            write_log(str, dic_path['path_to_log'])

        np.save(os.path.join(dic_path['res_dir'], 'batch_loss.npy'), batch_loss)
        np.save(os.path.join(dic_path['res_dir'], 'layers_h_score.npy'), layers_h_score)
        np.save(os.path.join(dic_path['res_dir'], 'layers_l2.npy'), layers_l2)


if __name__ == '__main__':
    dataset = 'CIFAR10'  # CIFAR10,MNIST,EMG
    model_name = 'VGG11'  # mobileNet , LSTM, CNN
    t_channel = 8  # [8,12,16,24] for experiment，or standard

    kwargs_CIFAR10 = {'num_classes': 6, 'num_channel': 8,
                      'out1': 8, 'out2': 8, 'out3': 16, 'out4': 16, 'out5': 16, 'f1': 2304, 'f2': 300}

    kwargs_har = {
        'in_channel': 9,
        'out1_channe': 36,
        'out2_channel': 72,
        'fc': 300,
        'out_classes': 6,
        'kernel_size': 5,
        'flatten_factor': 5
    }

    kwargs_EMG = {'num_classes': 6,
                  'num_channel': 8,
                  'out1': 36,
                  'out2': 72,
                  'k_size': (1, 3),  # 3->576, 7->288
                  'c_stride': (1, 1),
                  'p_kernel': (1, 2),
                  'fc1': 3 * 72,
                  'fc2': 300}

    kwargs_standard = {'num_classes': 10, 'num_channel': 1,
                       'out1': 20, 'out2': 50, 'fc1': 800, 'fc2': 500,
                       'k_size': (5, 5),
                       'c_stride': 1,
                       'p_kernel': (2, 2),
                       }

    kwargs_ours = {'num_classes': 10, 'num_channel': 1,
                   'out1': 9, 'out2': 16, 'fc1': 256, 'fc2': 300,
                   'k_size': (5, 5),
                   'c_stride': 1,
                   'p_kernel': (2, 2),
                   }

    kwargs_cnn = {'num_classes': 10, 'num_channel': 1,
                  'out1': t_channel, 'out2': t_channel, 'fc1': t_channel, 'fc2': 128,
                  'k_size': 3,
                  'c_stride': 1,
                  'p_stride': 2,
                  }

    param = {
        'learning_rate': 0.0005,
        'epoch': 3,
        'batch_size': 25,
        'shuffle': True,
        'kernel_size': 5,
        'model_str': model_name
    }

    # CNN
    # path = dir_path(
    #     '{}_{}_C{}_{}_K{}_'.format(dataset, model_name, kwargs_ours['out1'], kwargs_ours['out2'], param['kernel_size']))

    # AlexNet
    for i in range(1, 2):
        path = dir_path(
            '{}_{}_'.format(dataset, model_name))

        # generate model
        model = generate_mode(None, model_name)
        # save_model(model=model, path_to_model="/Users/zber/Desktop/model.ckt", mode='entire')
        # model = CNN.Net(**kwargs_EMG)
        # model = CNN.LeNet5_GROW()
        # model = LeNet5(**kwargs_har)

        model = model.to(device)

        # generate data loader
        trainloader, testloader = generate_data_loader(batch_size=param['batch_size'], dataset=dataset)
        # training and test dataset
        train_test(model, trainloader, testloader, learning_rate=param['learning_rate'], epoch=param['epoch'],
                   dic_path=path, str_channel=param['model_str'], print_grad=False, is_two=False)
