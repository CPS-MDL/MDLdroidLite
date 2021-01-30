import torch
import torch.nn as nn
from model.CNN import LeNet5
import time
import numpy as np
from torch.utils.data import DataLoader
import os
from glob import glob
import re
from sklearn.model_selection import train_test_split

from utils import save_model, write_log, train, test, dir_path, count_parameters, weights_init
from data_loader import generate_data_loader, DataSet, DataSetNP
from utils import device
from model.mobileNet import NetS


def train_test(model, trainloader, testloader, learning_rate, epoch, dic_path, is_two=False):
    # initialize critierion
    criterion = nn.CrossEntropyLoss()

    # initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # start training
    start = time.time()

    # loss , acc, time
    test_loss = []
    test_acc = []
    batch_loss = []
    train_loss_mean = []
    train_loss_std = []
    time_in_epoch = []

    for e in range(epoch):
        # train
        train_loss = train(trainloader, model, criterion, optimizer, epoch=(e + 1), to_log=dic_path['path_to_log'], print_freq=param['print_freq'], two_output=is_two)

        batch_loss.append(train_loss)
        train_loss_mean.append(np.mean(batch_loss))
        train_loss_std.append(np.std(batch_loss))

        # test after train
        loss, acc = test(model, test_loader=testloader, criterion=criterion, to_log=dic_path['path_to_log'],
                         is_two_out=is_two)
        test_loss.append(loss)
        test_acc.append(acc)
        time_in_epoch.append(time.time() - start)

    # subject test lastly
    if subject_test:
        print('Subject test:')
        loss, acc = test(model, test_loader=subloader, criterion=criterion, to_log=dic_path['path_to_log'],
                         is_two_out=is_two)
        write_log('Subject test: {:.4f}'.format(acc), dic_path['path_to_log'])

    # total time
    total_time = time.time() - start

    # save final model
    save_model(model=model, path_to_model=(dic_path['path_to_model']).format(str='final'))

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


def obtain_ff(dic, width=128):
    if 'padding' in dic.keys():
        after_first = (width + dic['padding'][1] * 2 - dic['kernel_size'] + 1) // 2
        ff = (after_first + dic['padding'][1] * 2 - dic['kernel_size'] + 1) // 2
    else:
        after_first = (width - dic['kernel_size'] + 1) // 2
        ff = (after_first - dic['kernel_size'] + 1) // 2
    return ff


def data_path(path):
    dic_path = {
        'train_x_others': None,
        'train_y_others': None,
        'test_x_others': None,
        'test_y_others': None,
        'test_x_subject': None,
        'test_y_subject': None,
    }
    for key in dic_path.keys():
        f_name = '{}.npy'.format(key)
        dic_path[key] = os.path.join(path, f_name)
    return dic_path


def reglob(path, exp, invert=False):
    """glob.glob() style searching which uses regex

    :param exp: Regex expression for filename
    :param invert: Invert match to non matching files
    """

    m = re.compile(exp)

    if invert is False:
        res = [f for f in os.listdir(path) if m.search(f)]
    else:
        res = [f for f in os.listdir(path) if not m.search(f)]

    res = map(lambda x: "%s/%s" % (path, x,), res)
    return res


if __name__ == '__main__':
    dataset = 'myHealth'  # Har, EMG, myHealth
    model_name = 'LeNet'  # mobileNet , LSTM, CNN
    subject_test = False

    param = {
        'learning_rate': 0.0005,
        'epoch': 20,
        'batch_size': 32,
        'shuffle': True,
        'print_freq': 100
    }

    if dataset == 'Har':
        width = 128
        kwargs = {
            'in_channel': 9,
            'out1_channel': 36,
            'out2_channel': 72,
            'fc': 300,
            'out_classes': 6,
            'kernel_size': 32,
            'flatten_factor': 5
        }
        path_to_subject = "/Users/zber/ProgramDev/data_process_jupyter/Har/data_gen/subject_data"

    elif dataset == 'EMG':
        width = 100
        kwargs = {
            'in_channel': 8,
            'out1_channel': 36,
            'out2_channel': 72,
            'fc': 300,
            'out_classes': 6,
            'kernel_size': 12,
            'flatten_factor': 5,
            # 'padding': (0, 4),
        }
        path_to_subject = "/Users/zber/ProgramDev/data_process_jupyter/EMG/data_gen/subject_data"

    elif dataset == 'myHealth':
        width = 100
        kwargs = {
            'in_channel': 23,
            'out1_channel': 36,
            'out2_channel': 72,
            'fc': 300,
            'out_classes': 11,
            'kernel_size': 32,
            'flatten_factor': 5,
        }
        path_to_subject = "/Users/zber/ProgramDev/data_process_jupyter/MyHealth/gen_data/subject_data"

    # folders = [f.path for f in os.scandir(path_to_subject) if f.is_dir()]
    #
    # for f in folders:
    #     # subject name
    #     s_name = os.path.basename(f)
    #
    #     # obtain file path
    #     dic_path = data_path(f)
    #
    #     # create data loader
    #     trainset = DataSet(dic_path['train_x_others'], dic_path['train_y_others'])
    #     testset = DataSet(dic_path['test_x_others'], dic_path['test_y_others'])
    #     subtest = DataSet(dic_path['test_x_subject'], dic_path['test_y_subject'])
    #     trainloader = DataLoader(trainset, batch_size=param['batch_size'], shuffle=True)
    #     testloader = DataLoader(testset, batch_size=param['batch_size'], shuffle=True)
    #     subloader = DataLoader(subtest, batch_size=param['batch_size'], shuffle=True)
    #
    #     # create model
    #     ff = obtain_ff(kwargs, width)
    #     kwargs['flatten_factor'] = ff
    #     model = LeNet5(**kwargs)
    #     model.apply(weights_init)
    #     model = model.to(device)
    #
    # path = dir_path(
    #     '{}_{}_{}_'.format(dataset, model_name, s_name))
    #
    #     train_test(model, trainloader, testloader, learning_rate=param['learning_rate'], epoch=param['epoch'], dic_path=path, is_two=True)

    # load model
    # ff = obtain_ff(kwargs, width)
    # kwargs['flatten_factor'] = ff
    # model = LeNet5(**kwargs)
    # # model = NetS()
    # model.apply(weights_init)
    # model = model.to(device)

    # model.load_state_dict(torch.load("/Users/zber/ProgramDev/exp_pyTorch/results/EMG_LeNet_C36_72_F300_K12_20200615-014020/model_final.ckpt"))

    # load data
    # trainloader, testloader = generate_data_loader(batch_size=param['batch_size'], dataset=dataset, is_main=False)

    # Har 2
    # subject_path = "/Users/zber/ProgramDev/data_process_jupyter/Har/data_gen/subject_data/"

    # res = reglob(path_to_subject, r'subject_((2|4|9)|(10))$')

    # Emg
    # subject_path = "/Users/zber/ProgramDev/data_process_jupyter/EMG/data_gen/subject_data/"
    #
    # res = reglob(path_to_subject, r'subject_(3|7|1|21|15|29)_(4|8|2|22|16|30)$')

    # myHealth
    res = reglob(path_to_subject, r'subject_(1|2|3|4|5|6|7|8|9)$')

    criterion = nn.CrossEntropyLoss()

    dic_path = dir_path(
        '{}_{}_{}_'.format(dataset, model_name, 'specfic_user'))

    for path in res:
        path_xs = os.path.join(path, 'test_x_subject.npy')
        path_ys = os.path.join(path, 'test_y_subject.npy')

        x_data = np.load(path_xs)
        y_data = np.load(path_ys)

        train_x, test_x, train_y, test_y = train_test_split(x_data, y_data, test_size=0.2, random_state=123)

        trainset = DataSetNP(train_x,train_y)
        testset = DataSetNP(test_x,test_y)

        trainloader = DataLoader(trainset, batch_size=param['batch_size'], shuffle=True)
        testloader = DataLoader(testset, batch_size=param['batch_size'], shuffle=True)
        print(path)

        # load model
        ff = obtain_ff(kwargs, width)
        kwargs['flatten_factor'] = ff
        model = LeNet5(**kwargs)
        # model = NetS()
        model.apply(weights_init)
        model = model.to(device)

        train_test(model, trainloader, testloader, learning_rate=param['learning_rate'], epoch=param['epoch'],
                   dic_path=dic_path, is_two=True)

        loss, acc = test(model, test_loader=testloader, criterion=criterion, to_log=None,
                         is_two_out=True)

    # EMG
    # subject_path = "/Users/zber/ProgramDev/data_process_jupyter/EMG/data_gen/subject_data/"
    #
    # res = reglob(subject_path, r'subject_(3|7|1|21|15|29)_(4|8|2|22|16|30)$')
    #
    # criterion = nn.CrossEntropyLoss()
    #
    # for path in res:
    #     path_xs = os.path.join(path, 'test_x_subject.npy')
    #     path_ys = os.path.join(path, 'test_y_subject.npy')
    #     subtest = DataSet(path_xs, path_ys)
    #     subloader = DataLoader(subtest, batch_size=param['batch_size'], shuffle=True)
    #     print(path)
    #     loss, acc = test(model, test_loader=subloader, criterion=criterion, to_log=None,
    #                      is_two_out=True)

    # path_xs = "/Users/zber/ProgramDev/data_process_jupyter/Har/data_gen/subject_data/subject_2/test_x_subject.npy"
    # path_ys = "/Users/zber/ProgramDev/data_process_jupyter/Har/data_gen/subject_data/subject_2/test_y_subject.npy"

    # path_xs = "/Users/zber/ProgramDev/data_process_jupyter/Har/data_gen/data_npy_part/test_x_s.npy"
    # path_ys = "/Users/zber/ProgramDev/data_process_jupyter/Har/data_gen/data_npy_part/test_y_s.npy"

    # path = dir_path(
    #     '{}_{}_C{}_{}_F{}_K{}_'.format(dataset, model_name, kwargs['out1_channel'], kwargs['out2_channel'], kwargs['fc'], kwargs['kernel_size']))
    #
    # train_test(model, trainloader, testloader, learning_rate=param['learning_rate'], epoch=param['epoch'],
    #            dic_path=path, is_two=True)
