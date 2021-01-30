from model import CNN
import torchvision
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
from main import train
from main import test
from main import save_model
import torch.nn as nn
from grow.weights_distribution import generate_weights
import datetime
import time

# constant variable
FULLY_LAYER_MULTIPLIER = 16
FULLY_LAYER_FLATTEN = 144

# the network structure need to be changed,
t_channel = [8, 12, 16, 24]
t_fc = {8: 1152, 12: 1728, 16: 2304, 24: 3456}


# the t_channel and t_fc is predefined.
def generate_kwargs(index):
    # experiment settings for number of channels
    kwargs = {'num_classes': 10, 'num_channel': 1,
              'out1': t_channel[index], 'out2': t_channel[index], 'fc1': t_fc[t_channel[index]], 'fc2': 128,
              'k_size': 3,
              'c_stride': 1,
              'p_stride': 2,
              }
    return kwargs


def create_net(out):
    kwargs = {'num_classes': 10, 'num_channel': 1,
              'out1': out, 'out2': out, 'fc1': out * FULLY_LAYER_FLATTEN, 'fc2': out * FULLY_LAYER_MULTIPLIER,
              'k_size': 3,
              'c_stride': 1,
              'p_stride': 2,
              }
    net = CNN.Net(**kwargs)
    return net


if __name__ == '__main__':
    # growth mode
    mode = 'avg'

    # logging file directory
    parent_dir = os.path.dirname(os.getcwd())
    res_dir = os.path.join(parent_dir, 'results/')
    res_dir = res_dir + 'exp_growth_{}_{}_{}_'.format(t_channel[0], t_channel[len(t_channel) - 1], mode)
    res_dir = res_dir + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    # log path
    path_to_log = os.path.join(res_dir, 'log.txt')

    # global data setting
    batch_size = 16  # 100
    epoch = 1  # only 1 epoch for each model growth
    learning_rate = 0.001
    # the model configuration from index = 0
    i = 0

    # load data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    trainset = torchvision.datasets.MNIST(root='../data', train=True,
                                          download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=0)
    testset = torchvision.datasets.MNIST(root='../data', train=False,
                                         download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=0)

    # model settings
    kwargs = generate_kwargs(index=i)

    # crate 2 model, one is temporary and cureently use
    model = CNN.Net(**kwargs)
    temp_model = None
    criterion = nn.CrossEntropyLoss()

    # the variable needed to be changed
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    test_loss = []
    test_acc = []
    batch_loss = []
    time_array = []
    start = time.time()
    for n in range(len(t_channel)):
        for e in range(epoch):
            train_loss = train(trainloader, model, criterion, optimizer, epoch=(e + 1), to_log=path_to_log)
            time_array.append(time.time() - start)
            batch_loss.append(train_loss)
            # validate(testloader, model, criterion)
            loss, acc = test(model, test_loader=testloader, criterion=criterion, to_log=path_to_log)
            test_loss.append(loss)
            test_acc.append(acc)
            path_to_model_epoch = os.path.join(res_dir, ('model_C{}_E{}.ckpt'.format(t_channel[n], e)))
            save_model(model=model, path_to_model=path_to_model_epoch, mode='entire')


        # create temp_model
        if n + 1 < len(t_channel):
            model_weights = model.conv1.weight.data.numpy()
            new_weights = generate_weights(model_weights, t_channel[n + 1], mode=mode)
            temp_model = CNN.Net(**generate_kwargs(index=(n + 1)))
            with torch.no_grad():
                temp_model.conv1.weight.data = torch.from_numpy(new_weights)
            model = temp_model
            log = open(path_to_log, 'a')
            log.write('Model from C{} to C{}'.format(t_channel[n], t_channel[n + 1]))
            log.close()
            test(model, test_loader=testloader, criterion=criterion, to_log=path_to_log)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = open(path_to_log, 'a')
    for i, l in enumerate(batch_loss):
        log.write('epoch_{}_batch_train_loss:\n{}\n\n'.format(i, l))
    log.write('time:\n{}'.format(time_array))
    log.close()

# standard
# kwargs = {'num_classes': 10, 'num_channel': 1,
#           'out1': 20, 'out2': 50, 'fc1': 8450, 'fc2': 500,
#           'k_size': 2,
#           'c_stride': 1,
#           'p_stride': 2,
#           }

print('OK')
