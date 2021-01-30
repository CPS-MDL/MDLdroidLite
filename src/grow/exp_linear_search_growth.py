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
from grow.weights_distribution import generate_weights, gen_fc_weights
import datetime
import time

# constant variable
FULLY_LAYER_MULTIPLIER = 16
FULLY_LAYER_FLATTEN = 144


# # the network structure need to be changed,
# t_channel = [8, 12, 16, 24]
# t_fc = {8: 1152, 12: 1728, 16: 2304, 24: 3456}


# the t_channel and t_fc is predefined.
# def generate_kwargs(index):
#     # experiment settings for number of channels
#     kwargs = {'num_classes': 10, 'num_channel': 1,
#               'out1': t_channel[index], 'out2': t_channel[index], 'fc1': t_fc[t_channel[index]], 'fc2': 128,
#               'k_size': 3,
#               'c_stride': 1,
#               'p_stride': 2,
#               }
#     return kwargs


def create_net(out):
    kwargs = {'num_classes': 10, 'num_channel': 1,
              'out1': out, 'out2': out, 'fc1': out * FULLY_LAYER_FLATTEN, 'fc2': out * FULLY_LAYER_MULTIPLIER,
              'k_size': 3,
              'c_stride': 1,
              'p_stride': 2,
              }
    net = CNN.Net(**kwargs)
    return net


def terminate(loss):
    length = len(loss)
    if length > 1 and loss[length - 1] > loss[length - 2]:
        return True
    return False


def create_net_based_old(model, out_channel, g_mdoe, i_mode, b_to_s=False):
    conv1_model_weights = model.conv1.weight.data.numpy()
    conv1_new_weights = generate_weights(conv1_model_weights, out_channel, generate_mode=g_mdoe, insert_mode=i_mode,
                                         big_to_small=b_to_s)
    conv2_model_weights = model.conv2.weight.data.numpy()
    conv2_new_weights = generate_weights(conv2_model_weights, out_channel, generate_mode=g_mdoe, insert_mode=i_mode,
                                         num_layer=2, big_to_small=b_to_s)
    fc1_weights = model.fc1.weight.data.numpy()
    fc1_new_weights = gen_fc_weights(fc1_weights, num_in=out_channel * FULLY_LAYER_FLATTEN,
                                     num_out=out_channel * FULLY_LAYER_MULTIPLIER, big_to_small=b_to_s)
    fc2_weights = model.fc2.weight.data.numpy()
    fc2_new_weights = gen_fc_weights(fc2_weights, num_in=out_channel * FULLY_LAYER_MULTIPLIER,
                                     num_out=num_classes, big_to_small=b_to_s)
    temp_model = create_net(out=out_channel)
    # fully layer
    with torch.no_grad():
        temp_model.conv1.weight.data = torch.from_numpy(conv1_new_weights)
        temp_model.conv2.weight.data = torch.from_numpy(conv2_new_weights)
        temp_model.fc1.weight.data = torch.from_numpy(fc1_new_weights)
        temp_model.fc2.weight.data = torch.from_numpy(fc2_new_weights)
    new_model = temp_model
    return new_model


if __name__ == '__main__':
    # growth mode
    g_mode = 'avg'
    i_mode = 'stack'

    # logging file directory
    parent_dir = os.path.dirname(os.getcwd())
    res_dir = os.path.join(parent_dir, 'results/')
    res_dir = res_dir + 'exp_growth_linear_search_Mode_{}_{}_'.format(g_mode, i_mode)
    res_dir = res_dir + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    # log path
    path_to_log = os.path.join(res_dir, 'log.txt')

    # global data setting
    batch_size = 32  # 100
    epoch = 1  # only 1 epoch for each model growth
    learning_rate = 0.0001
    num_classes = 10
    # the model configuration from index = 0
    count_epoch = 0

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

    # start from 3 channels
    out_channel = 3

    # crate 2 model, one is temporary and cureently use
    model = create_net(out=out_channel)
    old_model = None
    criterion = nn.CrossEntropyLoss()

    # the variable needed to be changed
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    test_loss = []
    test_acc = []
    batch_loss = []
    time_array = []
    channel_history = []
    start = time.time()
    is_terminate = False

    while True:
        for e in range(epoch):
            train_loss = train(trainloader, model, criterion, optimizer, epoch=(e + 1), to_log=path_to_log)
            time_array.append(time.time() - start)
            batch_loss.append(train_loss)
            # validate(testloader, model, criterion)
            loss, acc = test(model, test_loader=testloader, criterion=criterion, to_log=path_to_log)
            test_loss.append(loss)
            test_acc.append(acc)
            path_to_model_epoch = os.path.join(res_dir, ('model_C{}_E{}.ckpt'.format(out_channel, e)))
            save_model(model=model, path_to_model=path_to_model_epoch, mode='entire')
            channel_history.append(out_channel)
            count_epoch += 1
        if count_epoch == 20:
            break

        # growth
        out_channel = out_channel + 1
        model = create_net_based_old(model, out_channel, g_mode, i_mode)
        log = open(path_to_log, 'a')
        print('Model from C{} to C{}\ntest after transfer model:\n'.format(out_channel - 1, out_channel))
        log.write('Model from C{} to C{}\ntest after transfer model:'.format(out_channel - 1, out_channel))
        log.close()
        test(model, test_loader=testloader, criterion=criterion, to_log=path_to_log)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        if channel_history[count_epoch-1] == 17:
            epoch = 20 - count_epoch

    log = open(path_to_log, 'a')
    log.write('\nTotal time used for each epoch:{}\n'.format(time_array))
    for i, l in enumerate(batch_loss):
        log.write('\nepoch_{}_C{}_train_batch_loss={};\n'.format(i, channel_history[i], l))
    log.close()

print('OK')
