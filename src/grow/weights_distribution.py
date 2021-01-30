import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import datetime
import torchvision.transforms as transforms
import torchvision
import math

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def plot_conv_weights(epochs, in_channels):
    fig, axs = plt.subplots(3)
    n_fig = 0
    for i in epochs:
        path_to_model_location = 'EMG_CNN_C24_K320200102-191749/model_epoch_{}.ckpt'.format(i)
        path_to_model = os.path.join(res_dir, path_to_model_location)
        model = torch.load(path_to_model)
        conv1_weight = model.conv2.weight.data.numpy()
        conv1_plot = np.reshape(conv1_weight, (in_channels, -1))
        for fm in conv1_plot:
            axs[n_fig].plot(fm)
        n_fig += 1
    plt.show()


def plot_fully_weights(epochs, in_channels, x, colors, markers):
    i_color = 0
    for i in epochs:
        path_to_model_location = 'EMG_CNN_C8_K320200102-145312/model_epoch_{}.ckpt'.format(i)
        path_to_model = os.path.join(res_dir, path_to_model_location)
        model = torch.load(path_to_model)
        fc_weight = model.fc2.weight.data.numpy()
        fc_plot = np.reshape(fc_weight, (in_channels), order='F')
        plt.scatter(x, fc_plot, s=5, c=colors[i_color], marker=markers[i_color])
        i_color += 1
    plt.show()


def gaussian(model, txt_loc):
    modules = [module for module in model.modules()]
    c = 1
    l = 2
    log = open(txt_loc, 'a')
    for module in modules:
        if isinstance(module, nn.Conv2d):
            w = module.weight.data.numpy()
            name = 'Conv{}'.format(c)
            c += 1
            mean = np.mean(w)
            std = np.std(w)
            log.write('Layer_{}: mean={}, std={}\n'.format(name, mean, std))
        if isinstance(module, nn.Linear):
            w = module.weight.data.numpy()
            name = 'FC{}'.format(c)
            c += 1
            mean = np.mean(w)
            std = np.std(w)
            log.write('Layer_{}: mean={}, std={}\n'.format(name, mean, std))
    log.close()


# this function is used to print the std and mean for each epoch
def gaussian_models(models_loc, txt_loc, layer):
    mean = []
    std = []
    log = open(txt_loc, 'a')
    for model_loc in models_loc:
        model = torch.load(model_loc)
        modules = [module for module in model.modules()]
        w = modules[layer].weight.data.numpy()
        mean.append(np.mean(w))
        std.append(np.std(w))
    log.write('mean={}\n std={}\n'.format(mean, std))


# write mean and std to files.
def random_from_models(models_loc, txt_loc, layer):
    mean = []
    std = []
    log = open(txt_loc, 'a')
    for model_loc in models_loc:
        model = torch.load(model_loc)
        modules = [module for module in model.modules()]
        w = modules[layer].weight.data.numpy()
        np.random.shuffle(w)
        w8 = w[:8, :, :, :]
        mean.append(np.mean(w8))
        std.append(np.std(w8))
    log.write('mean={}\n std={}\n'.format(mean, std))


def ratio_channel_large_then_mean(models_loc, txt_loc, layer):
    # count ratio for different channel
    ratio_in_channel_layer = []
    for model_loc in models_loc:
        model = torch.load(model_loc)
        modules = [module for module in model.modules()]
        w = modules[layer].weight.data.numpy()
        ratio_in_layers = []
        for channel in w:
            abs_channel = np.absolute(channel)
            mean_abs_channel = np.mean(abs_channel)
            count_greater = len(abs_channel[np.where(abs_channel > mean_abs_channel)])
            count_flatten = len(np.reshape(channel, (-1)))
            ratio = count_greater / count_flatten
            ratio_in_layers.append(ratio)
        ratio_in_channel_layer.append(ratio_in_layers)
    return ratio_in_channel_layer


def ratio_layer_large_than_mean(models_loc, txt_loc, layer):
    # count ratio for different channel
    ratio_in_layers = []
    for model_loc in models_loc:
        model = torch.load(model_loc)
        modules = [module for module in model.modules()]
        w = modules[layer].weight.data.numpy()
        abs_channel = np.absolute(w)
        mean_abs_channel = np.mean(abs_channel)
        count_greater = len(abs_channel[np.where(abs_channel > mean_abs_channel)])
        count_flatten = len(np.reshape(w, (-1)))
        ratio = count_greater / count_flatten
        ratio_in_layers.append(ratio)
    return ratio_in_layers


# weights is numpy format, and channel is how many you want. #generate_mode='avg',
def gen_fc_weights(weights, num_in, num_out, big_to_small=False, insert_mode='stack'):
    min = np.min(weights)
    max = np.max(weights)
    old_out, old_in = weights.shape

    new_weights = np.random.uniform(min / 10, max / 10, num_in * num_out)
    new_weights = np.reshape(new_weights, (num_out, num_in))
    if big_to_small:
        # np.random.shuffle(weights)
        new_weights = weights[:num_out, :num_in]
        return new_weights.astype(np.float32)
    if insert_mode == 'stack':
        new_weights[:old_out, :old_in] = weights
    if insert_mode == 'shuffle':
        new_weights[:old_out, :old_in] = weights
        np.random.shuffle(new_weights)
    if insert_mode == 'insert':
        num_insert = num_out - old_out  # calculate how many times needed to insert new weights
        old_i = 0
        i = 0
        skip = 1 + 1
        while old_i < num_insert:
            new_weights[i, :old_in] = weights[old_i]
            i = i + skip
            old_i = old_i + 1
        new_weights[i:, :old_in] = weights[old_i:, :]
    return new_weights.astype(np.float32)



# weights is numpy format, and channel is how many you want.
def generate_weights(weights, channel, generate_mode='avg', insert_mode='stack', num_layer=1,
                     big_to_small=False):
    shape = weights.shape
    o_c, i_c, k_h, k_w = shape[0], shape[1], shape[2], shape[3]
    new_out_c = channel
    new_in_c = i_c
    if num_layer == 2:
        new_out_c = channel
        new_in_c = channel
    mean = np.mean(weights)
    std = np.std(weights)
    size_needed = new_out_c * new_in_c * k_h * k_w

    if big_to_small:
        new_weights = weights[:new_out_c, :new_in_c, :, :]
        return new_weights.astype(np.float32)

    if generate_mode == 'uniform':
        min = np.min(weights)
        max = np.max(weights)
        n_weights = np.random.uniform(min, max, size_needed)
        new_weights = np.reshape(n_weights, (new_out_c, new_in_c, k_h, k_w)).astype(np.float32)
    if generate_mode == 'random':
        n_weights = np.random.normal(mean, std, size_needed)
        new_weights = np.reshape(n_weights, (new_out_c, new_in_c, k_h, k_w))
    if generate_mode == 'avg':
        # needed to add bias
        avg = np.average(weights)
        range_bias = avg / 10
        n_weights = np.random.uniform(-1 * range_bias, range_bias, size_needed)
        shaped_weights = np.reshape(n_weights, (new_out_c, new_in_c, k_h, k_w))
        new_weights = shaped_weights + avg

    if insert_mode == 'stack':
        new_weights[:o_c, :i_c, :, :] = weights
    if insert_mode == 'shuffle':
        new_weights[:o_c, :i_c, :, :] = weights
        np.random.shuffle(new_weights)

    # if insert_mode == 'iNsert':
    #     if 1 == 0:
    #         skip = math.floor(out_c / new_c)
    #     temp = np.zeros((channel, i_c, k_h, k_w))
    #     new_i = 0
    #     old_i = 0
    #     i = 0
    #     while i < channel:
    #         for s in range(skip):
    #             if old_i < o_c:
    #                 temp[i, :, :, :] = weights[old_i]
    #                 i = i + 1
    #                 old_i = old_i + 1
    #         if new_i < new_c:
    #             temp[i, :, :, :] = shaped_weights[new_i, :, :, :]
    #             new_i = new_i + 1
    #             i = i + 1
    #     weights = temp.astype(np.float32)

    if insert_mode == 'insert':
        num_insert = new_out_c - o_c
        old_i = 0
        i = 0
        skip = 1 + 1
        while old_i < num_insert:
            new_weights[i, :i_c, :, :] = weights[old_i]
            i = i + skip
            old_i = old_i + 1
        new_weights[i:, :i_c, :, :] = weights[old_i:, :, :, :]
    return new_weights.astype(np.float32)


def test(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target)  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)
    format_str = '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), acc
    )
    print(format_str)
    return test_loss.item(), acc


def data_loader():
    batch_size = 16
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
    return testloader


def loss_mean_std(txt_loc, save_to_location):
    with open(txt_loc, 'rt') as f:
        data = f.readlines()
    for line in data:
        if line.__contains__('Batch_100'):
            log = open(save_to_location, 'a')
            log.write(line)
            log.close()


"""
if __name__ == '__main__':
    parent_dir = os.path.dirname(os.getcwd())
    model_dir = os.path.join(parent_dir, 'results/exp1_res')
    path_to_model8 = os.path.join(model_dir, 'MNIST_CNN_C8_K3_E_40_20200109-131023/model_epoch_10.ckpt')
    path_to_model12 = os.path.join(model_dir, 'MNIST_CNN_C12_K3_20200107-132035/model_epoch_10.ckpt')
    model_8 = torch.load(path_to_model8)
    model_12 = torch.load(path_to_model12)
    model8_weights = model_8.conv1.weight.data.numpy()
    new_model12_weights = generate_weights(model8_weights, 12)

    # change model_12's weights
    with torch.no_grad():
        model_12.conv1.weight.data = torch.from_numpy(new_model12_weights)

    # criterion
    criterion = nn.CrossEntropyLoss()
    test_loader = data_loader()
    loss, acc = test(model_12, test_loader, criterion)
    print('loss: ', loss, '\nacc: ', acc)
"""

if __name__ == '__main__':
    # load model
    epoch = 20
    parent_dir = os.path.dirname(os.getcwd())
    model_dir = os.path.join(parent_dir, 'results/exp2_results')
    res_dir = os.path.join(parent_dir, 'results/')
    res_dir = res_dir + 'loss_mean_std'
    # create dir for log
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    path_locations = []

    # exp1_result
    # path_to_8model_location = 'MNIST_CNN_C8_K3_20200103-165154'
    # path_to_12model_location = 'MNIST_CNN_C12_K3_20200107-132035'
    # path_to_16model_location = 'MNIST_CNN_C16_K3_20200107-135625'
    # path_to_20model_location = 'MNIST_CNN_C20_K3_20200107-144313'
    # path_to_24model_location = 'MNIST_CNN_C24_K3_20200103-141302'
    # path_to_48model_location = 'MNIST_CNN_C48_K3_20200103-171802'

    # exp2_results
    path_to_8model_location = 'MNIST_CNN_C8_K3_B100_20200130-193734'
    path_to_12model_location = 'MNIST_CNN_C12_K3_B100_20200130-210115'
    path_to_16model_location = 'MNIST_CNN_C16_K3_B100_20200130-222723'
    path_to_20model_location = 'MNIST_CNN_C20_K3_B100_20200130-232120'
    path_to_24model_location = 'MNIST_CNN_C24_K3_B100_20200131-152555'
    path_to_standard_location = 'MNIST_CNN_standard_K3_20200130-183426'
    # path_to_48model_location = 'MNIST_CNN_C48_K3_20200103-171802'

    path_locations.append(path_to_8model_location)
    path_locations.append(path_to_12model_location)
    path_locations.append(path_to_16model_location)
    path_locations.append(path_to_20model_location)
    path_locations.append(path_to_24model_location)
    path_locations.append(path_to_standard_location)

    # path_to_model = os.path.join(model_dir, path_to_model_location)
    log_dir = os.path.join(res_dir, 'mean_std.txt')

    log = open(log_dir, 'a')

    for path in path_locations:
        path_to_log = os.path.join(model_dir, path)
        path_to_log = os.path.join(path_to_log, 'log.txt')
        loss_mean_std(path_to_log, log_dir)

    # for path in path_locations:
    #     path_to_models_location = path
    #     path_to_models = [(os.path.join(model_dir, path_to_models_location) + '/model_epoch_{}.ckpt'.format(i)) for i in
    #                       range(epoch)]
    #     ratio_in_layers = ratio_layer_large_than_mean(path_to_models, log_dir, 3)
    #     log.write('density_{} = {}\n'.format(path[:13], ratio_in_layers))
    #     print('density_{} = {}\n'.format(path[:13], ratio_in_layers))

    # print 1 layer for each epoch
    #    random_from_models(path_to_models, log_dir, 3)

    # print the ratio
    # log.close()
"""
# print one model but all layers

# model = torch.load(path_to_model)
#
# # write model name to txt
# log = open(log_dir, 'a')
# log.write('==========' + path_to_model_location + '==========' + '\n')
# log.close()
#
# gaussian(model, log_dir)
"""

"""
    # all modules in model
    modules = [module for module in model.modules()]
    # modules = [(key, module) for (key, module) in model.modules()]

    print(isinstance(modules[1], nn.Conv2d))
    print(isinstance(modules[6], nn.Linear))
    # settings
    t_channel = 8
    t_kernel = 3
    kwargs = {'num_classes': 6,
              'num_channel': 8,
              'out1': t_channel,
              'out2': t_channel,
              'k_size': (1, 3),  # 3->576, 7->288
              'c_stride': (1, 1),
              'p_stride': (1, 2),
              'fc1': 256,
              'fc2': 300}

    # x axis
    x = []
    i = 0.0
    while i < 180.0:
        i = float('%.1f' % i)
        x.append(i)
        i = i + 0.1

    epoch_array = [0, 10, 19]
    colors = ['r', 'g', 'b']
    markers = ['o', 'x', '+']
    in_channels = 24
    plot_conv_weights(epochs=epoch_array, in_channels=in_channels)
"""
