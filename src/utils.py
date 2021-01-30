import numpy as np
import torch
from torch import nn
import json
from torch.autograd import Variable
import datetime
import time
import os
from torch.nn.init import _calculate_fan_in_and_fan_out
import math
import torch.nn.init as init
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import data_loader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')


def find_layers(model):
    layers = []
    for index_str, module in model.features._modules.items():
        if isinstance(module, torch.nn.Conv2d):
            layers.append(module)

    for index_str, module in model.classifier._modules.items():
        if isinstance(module, torch.nn.Linear):
            layers.append(module)
    return layers


def dir_path(dir_name):
    parent_dir = os.path.dirname(os.getcwd())
    if parent_dir[-3:] == 'src':
        parent_dir = os.path.dirname(parent_dir)
    res_dir = os.path.join(parent_dir, 'results')
    res_dir = os.path.join(res_dir, (dir_name + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    path_to_model = os.path.join(res_dir, 'model_{str}.ckpt')
    path_to_log = os.path.join(res_dir, '{}_log.txt'.format(dir_name))
    path_to_layers = {
        0: os.path.join(res_dir, 'layer0_log.txt'),
        1: os.path.join(res_dir, 'layer1_log.txt'),
        2: os.path.join(res_dir, 'layer2_log.txt'),
        3: os.path.join(res_dir, 'layer3_log.txt'),
    }
    path_to_json = os.path.join(res_dir, '{}_json_log.json'.format(dir_name))
    path_to_json1 = os.path.join(res_dir, '{}_json_in_out.json'.format(dir_name))
    path_to_json_dic = os.path.join(res_dir, '{}_json_dic.json'.format(dir_name))
    path_to_gate_dic = os.path.join(res_dir, '{}_Gate.json'.format(dir_name))
    path_to_lr_dic = os.path.join(res_dir, '{}_LR.json'.format(dir_name))
    path_to_timer_dic = os.path.join(res_dir, '{}_Timer.json'.format(dir_name))
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    dic = {
        'res_dir': res_dir,
        'path_to_log': path_to_log,
        'path_to_model': path_to_model,
        'path_to_test': path_to_layers,
        'path_to_json': path_to_json,
        'path_to_json1': path_to_json1,
        'path_to_json_dic': path_to_json_dic,
        'path_to_gate': path_to_gate_dic,
        'path_to_lr': path_to_lr_dic,
        'path_to_timer': path_to_timer_dic,
    }

    return dic


def calculate_l1(module):
    grad_tensor = module.weight.grad.data.cpu().numpy()
    channel_grads = []
    for i in range(len(grad_tensor)):
        channel_grad = grad_tensor[i]
        result = np.sum(np.abs(channel_grad))
        channel_grads.append(result)
    layer_grads = np.sum(np.abs(grad_tensor))
    return channel_grads, np.around(layer_grads, 5)


def check_dic_key(dic, key):
    if key not in dic:
        dic[key] = {}


def calculate_l2(module):
    grad_tensor = module.weight.grad.data.cpu().numpy()
    channel_grads = []
    for i in range(len(grad_tensor)):
        channel_grad = grad_tensor[i]
        result = np.sqrt(np.sum(channel_grad ** 2))
        channel_grads.append(result)
    layer_grads = np.sqrt(np.sum(grad_tensor ** 2))
    return channel_grads, np.around(layer_grads, 5)


def calculate_norm(module, mode='out', order=1):
    dic_axis = {'out': {2: 1, 4: (1, 2, 3), 1: None}, 'in': {2: 0, 4: (0, 2, 3), 1: None},
                'layer': {2: None, 4: None, 1: None}}
    grad_tensor = module.weight.grad.data.cpu()
    d = dic_axis[mode][grad_tensor.ndim]
    result = grad_tensor.norm(p=order, dim=d).tolist()
    return result


def calculate_s_score(module, index_list=None):
    grad_tensor = module.weight.grad.data.cpu().numpy()
    weights_tensor = module.weight.data.cpu().numpy()
    channel_score = []
    for i in range(len(grad_tensor)):
        channel_grad = grad_tensor[i]
        channel_weight = weights_tensor[i]
        result = np.sum(np.abs(channel_grad * channel_weight))
        if index_list is None:
            channel_score.append(result)
        else:
            if i in index_list:
                channel_score.append(result)
    layer_score = np.sum(np.abs(grad_tensor * weights_tensor))
    return channel_score, np.around(layer_score, 5)


def calculate_s_score_new(module, mode='out', index_list=None):
    grad_tensor = module.weight.grad.data.cpu().numpy()
    weights_tensor = module.weight.data.cpu().numpy()
    channel_score = []

    if mode == 'layer':
        layer_score = np.sum(np.abs(grad_tensor * weights_tensor))
        return layer_score
    elif mode == 'out':
        axis = 0
    elif mode == 'in':
        axis = 1

    if index_list is None:
        iter_list = range(grad_tensor.shape[axis])
    else:
        iter_list = index_list

    for index in iter_list:
        channel_grad = get_ndarray(grad_tensor, axis, index)
        channel_weight = get_ndarray(weights_tensor, axis, index)
        result = np.sum(np.abs(channel_grad * channel_weight))
        channel_score.append(result)
    return channel_score


def calculate_s_score_new(module, mode='out', index_list=None):
    grad_tensor = module.weight.grad.data.cpu().numpy()
    weights_tensor = module.weight.data.cpu().numpy()
    channel_score = []

    if mode == 'layer':
        layer_score = np.sum(np.abs(grad_tensor * weights_tensor))
        return layer_score
    elif mode == 'out':
        axis = 0
    elif mode == 'in':
        axis = 1

    if index_list is None:
        iter_list = range(grad_tensor.shape[axis])
    else:
        iter_list = index_list

    for index in iter_list:
        channel_grad = get_ndarray(grad_tensor, axis, index)
        channel_weight = get_ndarray(weights_tensor, axis, index)
        result = np.sum(np.abs(channel_grad * channel_weight))
        channel_score.append(result)
    return channel_score


def target_scale_calculator(tensor, mode='Kaiming'):
    if mode == 'Kaiming':
        fan_in, _ = _calculate_fan_in_and_fan_out(tensor)
        scale = 1 / math.sqrt(fan_in)
    elif mode == 'Xavier':
        fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
        scale = math.sqrt(6) / math.sqrt(fan_in + fan_out)
    return scale


def target_std_scale_calculator(tensor, mode='Kaiming', ratio=0.5):
    if mode == 'Kaiming':
        fan_in, _ = _calculate_fan_in_and_fan_out(tensor)
        scale = 1 / math.sqrt(fan_in)
        std = scale * (1 / math.sqrt(3))
        if tensor.dim() == 2:
            std = std * ratio
        # std = std * ratio
    elif mode == 'Xavier':
        fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
        scale = math.sqrt(6) / math.sqrt(fan_in + fan_out)
        std = scale * (1 / math.sqrt(3))
    return std


def fill_zeros_to_diff(og, new):
    new_shape = new.shape
    og_zero = np.zeros_like(og)
    og_zero[:new_shape[0], :new_shape[1]] = new
    return og_zero


def calculate_vg_score_new(module, vg, mode='out', index_list=None):
    grad_tensor = vg.numpy()
    weights_tensor = module.weight.data.cpu().numpy()
    channel_score = []
    if grad_tensor.shape != weights_tensor.shape:
        return channel_score

    if mode == 'layer':
        layer_score = np.sum(np.abs(grad_tensor * weights_tensor))
        return layer_score
    elif mode == 'out':
        axis = 0
    elif mode == 'in':
        axis = 1

    if index_list is None:
        iter_list = range(grad_tensor.shape[axis])
    else:
        iter_list = index_list

    for index in iter_list:
        channel_grad = get_ndarray(grad_tensor, axis, index)
        channel_weight = get_ndarray(weights_tensor, axis, index)
        result = np.sum(np.abs(channel_grad * channel_weight))
        channel_score.append(result)

    return channel_score


def calculate_cosine_similarity(module, exist_index=None, rank=None, mode='avg'):
    # grad_tensor = module.weight.grad.data.cpu().numpy()
    cos_list = []
    weights_tensor = module.weight
    weights_exist = weights_tensor[:exist_index]
    weights_new = weights_tensor[exist_index:]
    if mode == 'avg':
        weights_point = torch.mean(weights_exist, dim=0)
    elif mode == 'rank':
        weights_point = weights_exist[rank[0]]

    cos = nn.CosineSimilarity(dim=0)
    weights_point = torch.flatten(weights_point)

    for t in weights_new:
        t_new = torch.flatten(t)
        cs = cos(weights_point, t_new)
        cos_list.append(cs.tolist())

    return cos_list


def get_ndarray(ndarray, axis, index):
    if isinstance(index, tuple):
        s_index = index[0]
        e_index = index[1]
    else:
        s_index = index
        e_index = index + 1
    if axis == 0:
        return ndarray[s_index:e_index]
    if axis == 1:
        return ndarray[:, s_index:e_index]
    if axis == 2:
        return ndarray[:, :, s_index:e_index]
    if axis == 3:
        return ndarray[:, :, :, s_index:e_index]


def calculate_weight_old(module, mode='out', index_list=None):
    weight = module.weight.data.cpu().numpy()
    mean_list = []
    std_list = []
    if mode == 'layer':
        return np.mean(weight), np.std(weight)
    elif mode == 'out':
        axis = 0
    elif mode == 'in':
        axis = 1

    if index_list is None:
        iter_list = range(weight.shape[axis])
    else:
        iter_list = index_list

    for index in iter_list:
        weight_needed = get_ndarray(weight, axis, index)
        m = np.around(np.mean(weight_needed), 5)
        s = np.around(np.std(weight_needed), 5)
        mean_list.append(m)
        std_list.append(s)
    return mean_list, std_list


def calculate_weight(module, mode='out'):
    dic_axis = {'out': {2: 1, 4: (1, 2, 3), 1: None}, 'in': {2: 0, 4: (0, 2, 3), 1: None},
                'layer': {2: None, 4: None, 1: None}}

    weight = module.weight.data

    if mode == 'layer':
        return torch.norm(weight, p=1).tolist(), torch.std(weight).tolist()

    d = dic_axis[mode][weight.ndim]
    l1 = weight.norm(p=1, dim=d).tolist()
    std = weight.std(dim=d).tolist()

    return l1, std


def calculate_sparsity(module, mode='layer', threshold_percentage=0.05):
    sparsity_list = []
    weight = module.weight.data.cpu()
    weight_abs = torch.abs(weight)
    max_weight = torch.max(weight_abs)
    threshold_weight = max_weight * threshold_percentage
    t_one = torch.ones_like(weight)
    t_zeros = torch.zeros_like(weight)
    t_threshold = torch.where(weight_abs < threshold_weight, t_zeros, t_one)
    if mode == 'layer':
        total_count = torch.flatten(weight).shape[0]
        t_sum = torch.sum(t_threshold)
        sparsity = t_sum / total_count * 100
        return sparsity.tolist()
    elif mode == 'out':
        axis = 0
    elif mode == 'in':
        axis = 1

    iter_list = range(weight.shape[axis])

    for index in iter_list:
        weight_needed = get_ndarray(t_threshold, axis, index)
        total_count = torch.flatten(weight_needed).shape[0]
        t_sum = torch.sum(weight_needed)
        sparsity = t_sum / total_count * 100
        sparsity_list.append(sparsity.tolist())
    return sparsity_list


def str_to_float(str):
    num_list = []
    str = str[1:-1]
    str_arry = str.split(', ')
    for num in str_arry:
        num_list.append(float(num))
    return np.asarray(num_list)


def train(train_loader, model, criterion, optimizer, to_log, print_freq=100, data_type='image', epoch=0,
          print_grad=False, two_output=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    train_loss = []
    layers_grad = []
    # switch to train mode
    model.train()
    inputs, target = None, None
    # record start time
    start = time.time()

    for i, (inputs, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - start)
        grad = []
        # prepare input and target
        if data_type == 'image':
            inputs = inputs.to(device)
            target = target.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # gradient and do SGD step
        if two_output:
            feature_out, output = model(inputs)
        else:
            output = model(inputs)
        loss = criterion(output, target)
        train_loss.append(loss.item())
        loss.backward()
        optimizer.step()

        if print_grad:
            for module in find_layers(model):
                grad_tensor = module.weight.grad.data.cpu().numpy()
                result = np.sqrt(np.sum(grad_tensor ** 2))
                grad.append(result)
            # for index_str, module in model.features._modules.items():
            #     if isinstance(module, torch.nn.Conv2d) and int(index_str) == 0:
            #         grad_tensor = module.weight.grad.data.cpu().numpy()
            #         result = np.sqrt(np.sum(grad_tensor ** 2))
            #         grad.append(result)
            #     elif isinstance(module, torch.nn.Conv2d) and int(index_str) > 0:
            #         grad_tensor = module.weight.grad.data.cpu().numpy()
            #         result = np.sqrt(np.sum(grad_tensor ** 2))
            #         grad.append(result)
            #
            # for index_str, module in model.classifier._modules.items():
            #     if isinstance(module, torch.nn.Linear) and int(index_str) == 0:
            #         grad_tensor = module.weight.grad.data.cpu().numpy()
            #         result = np.sqrt(np.sum(grad_tensor ** 2))
            #         grad.append(result)
            layers_grad.append(grad)
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - start)
        start = time.time()

        if i % print_freq == 0:
            str = ('Epoch: [{0}][{1}/{2}]\t'
                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                   'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                   'Prec@1 {top1.val:3.3f} ({top1.avg:3.3f})\t'
                   'Prec@5 {top5.val:3.3f} ({top5.avg:3.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
            print(str)
            write_log(str + '\n', to_log)
    if print_grad:
        return train_loss, layers_grad, inputs, target
    else:
        return train_loss


# custom weights initialization called on nets
# Model_10_17_50
# Conv1 Bound:
# 0.20000|0.11858|0.20000|0.12449|
# Conv2 Bound:
# 0.06325|0.03691|0.06325|0.03896|
# linear3 Bound:
# 0.06063|0.03499|0.06063|0.03890|
# linear4 Bound:
# max_value4142|0.08255|max_value4142|0.08416|
def weights_init(m, min_value=-0.1, max_value=0.1, mode='Xavier'):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
    # if classname.find('Linear') != -1:
        if mode == 'uniform':
            m.weight.data.uniform_(min_value, max_value)
            m.bias.data.uniform_(min_value, max_value)
        elif mode == 'Xavier':
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                fan_in, fan_out = init._calculate_fan_in_and_fan_out(m.weight)
                bound = math.sqrt(6) / math.sqrt(float(fan_in + fan_out))
                init.uniform_(m.bias, -bound, bound)
            # m.weight.data. (min_value, max_value)
            # m.bias.data.normal_(min_value, max_value)
        # m.weight.data.normal_(0.0, 0.05)
        # m.bias.data.normal_(0.0, 0.05)
    # elif classname.find('BatchNorm') != -1:
    #     m.weight.data.normal_(1.0, 0.02)
    #     m.bias.data.fill_(0)
    # elif classname.find('Linear') != -1:
    #     m.weight.data.uniform_(min_value, max_value)
    #     m.bias.data.normal_(min_value, max_value)
    # m.bias.data.fill_(0)


def test(model, test_loader, criterion, to_log=None, is_two_out=False):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            if is_two_out:
                _, output = model(data)
            else:
                output = model(data)
            test_loss += criterion(output, target)  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.sampler)
    test_loss *= test_loader.batch_size
    acc = 100. * correct / len(test_loader.sampler)
    format_str = 'Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.sampler), acc
    )
    print(format_str)
    if to_log is not None:
        write_log(format_str, to_log)
    return test_loss.item(), acc


def test1(model, test_loader, criterion, is_two=False):
    model.eval()
    test_loss = 0
    correct = 0
    cor_list = []
    pred_list = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            if is_two:
                _, output = model(data)
            else:
                output = model(data)
            test_loss += criterion(output, target)  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            pred_list.append(pred.numpy().tolist())
            cor_list.append(target.numpy().tolist())
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.sampler)
    test_loss *= test_loader.batch_size
    acc = 100. * correct / len(test_loader.sampler)
    format_str = 'Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.sampler), acc
    )
    print(format_str)
    return pred_list, cor_list


def validate(val_loader, model, criterion, to_log, data_type='image', print_freq=100):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # prepare a txt file for logging
    log = open(to_log, 'a')

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (inputs, target) in enumerate(val_loader):

        if data_type != 'image':
            inputs = inputs.to(device)
            target = target.to(device)

        # compute output
        output = model(inputs)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            str = ('Test: [{0}/{1}]\t'
                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                   'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                   'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1, top5=top5))
            print(str)
            log.write(str + '\n')

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
    log.write(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
    log.close()
    return top1.avg


def diff(arr):
    a = arr[1:]
    b = arr[:-1]
    return np.asarray(a) - np.asarray(b)


# timer.py
class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""


class Timer:
    timers = dict()

    def __init__(self, name=None):
        self.name = name
        self._start_time = None
        self._pause_time = None
        self.paused_time = []

        if name:
            self.timers.setdefault(name, [])

    def start(self):
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def pause(self):
        if self._start_time is None:
            raise TimerError(f"Timer is  not running. Use .start() to start it")
        if self._pause_time is not None:
            raise TimerError(f"Timer is paused. Use .resume() to resume it")
        self._pause_time = time.perf_counter()

    def resume(self):
        if self._pause_time is None:
            raise TimerError(f"Timer is not paused. Use .pause() to pause it")
        self.paused_time.append(time.perf_counter() - self._pause_time)
        self._pause_time = None

    def stop(self):
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        elapsed_time = time.perf_counter() - self._start_time
        for p_time in self.paused_time:
            elapsed_time -= p_time

        self._start_time = None
        self._pause_time = None
        self.paused_time = []
        # log time
        self.timers[self.name].append(elapsed_time)

    def dump_json(self, path):
        with open(path, 'w') as f:
            json.dump(self.timers, f, indent=4)


"""
# Train the model
def train(data_loader=None, model=None, criterion=None, optimizer=None, num_epochs=20):
    total_step = len(data_loader)
    for epoch in range(num_epochs):
        for i, (x, y) in enumerate(data_loader):
            x = x.to(device)
            y = y.to(device)

            # Forward pass
            outputs = model(x)
            loss = criterion(outputs, y)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 20 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
    return model
"""
"""
# Test the model
def test(model=None, test_loader=None):
    with torch.no_grad():
        correct = 0
        total = 0
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))




# def EMG(model_name='lstm', path_to_x=None, path_to_y=None):
#     kwargs = {'num_classes': 6, 'num_channel': 8,
#               'out1': 8, 'out2': 8, 'out3': 16, 'out4': 16, 'out5': 16, 'f1': 300, 'f2': 300}
#     EMG = data_loader.Config(path_x=path_to_x, path_y=path_to_y, input_width=150, num_classes=6)
#     data_loader = EMG.data_loader()
#     if model_name == 'lstm':
#         model = LSTM.RNN(input_size=EMG.channel, hidden_size=128, num_layers=2, num_classes=EMG.num_classes).to(device)
#     if model_name == 'mobileNet':
#         model = mobileNet.MobileNetV2(n_class=EMG.num_classes, input_size=EMG.input_width, width_mult=1.,
#                                       channel=EMG.channel)
#     if model_name == 'AlexNet':
#         model = AlexNet.AlexNet(**kwargs)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=EMG.learning_rate)
#     model = train(model=model, data_loader=data_loader, criterion=criterion, optimizer=optimizer)
#     test(model=model, test_loader=data_loader)
#     return model
"""


# Save the model checkpoint
def save_model(model=None, path_to_model=None, mode='part'):
    """Save model in two ways :  state_dic, entire"""

    if mode == 'part':
        torch.save(model.state_dict(), path_to_model)
    if mode == 'entire':
        torch.save(model, path_to_model)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


# Calculate the size of parameters.
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def write_log(str, path_to_log):
    if isinstance(path_to_log, dict):
        for key in path_to_log:
            log = open(path_to_log[key], 'a')
            log.write(str)
            log.close()

    else:
        log = open(path_to_log, 'a')
        log.write(str)
        log.close()


def prune_rate(model, verbose=True):
    """
    Print out prune rate for each layer and the whole network
    """
    total_nb_param = 0
    nb_zero_param = 0

    layer_id = 0

    for parameter in model.parameters():

        param_this_layer = 1
        for dim in parameter.data.size():
            param_this_layer *= dim
        total_nb_param += param_this_layer

        # only pruning linear and conv layers
        if len(parameter.data.size()) != 1:
            layer_id += 1
            zero_param_this_layer = \
                np.count_nonzero(parameter.cpu().data.numpy() == 0)
            nb_zero_param += zero_param_this_layer

            if verbose:
                print("Layer {} | {} layer | {:.2f}% parameters pruned" \
                    .format(
                    layer_id,
                    'Conv' if len(parameter.data.size()) == 4 \
                        else 'Linear',
                    100. * zero_param_this_layer / param_this_layer,
                ))
    pruning_perc = 100. * nb_zero_param / total_nb_param
    if verbose:
        print("Final pruning rate: {:.2f}%".format(pruning_perc))
    return pruning_perc


def arg_nonzero_min(a):
    """
    nonzero argmin of a non-negative array
    """

    if not a:
        return

    min_ix, min_v = None, None
    # find the starting value (should be nonzero)
    for i, e in enumerate(a):
        if e != 0:
            min_ix = i
            min_v = e
    if not min_ix:
        print('Warning: all zero')
        return np.inf, np.inf

    # search for the smallest nonzero
    for i, e in enumerate(a):
        if e < min_v and e != 0:
            min_v = e
            min_ix = i

    return min_v, min_ix


def generate_data_loader(batch_size, dataset='MNIST', is_main=True):
    testloader = None
    trainloader = None
    if dataset == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        if is_main:
            root_dir = '../data'
        else:
            root_dir = '../../data'
        trainset = torchvision.datasets.MNIST(root=root_dir, train=True,
                                              download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=0)
        testset = torchvision.datasets.MNIST(root=root_dir, train=False,
                                             download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=0)
    elif dataset == 'CIFAR10':
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                                download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=0)
        testset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                               download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=0)
    elif dataset == 'EMG':
        num_channels = 8
        width = 20
        Mode = '2D'
        path_to_x_train = '/Users/zber/ProgramDev/exp_pyTorch/data/EMG/W20_overlap_x_train.npy'
        path_to_y_train = '/Users/zber/ProgramDev/exp_pyTorch/data/EMG/W20_overlap_y_train.npy'
        path_to_x_test = '/Users/zber/ProgramDev/exp_pyTorch/data/EMG/W20_overlap_x_test.npy'
        path_to_y_test = '/Users/zber/ProgramDev/exp_pyTorch/data/EMG/W20_overlap_y_test.npy'

        trainset = data_loader.Data(x=path_to_x_train, y=path_to_y_train, num_channels=num_channels, width=width,
                                    Mode=Mode)
        testset = data_loader.Data(x=path_to_x_test, y=path_to_y_test, num_channels=num_channels, width=width,
                                   Mode=Mode)
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)
    elif dataset == 'Har':
        num_channels = 9
        width = 128
        Mode = '2D'
        path_to_x_train = '/Users/zber/Desktop/Desktop2/1_Data/data/Har/normal_train_X.npy'
        path_to_y_train = '/Users/zber/Desktop/Desktop2/1_Data/data/Har/normal_train_y.npy'
        path_to_x_test = '/Users/zber/Desktop/Desktop2/1_Data/data/Har/normal_test_X.npy'
        path_to_y_test = '/Users/zber/Desktop/Desktop2/1_Data/data/Har/normal_test_y.npy'

        trainset = data_loader.Data(x=path_to_x_train, y=path_to_y_train, num_channels=num_channels, width=width,
                                    Mode=Mode)
        testset = data_loader.Data(x=path_to_x_test, y=path_to_y_test, num_channels=num_channels, width=width,
                                   Mode=Mode)
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)

    return trainloader, testloader