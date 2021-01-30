import torch
import torch.nn as nn
import time
import numpy as np
import os
import datetime
from torch.optim import Adam
from numpy import dot
from numpy.linalg import norm
import json
import sys
import math
import copy
# from memory_profiler import profile

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ours
from grow.gate_hook import register_hook_delete, remove_hook
from ri_control.ri_grow import RIGrow
# from ri_control.ri_grow_new import RIGrow
from utils import Timer, AverageMeter, write_log, accuracy, weights_init
from ri_control.recorder import Recorder
from model.CNN import LeNet5_GROW_P, LeNet5_GROW_BN, LeNet5
from model.VGG import VGG, VGG_BN
from model.mobileNet import NetS
from grow.analysis_recorder import AnalysisRecorder
from ri_control.ri_gate import RIGate
from optmizer.ours_adam import AdamW
from grow.weight_decay import weight_decay
from grow.growth_utils import find_modules_short, model_size, find_modules, set_mode
from grow.neuron_grow import device
from ri_control.ri_adaption import grow
from ri_control.ri_adaption_mbnet import grow as mb_grow
from model.summary import summary
from data_loader import generate_data_loader
from ri_control.regression import plot_regression

#################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# torch.manual_seed(1535)
# np.random.seed(1535)
#################################


def dir_path(dir_name):
    parent_dir = os.path.dirname(os.getcwd())
    if parent_dir[-3:] == 'src':
        parent_dir = os.path.dirname(parent_dir)
    res_dir = os.path.join(parent_dir, 'results')
    res_dir = os.path.join(res_dir, (dir_name + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    path_to_model = os.path.join(res_dir, 'model_{str}.ckpt')
    path_to_log = os.path.join(res_dir, '{}_log.txt'.format(dir_name))
    path_to_json = os.path.join(res_dir, '{}_log.json'.format(dir_name))
    path_to_json1 = os.path.join(res_dir, '{}_analysis_recode.json'.format(dir_name))
    path_to_gate_dic = os.path.join(res_dir, '{}_gate.json'.format(dir_name))
    path_to_timer_dic = os.path.join(res_dir, '{}_time.json'.format(dir_name))

    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    dic = {
        'res_dir': res_dir,
        'path_to_log': path_to_log,
        'path_to_model': path_to_model,
        'path_to_json': path_to_json,
        'path_to_recorder': path_to_json1,
        'path_to_gate': path_to_gate_dic,
        'path_to_timer': path_to_timer_dic,
    }
    return dic


def gen_gradient(model, criterion, inputs, target):
    feature_out, output = model(inputs)
    loss = criterion(output, target)
    loss.backward()
    return feature_out


def put_to_ndarray(small_nd, large_nd):
    if small_nd.ndim == 4:
        s_out, s_in, k_h, k_w = small_nd.shape
        large_nd[:s_out, :s_in, :k_h, :k_w] = small_nd
    elif small_nd.ndim == 2:
        s_out, s_in = small_nd.shape
        large_nd[:s_out, :s_in] = small_nd
    elif small_nd.ndim == 1:
        s_out = small_nd.shape[0]
        large_nd[:s_out] = small_nd
    return large_nd


# def obtain_ff(dic, width=128):
#     if 'padding' in dic.keys():
#         after_first = (width + dic['padding'][1] * 2 - dic['kernel_size'] + 1) // 2
#         ff = (after_first + dic['padding'][1] * 2 - dic['kernel_size'] + 1) // 2
#     else:
#         after_first = (width - dic['kernel_size'] + 1) // 2
#         ff = (after_first - dic['kernel_size'] + 1) // 2
#     return ff

def obtain_ff(dic, width=128):
    strides = [2, 1, 2, 2]
    for i in range(4):
        width = (width - dic['kernel_size']) // strides[i] + 1
    return width


def obtain_ff_lenet(dic, width=128):
    if 'padding' in dic.keys():
        after_first = (width + dic['padding'][1] * 2 - dic['kernel_size'] + 1) // 2
        ff = (after_first + dic['padding'][1] * 2 - dic['kernel_size'] + 1) // 2
    else:
        after_first = (width - dic['kernel_size'] + 1) // 2
        ff = (after_first - dic['kernel_size'] + 1) // 2
    return ff


def replace_optimizer(model, optimizer, avg=False, scale_list=None, is_scale=False,
                      optimizer_mode='AdamW'):
    lr = optimizer.defaults['lr']
    # optimizer = optimizer
    if optimizer_mode == 'Adam':
        new_optimizer = Adam(model.parameters(), lr=lr)
    if optimizer_mode == 'AdamW':
        new_optimizer = AdamW(model.parameters(), lr=lr)
        new_optimizer.vs = optimizer.vs
        new_optimizer.ms = optimizer.ms
        new_optimizer.gs = optimizer.gs
        new_optimizer.grads = optimizer.grads
        new_optimizer.vg = optimizer.vg

    for current_group, new_group in zip(optimizer.param_groups, new_optimizer.param_groups):
        layer_index = 0
        for current_p, new_p in zip(current_group['params'], new_group['params']):
            if is_scale and scale_list is not None and layer_index / 2 < 3:
                scale = scale_list[int(layer_index / 2)]
                scale = 1 / scale
            else:
                scale = 1

            current_state = optimizer.state[current_p]
            state = new_optimizer.state[new_p]

            # State initialization
            state['step'] = current_state['step']

            # Exponential moving average of gradient values
            state['exp_avg'] = torch.zeros_like(new_p.data, memory_format=torch.preserve_format)
            new_exp_avg = state['exp_avg'].data
            current_exp_avg = current_state['exp_avg'].data * scale
            new_exp_avg = put_to_ndarray(current_exp_avg, new_exp_avg)

            # Exponential moving average of squared gradient values
            state['exp_avg_sq'] = torch.zeros_like(new_p.data, memory_format=torch.preserve_format)
            new_exp_avg_sq = state['exp_avg_sq'].data
            current_exp_avg_sq = current_state['exp_avg_sq'].data * scale
            new_exp_avg_sq = put_to_ndarray(current_exp_avg_sq, new_exp_avg_sq)

            if avg:
                mean_exp_avg = torch.mean(current_exp_avg)
                mean_exp_avg_sq = torch.mean(current_exp_avg_sq)
                new_exp_avg[new_exp_avg == 0.0] = mean_exp_avg
                new_exp_avg_sq[new_exp_avg_sq == 0.0] = mean_exp_avg_sq

            # state['exp_avg'].data = torch.from_numpy(new_exp_avg.astype(np.float32))
            # state['exp_avg'].data = state['exp_avg'].data.to(device)
            # state['exp_avg_sq'].data = torch.from_numpy(new_exp_avg_sq.astype(np.float32))
            # state['exp_avg_sq'].data = state['exp_avg_sq'].data.to(device)
            state['exp_avg'].data = state['exp_avg']
            state['exp_avg_sq'].data = state['exp_avg_sq']

            layer_index += 1

    return new_optimizer


def window_cos(score, loss, window_size, length):
    cosine = {}
    for key in score.keys():
        cosine[key] = []
        for start, end in generator(window_size, length):
            s = np.asarray(score[key][start:end])
            l = np.asarray(loss[start:end])
            cos_sim = dot(s, l) / (norm(s) * norm(l))
            cosine[key].append(cos_sim)
        if len(cosine[key]) > (length // window_size):
            cosine[key] = cosine[key][:(length // window_size)]
    return cosine


def generator(window=50, length=600):
    for i in range(0, length, window):
        yield i, i + window


def grab_input_weight_shape(model, input_shape=(1, 28, 28)):
    ins = []
    outs = []
    s = summary(model, input_shape)
    layers = find_modules(model)
    for key in s.keys():
        if key.startswith('Conv2d') or key.startswith('Linear'):
            ins.append(s[key]['input_shape'][1:])

    for module in layers:
        outs.append(module.weight.data.shape)

    return ins, outs


def convert_to_status(growth_list):
    layer_status = []
    for ele in growth_list:
        if ele[1] == 0:
            layer_status.append(False)
        else:
            layer_status.append(True)

    return layer_status


class RISearch:
    def __init__(self, model, model_name, optimizer, criterion, train_loader, test_loader, path, analysis_recorder):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.path = path
        # self.ri_gate = ri_gate
        # self.ri_recorder = ri_recorder
        self.analysis_recorder = analysis_recorder
        self.gate_active = False
        self.batch_loss = []
        self.old_size = []
        self.new_size = []
        self.num_essence = 3
        self.weight_json = {}
        self.search_models = []
        self.model_name = model_name
        self.best_model_size = []

    def save_weight(self, index_layer, epoch):
        layers = find_modules_short(self.model)
        target_layer = layers[index_layer]
        weight = target_layer.weight.data.cpu().numpy()
        flatten_weight = weight.reshape(-1).tolist()
        flatten_weight = np.abs(flatten_weight)
        self.weight_json[epoch] = flatten_weight

    def dump_weight_json(self):
        target_path = os.path.join(self.path["res_dir"], 'weight.json')
        with open(target_path, 'w') as f:
            json.dump(self.weight_json, f, indent=4)

    def write_model_size(self):
        for size in self.best_model_size:
            write_log('{}\n'.format(size), self.path['path_to_log'])

    def _train(self, epoch, model):

        # create multi average meters
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        start = time.time()
        for i_batch, (inputs, target) in enumerate(self.train_loader):

            # measure data loading time
            data_time.update(time.time() - start)

            # put data into available devices
            inputs, target = inputs.to(device), target.to(device)

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # gradient and do SGD step
            # model.train()
            model.train()

            if self.model_name == 'LeNet':
                features_output, output = model(inputs)
            else:
                output = model(inputs)

            loss = self.criterion(output, target)
            loss.backward()

            # optimizer step
            self.optimizer.step()

            # print summary each batch
            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - start)
            start = time.time()

            # record train loss
            # self.analysis_recorder.record_train_loss(loss.item())

            if (i_batch + 1) % 100 == 0:
                output_str = ('Epoch: [{0}][{1}/{2}]\t'
                              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                              'Prec@1 {top1.val:3.3f} ({top1.avg:3.3f})\t'
                              'Prec@5 {top5.val:3.3f} ({top5.avg:3.3f})'.format(
                    epoch, (i_batch + 1), len(self.train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5))
                print(output_str)
                write_log(output_str + '\n', self.path['path_to_log'])

    def _test(self, model):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(device), target.to(device)

                if self.model_name == 'LeNet':
                    _, output = model(data)
                else:
                    output = model(data)

                test_loss += criterion(output, target)  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(self.test_loader.sampler)
        test_loss *= self.test_loader.batch_size
        acc = 100. * correct / len(self.test_loader.sampler)
        format_str = 'Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(self.test_loader.sampler), acc)
        print(format_str)
        write_log(format_str, self.path['path_to_log'])
        return acc

    def gen_growth_list(self, step=0.2, max=0.61):
        g_list = []
        size = model_size(find_modules_short(self.model))

        if self.model_name == "LeNet":
            for rate in np.arange(0.0, max, step):
                grow_rate = 1 + rate
                growth_list = []
                for i in range(len(full_size)):
                    new_size = math.ceil(size[i] * grow_rate)
                    if new_size > full_size[i]:
                        new_size = full_size[i]
                    grow_size = new_size - size[i]
                    growth_list.append((i, grow_size))
                g_list.append(growth_list)

        elif self.model_name == "MobileNet":
            g_index = [0, 2, 4, 6]
            for rate in np.arange(0.0, max, step):
                grow_rate = 1 + rate
                growth_list = []
                for i in range(len(full_size)):
                    new_size = math.ceil(size[i] * grow_rate)
                    if new_size > full_size[i]:
                        new_size = full_size[i]
                    grow_size = new_size - size[i]
                    growth_list.append((g_index[i], grow_size))
                g_list.append(growth_list)

        return g_list

    def deep_copy_model(self):
        # TODO add mobiel net support

        size = model_size(find_modules_short(self.model))

        if self.model_name == 'LeNet':
            if dataset == 'MNIST':
                for s, key in zip(size, para_model.keys()):
                    para_model[key] = s
                model = LeNet5_GROW_P(**para_model)
            else:
                keys = ['out1_channel', 'out2_channel', 'fc']

                for s, key in zip(size, keys):
                    kwargs[key] = s
                model = LeNet5(**kwargs)

        elif self.model_name == 'MobileNet':
            keys = ['out1_channel', 'out2_channel', 'out3_channel', 'out4_channel']
            for g, key in enumerate(keys):
                kwargs[key] = size[g]
            model = NetS(**kwargs)

        model.load_state_dict(self.model.state_dict())
        model = model.to(device)
        return model

    def search_grow(self, epoch):
        print('Searching structure...')
        self.search_models = []
        g_list = self.gen_growth_list()

        info_dic = {}

        for i, growth_list in enumerate(g_list):
            # deep copy an og model
            original_model = self.deep_copy_model()

            new_model = self.grow_model(original_model, growth_list)

            optimizer_new = copy.deepcopy(self.optimizer)

            # trian new model
            self._train(epoch=epoch, model=new_model)

            # test new model
            acc = self._test(new_model)

            # add to info dic
            info_dic[i] = (new_model, acc, optimizer_new)
            if new_model == full_size:
                break

        # comparison between new models
        best_key = None
        for key in info_dic.keys():
            if best_key is None:
                best_key = key
            else:
                if info_dic[best_key][1] < info_dic[key][1]:
                    best_key = key

        # assign best model
        self.model = info_dic[best_key][0]
        self.optimizer = info_dic[best_key][2]

        best_size = model_size(find_modules_short(self.model))

        self.best_model_size.append(best_size)

        # print info
        print('Best model is {}, acc is {}, growth is {}'.format(best_size, info_dic[best_key][1], g_list[best_key]))
        self.analysis_recorder.record_test_acc(info_dic[best_key][1])

    def step(self, epoch):

        epoch_timer.start()
        if epoch > 1:
            self.search_grow(epoch)

        else:
            self._train(epoch, self.model)
            acc = self._test(self.model)
            self.analysis_recorder.record_test_acc(acc)

        epoch_timer.stop()

    def grow_model(self, model, growth_list, inputs=None, target=None):
        # ri_grow return growh_list
        epoch_timer.pause()

        old_size = model_size(find_modules_short(model))
        self.old_size = old_size
        print('Model old size: {}'.format(old_size))
        write_log('Model old size: {}\n'.format(old_size), self.path['path_to_log'])

        print('Before growth: ', end='')
        self._test(model)

        grow_timer.start()

        if self.model_name == 'LeNet':
            model = grow(model, growth_list, mode='random', ff=ff, inputs=inputs, target=target,
                         test_loader=self.test_loader, criterion=self.criterion)
        else:
            model = mb_grow(model, growth_list, mode='random', ff=ff, inputs=inputs, target=target,
                            test_loader=self.test_loader, criterion=self.criterion)

        grow_timer.stop()

        print('After growth: ', end='')
        self._test(model)

        model = model.to(device)

        new_size = model_size(find_modules_short(model))
        self.new_size = new_size
        print('Model new size: {}'.format(new_size))
        write_log('Model new size: {}\n'.format(new_size), self.path['path_to_log'])

        self.optimizer = replace_optimizer(model, self.optimizer)

        epoch_timer.resume()

        return model


if __name__ == '__main__':
    dataset = 'FinDroid'  # CIFAR10,MNIST,Har ,EMG
    model_name = 'MobileNet'  # MobileNet

    # create path
    path_name = 'Search_{}_{}_'.format(dataset, model_name)

    path = dir_path(path_name)

    # training
    # initialize timers
    grow_timer = Timer(name='grow')
    epoch_timer = Timer(name='epoch')

    if model_name == 'LeNet':
        if dataset == "MNIST":
            param = {
                'lr': 0.0005,
                'epoch': 10,
                'batch_size': 100,
                'is_bn': False,
            }

            # para_model = {'out1': 20, 'out2': 50, 'fc1': 500}
            # para_model = {'out1': 20, 'out2': 50, 'fc1': 500}
            # para_model = {'out1': 10, 'out2': 25, 'fc1': 250}
            para_model = {'out1': 2, 'out2': 5, 'fc1': 10}
            ff = 16
            # set torch and numpy random seed
            torch.manual_seed(1535)

            # BN seed
            # torch.manual_seed(7379)

            # create model
            # model = LeNet5_P(**para_model)
            if param['is_bn']:
                model = LeNet5_GROW_BN(**para_model)
            else:
                model = LeNet5_GROW_P(**para_model)
            model = model.to(device)
            input_shape = (1, 28, 28)
            full_size = [16, 42, 500]

        elif dataset == "Har":
            param = {
                'lr': 0.0005,
                'epoch': 10,
                'batch_size': 64,
                'is_bn': False,
            }
            input_shape = (9, 1, 128)
            width = 128
            kwargs = {
                'in_channel': 9,
                'out1_channel': 2,
                'out2_channel': 5,
                'fc': 10,
                'out_classes': 6,
                'kernel_size': 14,
                'flatten_factor': 5
            }
            path_to_subject = ""

            ff = obtain_ff_lenet(kwargs, width)
            kwargs['flatten_factor'] = ff
            # uniform
            # torch.manual_seed(7518)

            # normal
            # torch.manual_seed(9162)

            model = LeNet5(**kwargs)
            # model = NetS()
            # model.apply(weights_init)
            model = model.to(device)
            full_size = [18, 45, 500]

        elif dataset == "EMG":
            param = {
                'lr': 0.0005,
                'epoch': 30,
                'batch_size': 64,
                'is_bn': False,
            }
            input_shape = (8, 1, 100)
            width = 100
            kwargs = {
                'in_channel': 8,
                'out1_channel': 2,
                'out2_channel': 5,
                'fc': 20,
                'out_classes': 6,
                'kernel_size': 14,
                'flatten_factor': 5
            }

            ff = obtain_ff_lenet(kwargs, width)
            kwargs['flatten_factor'] = ff
            # uniform
            # torch.manual_seed(7518)

            # normal
            # torch.manual_seed(9162)

            model = LeNet5(**kwargs)
            # model = NetS()
            # model.apply(weights_init)
            model = model.to(device)
            full_size = [16, 42, 500]

        elif dataset == 'myHealth':

            width = 100
            input_shape = (23, 1, 100)
            kwargs = {
                'in_channel': 23,
                'out1_channel': 2,
                'out2_channel': 5,
                'fc': 10,
                'out_classes': 11,
                'kernel_size': 14,
                'flatten_factor': 5,
            }
            ff = obtain_ff_lenet(kwargs, width)
            kwargs['flatten_factor'] = ff
            # uniform
            # torch.manual_seed(7518)

            # normal
            # torch.manual_seed(9162)

            model = LeNet5(**kwargs)
            # model = NetS()
            # model.apply(weights_init)
            model = model.to(device)
            full_size = [18, 40, 500]

        elif dataset == 'FinDroid':
            param = {
                'lr': 0.0005,
                'epoch': 15,
                'batch_size': 64,
                'is_bn': False,
            }
            width = 150
            input_shape = (6, 1, 150)
            kwargs = {
                'in_channel': 6,
                'out1_channel': 2,
                'out2_channel': 5,
                'fc': 10,
                'out_classes': 6,
                'kernel_size': 14,
                'flatten_factor': 5,
            }
            ff = obtain_ff_lenet(kwargs, width)
            kwargs['flatten_factor'] = ff
            # uniform
            # torch.manual_seed(7518)

            # normal
            # torch.manual_seed(9162)

            model = LeNet5(**kwargs)
            # model = NetS()
            # model.apply(weights_init)
            model = model.to(device)
            full_size = [15, 46, 500]

        elif dataset == "CIFAR10":
            param = {
                'lr': 0.0005,
                'epoch': 20,
                'batch_size': 100,
                'is_bn': False,
            }
            ff = 1
            input_shape = (3, 34, 34)
            # create models
            if param['is_bn']:
                model = VGG_BN('seed')
            else:
                model = VGG('seed')
            model = model.to(device)
            full_size = [64, 128, 256, 256, 512, 512, 512, 512]

    elif model_name == "MobileNet":
        set_mode(model_name)
        if dataset == "Har":
            param = {
                'lr': 0.0005,
                'epoch': 15,
                'batch_size': 64,
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

            ff = obtain_ff(kwargs, width)
            kwargs['avg_factor'] = ff
            # torch.manual_seed(1725)
            model = NetS(**kwargs)
            # model.apply(weights_init)
            model = model.to(device)

            full_size = [26, 55, 128, 256]

        elif dataset == "EMG":
            param = {
                'lr': 0.0005,
                'epoch': 30,
                'batch_size': 64,
                'is_bn': False,
            }
            input_shape = (8, 1, 100)
            width = 100
            kwargs = {
                'in_channel': 8,
                'out1_channel': 8,
                'out2_channel': 10,
                'out3_channel': 12,
                'out4_channel': 25,
                'out_classes': 6,
                'kernel_size': 12,
                'avg_factor': 2
            }

            ff = obtain_ff(kwargs, width)
            kwargs['avg_factor'] = ff
            # uniform
            # torch.manual_seed(7518)

            # normal
            # torch.manual_seed(9162)

            model = NetS(**kwargs)
            # model = NetS()
            model.apply(weights_init)
            model = model.to(device)
            full_size = [32, 64, 128, 256]


        elif dataset == 'myHealth':
            param = {
                'lr': 0.0005,
                'epoch': 30,
                'batch_size': 64,
                'is_bn': False,
            }
            width = 100
            input_shape = (23, 1, 100)
            kwargs = {
                'in_channel': 23,
                'out1_channel': 3,
                'out2_channel': 6,
                'out3_channel': 12,
                'out4_channel': 25,
                'out_classes': 11,
                'kernel_size': 12,
                'avg_factor': 2
            }

            # f_kwargs = {
            #     'in_channel': 23,
            #     'out1_channel': 32,
            #     'out2_channel': 64,
            #     'out3_channel': 128,
            #     'out4_channel': 256,
            #     'out_classes': 11,
            #     'kernel_size': 12,
            #     'avg_factor': 1
            # }
            ff = obtain_ff(kwargs, width)
            kwargs['avg_factor'] = ff
            # uniform
            # torch.manual_seed(7518)

            # normal
            # torch.manual_seed(9162)

            # model = NetS(**f_kwargs)
            model = NetS(**kwargs)
            # model.apply(weights_init)
            model = model.to(device)
            full_size = [32, 64, 128, 256]

        elif dataset == 'FinDroid':
            param = {
                'lr': 0.0005,
                'epoch': 20,
                'batch_size': 64,
                'is_bn': False,
            }
            width = 150
            input_shape = (6, 1, 150)

            kwargs = {
                'in_channel': 6,
                'out1_channel': 3,
                'out2_channel': 6,
                'out3_channel': 12,
                'out4_channel': 25,
                'out_classes': 6,
                'kernel_size': 14,
                'avg_factor': 2
            }
            ff = obtain_ff(kwargs, width)
            kwargs['avg_factor'] = ff
            # uniform
            # torch.manual_seed(7518)

            # normal
            # torch.manual_seed(9162)

            model = NetS(**kwargs)
            # model.apply(weights_init)
            model = model.to(device)
            full_size = [32, 64, 128, 256]

    # generate data loader
    trainloader, testloader = generate_data_loader(batch_size=param['batch_size'], dataset=dataset, is_main=False)

    # optimizer and criterion
    criterion = nn.CrossEntropyLoss()

    # initialize optimizer
    optimizer = AdamW(model.parameters(), lr=param['lr'])

    num_layers = len(find_modules_short(model))
    # create ri components
    # ri_gate = RIGate(num_layers, len(trainloader))

    # ri_grow = RIGrow(num_layers,model_name ,path)
    # ri_grow = RIGrow(num_layers, model_name, path)
    # ri_grow = RIGrow(num_layers, model_name, path)

    # ri_recorder = Recorder(num_layers)

    analysis_recorder = AnalysisRecorder(optimizer, 'AdamW', None, path['path_to_log'])

    ri_search_controller = RISearch(model, model_name, optimizer, criterion, trainloader, testloader, path, analysis_recorder)

    for i in range(param['epoch']):
        ri_search_controller.step(i + 1)

    ri_search_controller.write_model_size()
    analysis_recorder.write_txt()
    epoch_timer.dump_json(path['path_to_timer'])
    # ri_gate.save_gate_dic_to_json(path['path_to_gate'])
