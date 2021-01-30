import torch
import torch.nn as nn
import time
import numpy as np
import os
import datetime
from torch.optim import Adam
from numpy import dot
from numpy.linalg import norm
import sys
import math
# from memory_profiler import profile

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ours
from grow.gate_hook import register_hook_delete, remove_hook
from ri_control.ri_grow import RIGrow
from utils import Timer, AverageMeter, write_log, accuracy, weights_init, save_model
from ri_control.recorder import Recorder
from model.CNN import LeNet5_GROW_P, LeNet5_GROW_BN, LeNet5
from model.mobileNet import NetS
from grow.analysis_recorder import AnalysisRecorder
from ri_control.ri_gate import RIGate
from optmizer.ours_adam import AdamW
from grow.weight_decay import weight_decay
from grow.growth_utils import set_mode, model_size, find_modules, find_modules_short
from grow.neuron_grow import device
from ri_control.ri_adaption_mbnet import grow
from model.summary import summary
from data_loader import generate_data_loader
from ri_control.flops import set_dataset
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


def obtain_ff(dic, width=128):
    strides = [2, 1, 2, 2]
    for i in range(4):
        width = (width - dic['kernel_size']) // strides[i] + 1
    return width


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


# def generator(window=50, length=600):
#     for i in range(0, length, window):
#         yield i, i + window

def generator(window=50, length=600):
    for i in range(0, length, window):
        if i+window <= length:
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


class RIController:
    def __init__(self, model, optimizer, criterion, mode, train_loader, test_loader, path, batch_timer, ri_gate,
                 ri_recorder, ri_grow, analysis_recorder):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.mode = mode
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.path = path
        self.batch_timer = batch_timer
        self.ri_gate = ri_gate
        self.ri_recorder = ri_recorder
        self.ri_grow = ri_grow
        self.analysis_recorder = analysis_recorder
        self.gate_active = False
        self.batch_loss = []
        self.old_size = []
        self.new_size = []
        self.num_essence = 3

    def _train(self, epoch):
        # train_loss = []
        self.gate_active = False
        first_time = True

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
            self.model.train()
            output = self.model(inputs)
            loss = self.criterion(output, target)
            loss.backward()

            # ri_gate
            if self.gate_active and self.mode == 'rank_cosine':

                # gate step according to layer status
                self.ri_gate.step(self.ri_recorder.get_pre_score(), self.ri_recorder.get_score(),
                                  self.ri_recorder.get_pre_weight(), self.ri_recorder.get_weight(),
                                  self.old_size, step=i_batch)

                ri_gate.record_gate(epoch, i_batch)
                if i_batch == len(self.train_loader) - 1:
                    self.ri_gate.reset_grad_gate()
                    self.ri_gate.reset()

                # gate gradient according to layer status
                self.ri_gate.gradient_decay(self.model, self.old_size)

            # if self.gate_active and self.mode == 'rank_cosine':
            #
            #     # gate step according to layer status
            #     self.ri_gate.step(self.ri_recorder.get_pre_score(), self.ri_recorder.get_score(),
            #                       self.ri_recorder.get_pre_weight(), self.ri_recorder.get_weight(),
            #                       self.old_size, step=i_batch)
            #
            #     ri_gate.record_gate(epoch, i_batch)
            #     if i_batch == len(self.train_loader) - 1:
            #         # self.ri_gate.reset_grad_gate()
            #         self.ri_gate.reset()
            #
            # if self.new_size:
            #     # gate gradient according to layer status
            #     self.ri_gate.gradient_decay(self.model, None)

            # during the training, data will be written in the grow controller
            # record all information in

            # optimizer step
            self.optimizer.step()

            # record in and out and og value is in each batch epoch
            self.ri_recorder.record(self.model)

            # if grow trigger grow
            if is_grow:
                if epoch in [2, 4, 7, 10] and i_batch == 0:
                    # if epoch in [] and i_batch == 0:
                    self.grow_model([(0, 1), (2, 2), (4, 4), (6, 8)], inputs, target)

            # grow controller
            if grow_control:
                if 1 < epoch <= 2 and first_time:
                    # self.grow_model(growth_list=[(0, 1), (2, 2), (4, 4), (6, 8)])
                    self.grow_model(growth_list=[(0, 1), (2, 2), (4, 3), (6, 4)])
                    first_time = False

                elif 2 < epoch <= self.num_essence and first_time:
                    # self.grow_model(growth_list=[(0, 2), (2, 4), (4, 8), (6, 16)])
                    self.grow_model(growth_list=[(0, 2), (2, 4), (4, 6), (6, 8)])
                    first_time = False

                elif epoch > self.num_essence and first_time:
                    self.grow_policy()
                    first_time = False

            # record each epoch in analysis
            epoch_timer.pause()
            batch_timer.pause()
            self.analysis_recorder.record(self.model, epoch, i_batch)
            epoch_timer.resume()
            batch_timer.resume()

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
            self.analysis_recorder.record_train_loss(loss.item())

            if (i_batch + 1) % 20 == 0:
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

    def _test(self, is_record=True, is_batch=False):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(device), target.to(device)
                output = self.model(data)
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
        if is_record and not is_batch:
            self.analysis_recorder.record_test_acc(acc)
            # test loss
            self.analysis_recorder.record_test_loss(test_loss.item())
        if is_batch:
            batch_timer.stop()
            self.analysis_recorder.record_batch_accuracy(acc)
            batch_timer.start()

    # @profile
    def step(self, epoch):
        # update ctl df
        # df_step = 1 if epoch - self.num_essence < 1 else epoch - self.num_essence
        # self.ri_grow.update_df(df_step)
        # trian

        epoch_timer.start()
        batch_timer.start()
        self._train(epoch)
        epoch_timer.stop()
        batch_timer.stop()

        if grow_control:
            self.pass_cos_sim(epoch)

        # after the controller is updated
        # if epoch > 1:
        #     plot_regression(self.ri_grow.ctls[0].get_svrs(), self.ri_grow.ctls[0].get_data())

        # test
        self._test()

        if old_new_compare:
            if epoch in [2, 4]:
                # remove existing channels to check the accuracy
                n_size = [new - old for new, old in zip(self.new_size, self.old_size)]
                h_dic = register_hook_delete(self.model, old_size=self.old_size, old_fill=0, new_size=n_size, new_fill=1)
                print('add old hook to temporarily delete the existing channel:')
                self._test(is_record=False)
                remove_hook(h_dic)

                # remove new channels to check the accuracy
                h_dic = register_hook_delete(self.model, old_size=self.old_size, old_fill=1, new_size=n_size, new_fill=0)
                print('add new hook to temporarily delete the new channel:')
                self._test(is_record=False)
                remove_hook(h_dic)

        # dump json file
        self.analysis_recorder.json_dump(path['path_to_recorder'])
        # record batch_loss in one epoch
        self.analysis_recorder.record_batch_loss(epoch)

    def pass_cos_sim(self, epoch):
        # pass cs(s,loss) to ri_grow
        cos_sim = window_cos(self.ri_recorder.get_epoch_score(), self.analysis_recorder.get_train_loss(), window_size=5, length=len(self.train_loader))
        size = model_size(find_modules_short(self.model))
        self.ri_grow.update_ctls(cos_sim, size, epoch)

    def grow_policy(self):

        # require growth_list from ri_grow
        epoch_timer.pause()
        control_timer.start()
        growth_list = self.ri_grow.grow_ctls()
        control_timer.stop()
        epoch_timer.resume()
        print('Controller growth: {}'.format(growth_list))
        write_log('Controller growth: {}\n'.format(growth_list), self.path['path_to_log'])

        # control grow size
        for i in range(len(growth_list)):
            if full_size[i] < growth_list[i][1] + self.new_size[i]:
                growth_list[i] = (growth_list[i][0], full_size[i] - self.new_size[i])

        # grow
        self.grow_model(growth_list)

    # @profile
    def grow_model(self, growth_list, inputs=None, target=None):
        # ri_grow return growh_list
        epoch_timer.pause()
        batch_timer.pause()
        self.ri_gate.remove_hook()
        old_size = model_size(find_modules_short(self.model))
        self.old_size = old_size

        print('Before growth: ', end='')
        self._test(is_record=False)

        grow_timer.start()
        self.model = grow(self.model, growth_list, mode=self.mode, ff=ff, inputs=inputs, target=target,
                          test_loader=self.test_loader, criterion=self.criterion)
        grow_timer.stop()

        print('After growth: ', end='')
        self._test(is_record=False)

        self.model = self.model.to(device)

        new_size = model_size(find_modules_short(self.model))
        self.new_size = new_size
        print('Model new size: {}'.format(new_size))
        write_log('Model new size: {}\n'.format(new_size), self.path['path_to_log'])

        if self.mode == 'rank_cosine':
            scale_list = weight_decay(self.model, old_size, growth_list)
            print('After weight decay: ', end='')
            self._test(is_record=False)

            # module_names = ['features.0', 'features.3', 'classifier.0']
            self.ri_gate.register_hook_new(self.model, old_size, new_size, scale_list)
            # self.ri_gate.register_hook(self.model, module_names, old_size, new_size, scale_list=scale_list)
            print('After hook gate: ', end='')
            self._test(is_record=False)

        if self.mode != 'rank_cosine':
            gen_gradient(self.model, self.criterion, inputs, target)

        self.optimizer = replace_optimizer(self.model, self.optimizer)

        # update ri_gate status
        self.ri_gate.set_layer_status(convert_to_status(growth_list))

        self.gate_active = True

        # update cost in ri_grow
        if grow_control:
            ins, outs = grab_input_weight_shape(self.model, input_shape)
            self.ri_grow.update_cost(ins, outs)

        epoch_timer.resume()
        batch_timer.resume()


if __name__ == '__main__':
    # dataset = 'FinDroid'  # CIFAR10,MNIST,EMG, FinDroid ,myHealth
    # dataset = 'Fin_Sitting'  # CIFAR10,MNIST,EMG, FinDroid ,myHealth
    # dataset = 'Fin_SitRun'  # CIFAR10,MNIST,EMG, FinDroid ,myHealth
    dataset = 'myHealth'  # CIFAR10, MNIST, EMG, FinDroid ,myHealth

    model_name = 'MobileNet'  # mobileNet , LSTM, CNN
    old_new_compare = False
    grow_control = False
    is_grow = False
    mode = 'rank_cosine'

    # make the finder can find currect layer structure
    set_mode(model_name)

    # set dataset for flops
    set_dataset(dataset)

    # param = {
    #     'lr': 0.0005,
    #     'epoch': 20,
    #     'batch_size': 100,
    # }

    # initialize timers
    batch_timer = Timer(name='batch')
    grow_timer = Timer(name='grow')
    epoch_timer = Timer(name='epoch')
    control_timer = Timer(name='control')

    # create path
    if is_grow:
        path_name = 'G_{}_{}_'.format(dataset, model_name)
    elif grow_control:
        path_name = 'GC_{}_{}_'.format(dataset, model_name)
    else:
        path_name = 'Standard_{}_{}_'.format(dataset, model_name)

    path = dir_path(path_name)

    if dataset == "Har":
        param = {
            'lr': 0.001,
            'epoch': 25,
            'batch_size': 64,
            'is_bn': False,
        }
        input_shape = (9, 1, 128)
        width = 128

        # kwargs = {
        #     'in_channel': 9,
        #     'out1_channel': 3,
        #     'out2_channel': 6,
        #     'out3_channel': 12,
        #     'out4_channel': 25,
        #     'out_classes': 6,
        #     'kernel_size': 14,
        #     'avg_factor': 2
        # }

        # TMC m2 [6-12-31-39-6]
        kwargs = {
            'in_channel': 9,
            'out1_channel': 6,
            'out2_channel': 12,
            'out3_channel': 31,
            'out4_channel': 39,
            'out_classes': 6,
            'kernel_size': 14,
            'avg_factor': 2
        }

        # kwargs['out1_channel'] = 32
        # kwargs['out2_channel'] = 64
        # kwargs['out3_channel'] = 128
        # kwargs['out4_channel'] = 256
        # kwargs = {
        #     'in_channel': 9,
        #     'out1_channel': 8,
        #     'out2_channel': 16,
        #     'out3_channel': 16,
        #     'out4_channel': 32,
        #     'out_classes': 6,
        #     'kernel_size': 14,
        #     'avg_factor': 2
        # }

        # f_kwargs = {
        #     'in_channel': 9,
        #     'out1_channel': 32,
        #     'out2_channel': 64,
        #     'out3_channel': 128,
        #     'out4_channel': 256,
        #     'out_classes': 6,
        #     'kernel_size': 14,
        #     'avg_factor': 2
        # }

        ff = obtain_ff(kwargs, width)
        kwargs['avg_factor'] = ff
        # torch.manual_seed(1725)
        # model = NetS(**kwargs)
        model = NetS(**kwargs)
        # model.apply(weights_init)
        model = model.to(device)
        # full_size = [32, 64, 34, 256]
        # full_size = [11, 12, 31, 39]
        full_size = [8, 12, 31, 39]

    elif dataset == "EMG":
        param = {
            'lr': 0.005,
            'epoch': 30,
            'batch_size': 64,
            'is_bn': False,
        }
        input_shape = (8, 1, 100)
        width = 100
        # kwargs = {
        #     'in_channel': 8,
        #     'out1_channel': 3,
        #     'out2_channel': 6,
        #     'out3_channel': 12,
        #     'out4_channel': 25,
        #     'out_classes': 6,
        #     'kernel_size': 12,
        #     'avg_factor': 2
        # }

        # TMC M2 [10-12-31-45-6]
        kwargs = {
            'in_channel': 8,
            'out1_channel': 10,
            'out2_channel': 12,
            'out3_channel': 31,
            'out4_channel': 45,
            'out_classes': 6,
            'kernel_size': 12,
            'avg_factor': 2
        }

        # kwargs['out1_channel'] = 32
        # kwargs['out2_channel'] = 64
        # kwargs['out3_channel'] = 128
        # kwargs['out4_channel'] = 256

        ff = obtain_ff(kwargs, width)
        kwargs['avg_factor'] = ff
        # uniform
        # torch.manual_seed(7518)

        # normal
        # torch.manual_seed(9162)

        model = NetS(**kwargs)
        # model = NetS()
        # model.apply(weights_init)
        model = model.to(device)
        # full_size = [32, 64, 128, 256]
        full_size = [10, 25, 31, 45]

    elif dataset == 'myHealth':
        param = {
            'lr': 0.005,
            'epoch': 25,
            'batch_size': 64,
            'is_bn': False,
        }
        width = 100
        input_shape = (23, 1, 100)
        # kwargs = {
        #     'in_channel': 23,
        #     'out1_channel': 3,
        #     'out2_channel': 6,
        #     'out3_channel': 12,
        #     'out4_channel': 25,
        #     'out_classes': 11,
        #     'kernel_size': 12,
        #     'avg_factor': 2
        # }

        # TMC M2  [15-17-23-39-11]
        kwargs = {
            'in_channel': 23,
            'out1_channel': 15,
            'out2_channel': 17,
            'out3_channel': 23,
            'out4_channel': 39,
            'out_classes': 11,
            'kernel_size': 12,
            'avg_factor': 2
        }

        # kwargs['out1_channel'] = 32
        # kwargs['out2_channel'] = 64
        # kwargs['out3_channel'] = 128
        # kwargs['out4_channel'] = 256

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
        # full_size = [20, 64, 128, 256]
        full_size = [32, 64, 128, 256]
        # full_size = [15, 32, 34, 45]

    elif dataset in ['FinDroid', 'Fin_Sitting', 'Fin_SitRun']:
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

        # transfer learning
        # kwargs = {
        #     'in_channel': 6,
        #     'out1_channel': 6,
        #     'out2_channel': 6,
        #     'out3_channel': 11,
        #     'out4_channel': 27,
        #     'out_classes': 6,
        #     'kernel_size': 14,
        #     'avg_factor': 2
        # }
        # param['lr'] = 0.001

        # grow grow
        # kwargs = {
        #     'in_channel': 6,
        #     'out1_channel': 8,
        #     'out2_channel': 14,
        #     'out3_channel': 35,
        #     'out4_channel': 41,
        #     'out_classes': 6,
        #     'kernel_size': 14,
        #     'avg_factor': 2
        # }
        # param['lr'] = 0.006

        # kwargs['out1_channel'] = 32
        # kwargs['out2_channel'] = 64
        # kwargs['out3_channel'] = 128
        # kwargs['out4_channel'] = 256

        ff = obtain_ff(kwargs, width)
        kwargs['avg_factor'] = ff
        # uniform
        # torch.manual_seed(7518)

        # normal
        # torch.manual_seed(9162)

        model = NetS(**kwargs)
        # model.apply(weights_init)
        model = model.to(device)

        # load weight
        # model.load_state_dict(torch.load("/Users/zber/ProgramDev/pytorch-pruning/src/mobile_prunedsize.pt"))
        # model.load_state_dict(torch.load("/Users/zber/ProgramDev/exp_pyTorch/results/GC_Fin_Sitting_MobileNet_20201011-172551/model_{str}.ckpt"))

        full_size = [10, 21, 61, 54]
        # full_size = [8, 14, 35, 41]
        # full_size = [17, 25, 45, 55]
        # full_size = [32, 64, 128, 256]
    # generate data loader
    trainloader, testloader = generate_data_loader(batch_size=param['batch_size'], dataset=dataset, is_main=False)

    # optimizer and criterion
    criterion = nn.CrossEntropyLoss()

    # initialize optimizer
    optimizer = AdamW(model.parameters(), lr=param['lr'])

    num_layers = len(find_modules_short(model))

    # create ri components
    ri_gate = RIGate(num_layers, len(trainloader))

    ri_grow = RIGrow(num_layers, model_name, path)

    ri_recorder = Recorder(num_layers)

    analysis_recorder = AnalysisRecorder(optimizer, 'AdamW', ri_recorder, path['path_to_log'])

    ri_grow_controller = RIController(model, optimizer, criterion, mode, trainloader, testloader, path, batch_timer,
                                      ri_gate, ri_recorder, ri_grow, analysis_recorder)

    for i in range(param['epoch']):
        ri_grow_controller.step(i + 1)

    analysis_recorder.write_txt()
    epoch_timer.dump_json(path['path_to_timer'])
    ri_gate.save_gate_dic_to_json(path['path_to_gate'])
    save_model(model, path['path_to_model'])