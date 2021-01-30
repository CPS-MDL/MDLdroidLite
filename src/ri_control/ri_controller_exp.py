import torch
import torch.nn as nn
import time
import numpy as np
import os
import datetime
from torch.optim import Adam
from numpy import dot
from numpy.linalg import norm
import math
import json

# ours
from grow.gate_hook import register_hook_delete, remove_hook
from ri_control.ri_grow import RIGrow
from utils import Timer, AverageMeter, write_log, accuracy, weights_init
from data_loader import generate_data_loader
from ri_control.recorder import Recorder
from model.CNN import LeNet5_GROW_P, LeNet5
from grow.analysis_recorder import AnalysisRecorder
from ri_control.ri_gate import RIGate
from optmizer.ours_adam import AdamW
from grow.weight_decay import weight_decay
from grow.growth_utils import find_modules_short, model_size
from ri_control.ri_adaption import grow
from utils import device


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


def gen_gradient(model, criterion, inputs, target):
    feature_out, output = model(inputs)
    loss = criterion(output, target)
    loss.backward()
    return feature_out


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
    return cosine


def generator(window=50, length=600):
    for i in range(0, length, window):
        yield i, i + window


def obtain_ff(dic, width=128):
    after_first = (width - dic['kernel_size'] + 1) // 2
    ff = (after_first - dic['kernel_size'] + 1) // 2
    return ff


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
        self.print_freq = param['print_freq']
        self.weight_json = {}

    def save_weight(self, index_layer, epoch):
        layers = find_modules_short(self.model)
        target_layer = layers[index_layer]
        weight = target_layer.weight.data.cpu().numpy()
        flatten_weight = weight.reshape(-1)
        flatten_weight = np.abs(flatten_weight)
        self.weight_json[epoch] = flatten_weight.tolist()

    def dump_weight_json(self):
        target_path = os.path.join(self.path['res_dir'], 'weight.json')
        with open(target_path, 'w') as f:
            json.dump(self.weight_json, f, indent=4)

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

        # start batch timer
        batch_timer.start()

        for i_batch, (inputs, target) in enumerate(self.train_loader):

            # measure data loading time
            data_time.update(time.time() - start)

            # put data into available devices
            inputs, target = inputs.to(device), target.to(device)

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # gradient and do SGD step
            self.model.train()
            features_output, output = self.model(inputs)
            loss = self.criterion(output, target)
            loss.backward()

            # ri_gate
            # if self.gate_active and self.mode == 'rank_cosine':
            #
            #     self.ri_gate.step(self.ri_recorder.get_pre_score(), self.ri_recorder.get_score(),
            #                       self.ri_recorder.get_pre_weight(), self.ri_recorder.get_weight(),
            #                       self.old_size, step=i_batch)
            #
            #     # ri_gate.record_gate(epoch, i_batch)
            #     if i_batch == len(self.train_loader) - 1:
            #         self.ri_gate.reset_grad_gate()
            #         self.ri_gate.reset()
            #
            # if self.old_size and self.mode == 'rank_cosine':
            #     self.ri_gate.gradient_decay(self.model, self.old_size)

            # during the training, data will be written in the grow controller
            # record all information in

            # optimizer step
            self.optimizer.step()

            # record in and out and og value is in each batch epoch
            self.ri_recorder.record(self.model)

            # place trigger grow
            # if epoch in [2, 4, 7, 10] and first_time:
            #     growth_list = [(0, 2), (1, 3), (2, 10)]
            #     # if epoch in [] and i_batch == 0:
            #     self.grow_model(growth_list, inputs, target)
            #     first_time = False

            # # if grow trigger grow
            # if epoch in [2, 4, 7, 10] and first_time and self.mode == 'rank_cosine':
            #     growth_list = [(0, 2), (0, 3), (0, 10)]
            #     # if epoch in [] and i_batch == 0:
            #     self.grow_model(growth_list,inputs, target)
            #     first_time = False

            if epoch > 1 and first_time and (self.mode == 'bridging' or self.mode == 'copy_n'):
                growth_list = []
                over = [20, 50, 500]
                size = model_size(find_modules_short(self.model))
                g_size = [1, 2, 20]
                for i in range(len(over)):
                    if g_size[i]>size[i]:
                        new_size = size[i] + size[i]
                    else:
                        new_size = size[i] + g_size[i]
                    if new_size > over[i]:
                        new_size = over[i]
                    grow_size = new_size - size[i]
                    growth_list.append((i, grow_size))
                self.grow_model(growth_list, inputs, target)
                size = model_size(find_modules_short(self.model))
                print('New size is:{}'.format(size))
                write_log('New size is:{}\n'.format(size), self.path['path_to_log'])
                first_time = False

            # rank grow 60%
            if epoch % 3 == 0 and first_time and self.mode == 'rank_baseline':
            # if epoch % 3 == 0 and first_time:  #and self.mode == 'rank_baseline'
            # if epoch == 4 and first_time:  # and self.mode == 'rank_baseline'
                over = [20, 50, 500]
                size = model_size(find_modules_short(self.model))
                print('Current size is:{}'.format(size))
                growth_list = []
                for i in range(len(over)):
                    new_size = math.ceil(size[i] * 1.6)
                    if new_size > over[i]:
                        new_size = over[i]
                    grow_size = new_size - size[i]
                    growth_list.append((i, grow_size))
                # save weight
                # self.save_weight(1, 'before')

                self.grow_model(growth_list, inputs, target)
                size = model_size(find_modules_short(self.model))
                print('New size is:{}'.format(size))
                write_log('New size is:{}\n'.format(size), self.path['path_to_log'])
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

            if (i_batch + 1) % self.print_freq == 0:
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
                if batch_test:
                    self._test(is_record=False, is_batch=True)
                    if (i_batch + 1) < len(self.train_loader):
                        batch_timer.start()

    def _test(self, is_record=True, is_batch=False):
        if is_batch:
            batch_timer.stop()
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(device), target.to(device)
                _, output = self.model(data)
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
        if is_record:
            self.analysis_recorder.record_test_acc(acc)
            # test loss
            self.analysis_recorder.record_test_loss(test_loss.item())
        if is_batch:
            self.analysis_recorder.record_batch_accuracy(acc)

    def step(self, epoch):
        # trian
        global old_new_compare

        epoch_timer.start()
        self._train(epoch)
        epoch_timer.stop()
        if not batch_test:
            batch_timer.stop()

        # save weight
        if epoch >= 3:
            self.save_weight(1, epoch)

        # test
        self._test()

        if old_new_compare:
            if epoch in [8]:
                # remove existing channels to check the accuracy
                old_size = [2, 5, 10]
                n_size = [new - old for new, old in zip(self.new_size, old_size)]
                h_dic = register_hook_delete(self.model, old_size=old_size, old_fill=0, new_size=n_size, new_fill=1)
                print('add old hook to temporarily delete the existing channel:')
                write_log('add old hook to temporarily delete the existing channel:', self.path['path_to_log'])
                self._test(is_record=True)
                remove_hook(h_dic)

                # remove new channels to check the accuracy
                # h_dic = register_hook_delete(self.model, old_size=self.old_size, old_fill=1, new_size=n_size, new_fill=0)
                # print('add new hook to temporarily delete the new channel:')
                # self._test(is_record=False)
                # remove_hook(h_dic)

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
        growth_list = self.ri_grow.grow_ctls()
        # grow
        self.grow_model(growth_list)

    def grow_model(self, growth_list, inputs=None, target=None):
        # ri_grow return growh_list
        epoch_timer.pause()
        batch_timer.pause()
        self.ri_gate.remove_hook()
        old_size = model_size(find_modules_short(self.model))
        self.old_size = old_size

        self.model = grow(self.model, growth_list, mode=self.mode, ff=ff, inputs=inputs, target=target,
                          test_loader=self.test_loader, criterion=self.criterion)

        print('After growth: ', end='')
        self._test(is_record=False)

        self.model = self.model.to(device)

        new_size = model_size(find_modules_short(self.model))
        self.new_size = new_size

        if self.mode == 'rank_cosine':
            scale_list = weight_decay(self.model, old_size)
            print('After weight decay: ', end='')
            self._test(is_record=False)

            module_names = ['features.0', 'features.3', 'classifier.0']
            # self.ri_gate.register_hook_new(self.model, old_size, new_size, scale_list)
            self.ri_gate.register_hook(self.model, module_names, old_size, new_size, scale_list=scale_list)
            print('After hook gate: ', end='')
            self._test(is_record=False)

        if self.mode != 'rank_cosine':
            gen_gradient(self.model, self.criterion, inputs, target)

        self.optimizer = replace_optimizer(self.model, self.optimizer)
        self.gate_active = True

        epoch_timer.resume()
        batch_timer.resume()


if __name__ == '__main__':
    dataset = 'MNIST'  # CIFAR10,MNIST,EMG,Har
    model_name = 'CNN'  # mobileNet,LSTM,CNN
    old_new_compare = False
    batch_test = False
    # mode = 'bridging'
    # mode = 'rank_baseline'
    # mode = 'copy_n'
    # mode = 'rank_cosine'
    # mode = 'standard'

    mode_list = [
                    'rank_baseline',
                    'bridging',
                    # 'copy_n',
                    # 'non_grow'
                    # 'standard',
                ]*2

    # mode_list = dict(
    #     standard_10_25_250={'out1': 10, 'out2': 25, 'fc1': 250},
    #     standard_14_35_350={'out1': 14, 'out2': 35, 'fc1': 350},
    #     standard_18_45_450={'out1': 18, 'out2': 45, 'fc1': 450},
    # )

    param = {
        'lr': 0.0005,
        # 'lr': 0.001,
        'epoch': 30,
        'batch_size': 64,
        'shuffle': True,
        'print_freq': 100,
        'is_bn': False,
    }

    # training
    # for key in mode_list.keys():
    for mode in mode_list:

        # initialize timers
        batch_timer = Timer(name='batch')
        grow_timer = Timer(name='grow')
        epoch_timer = Timer(name='epoch')

        # create path
        path_name = '{}_{}_'.format(dataset, mode)
        # path_name = key + '_'
        path = dir_path(path_name)

        # set torch and numpy random seed
        # torch.manual_seed(1535)
        # np.random.seed(1535)

        # para loading
        if dataset == 'Har':
            kwargs_har = {
                'in_channel': 9,
                'out1_channel': 2,
                'out2_channel': 5,
                'fc': 10,
                'out_classes': 6,
                'kernel_size': 14,
                'flatten_factor': 0
            }
            ff = obtain_ff(kwargs_har)
            kwargs_har['flatten_factor'] = ff
            trainloader, testloader = generate_data_loader(batch_size=param['batch_size'], dataset=dataset, is_main=False)
            model = LeNet5(**kwargs_har)
            # model.apply(weights_init)

        else:
            # generate data loader
            trainloader, testloader = generate_data_loader(batch_size=param['batch_size'], dataset=dataset, is_main=False)

            # para_model = {'out1': 18, 'out2': 23, 'fc1': 50}
            # para_model = {'out1': 10, 'out2': 25, 'fc1': 250}
            para_model = {'out1': 2, 'out2': 5, 'fc1': 10}
            # para_model = {'out1': 5, 'out2': 12, 'fc1': 125}

            # model = LeNet5_GROW_P(**mode_list[key])
            # create model
            model = LeNet5_GROW_P(**para_model)

        model = model.to(device)

        num_layers = len(find_modules_short(model))

        # optimizer and criterion[
        criterion = nn.CrossEntropyLoss()

        # initialize optimizer
        optimizer = AdamW(model.parameters(), lr=param['lr'])

        # create ri components
        ri_gate = RIGate(num_layers)

        # ri_grow = RIGrow(num_layers)

        ri_grow = None

        ri_recorder = Recorder(num_layers)

        analysis_recorder = AnalysisRecorder(optimizer, 'AdamW', ri_recorder, path['path_to_log'])

        ri_grow_controller = RIController(model, optimizer, criterion, mode, trainloader, testloader, path, batch_timer,
                                          ri_gate, ri_recorder, ri_grow, analysis_recorder)

        for i in range(param['epoch']):
            ri_grow_controller.step(i + 1)

        # dump weight json
        ri_grow_controller.dump_weight_json()
        analysis_recorder.write_txt()
        epoch_timer.dump_json(path['path_to_timer'])
        ri_gate.save_gate_dic_to_json(path['path_to_gate'])
        # with open(path['path_to_json1'], 'w') as f:
        #     json.dump(analysis_recorder.get_recorder_dic(), f, indent=4)
