import torch
import torch.nn as nn
import time
import numpy as np
import os
import datetime

# ours
from utils import device, Timer, AverageMeter, write_log, accuracy
from main import generate_data_loader
from ri_control.recorder import Recorder
from model.CNN import LeNet5_P
from grow.ri_adaption_old import RIAdaption
from grow.analysis_recorder import AnalysisRecorder
from ri_control.ri_gate import RIGate
from optmizer.ours_adam import AdamW


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


def model_loader(mode):
    if mode == 'a':
        pass
    return 0


class RIController:
    def __init__(self, model, optimizer, criterion, mode, train_loader, test_loader, path, batch_timer, ri_gate,
                 ri_adaption, ri_recorder, ri_grow, analysis_recorder):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.mode = mode
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.path = path
        self.batch_timer = batch_timer
        self.ri_gate = ri_gate
        self.ri_adaption = ri_adaption
        self.ri_recorder = ri_recorder
        self.ri_grow = ri_grow
        self.analysis_recorder = analysis_recorder
        self.gate_active = False
        self.batch_loss = []

    def _train(self, epoch):
        # train_loss = []
        self.gate_active = False

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
            if self.gate_active:
                self.ri_gate.step(self.ri_recorder.get_pre_score(), self.ri_recorder.get_score(),
                                  self.ri_recorder.get_pre_weight(), self.ri_recorder.get_weight(),
                                  self.ri_adaption.get_old_size(), step=i_batch)
                ri_gate.record_gate(epoch, i_batch)
                if i_batch == 599:
                    self.ri_gate.reset_grad_gate()
                    self.ri_gate.reset()

            if self.ri_adaption.get_old_size():
                self.ri_gate.gradient_decay(self.model, self.ri_adaption.get_old_size())

            # during the training, data will be written in the grow controller
            # record all information in

            # optimizer step
            self.optimizer.step()

            # record in and out and og value is in each batch epoch
            self.ri_recorder.record(self.model)

            # if grow trigger grow
            if epoch in [2, 4, 7, 10] and i_batch == 0:
                self.grow_model(inputs, target)

            # record each epoch in analysis
            self.analysis_recorder.record(self.model, epoch, i_batch)

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

    def _test(self):
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
        self.analysis_recorder.record_test_acc(acc)
        # test loss
        self.analysis_recorder.record_test_loss(test_loss.item())

    def step(self, epoch):

        # if epoch in [2, 4, 7, 10]:
        #     # ri_adaption grow according to growth_list
        #     self.ri_gate.remove_hook()
        #     self.model = self.ri_adaption.grow(growth_list)
        #     # self.optimizer = AdamW(self.model.parameters(), lr=0.0005)
        #     self.optimizer = self.ri_adaption.optimizer_adapt()
        #     scale_list = self.ri_adaption.get_scale()
        #     print('scale list: {}'.format(scale_list))
        #     old_size = self.ri_adaption.get_old_size()
        #     new_size = self.ri_adaption.get_new_size()
        #     self.ri_gate.register_hook_new(self.model, old_size, new_size, scale_list)
        #     self._test()
        #
        #     # gate will be running during next epoch
        #     self.gate_active = True
        # else:
        # gate will not be running during next epoch

        # trian
        epoch_timer.start()
        self._train(epoch)
        epoch_timer.stop()

        # record batch_loss in one epoch
        self.analysis_recorder.record_batch_loss()

        # test
        self._test()

    def grow_model(self, inputs=None, target=None):
        # ri_grow return growh_list
        growth_list = [(0, 2), (1, 3), (2, 10)]

        self.ri_gate.remove_hook()
        self.model = self.ri_adaption.grow(growth_list, inputs, target, self.criterion, self.test_loader)

        scale_list = self.ri_adaption.get_scale()
        print('scale list: {}'.format(scale_list))
        old_size = self.ri_adaption.get_old_size()
        new_size = self.ri_adaption.get_new_size()
        self.ri_gate.register_hook_new(self.model, old_size, new_size, scale_list)
        # make grad avaliable
        output = self.model(inputs)  # save acv after grow
        loss = self.criterion(output, target)
        loss.backward()
        self.optimizer = self.ri_adaption.optimizer_adapt()
        self._test()
        self.gate_active = True


if __name__ == '__main__':
    dataset = 'MNIST'  # CIFAR10,MNIST,EMG
    model_name = 'CNN'  # mobileNet , LSTM, CNN
    t_channel = 8  # [8,12,16,24] for experiment，or standard

    param = {
        'lr': 0.0005,
        'epoch': 10,
        'batch_size': 100,
        'shuffle': True,
    }

    para_model = {'out1': 2, 'out2': 5, 'fc1': 10}

    mode = 'ours'

    # training

    # initialize timers
    batch_timer = Timer(name='batch')
    grow_timer = Timer(name='grow')
    epoch_timer = Timer(name='epoch')

    # create path
    path_name = 'grow_{}'.format(mode)
    path = dir_path(path_name)

    # set torch and numpy random seed
    # torch.manual_seed(1535)
    # np.random.seed(1535)

    # create model
    model = LeNet5_P(**para_model)
    model = model.to(device)

    # generate data loader
    trainloader, testloader = generate_data_loader(batch_size=param['batch_size'], dataset=dataset)

    # optimizer and criterion
    criterion = nn.CrossEntropyLoss()

    # initialize optimizer
    optimizer = AdamW(model.parameters(), lr=param['lr'])

    # create ri components
    ri_adaption = RIAdaption(model, optimizer, mode)
    ri_gate = RIGate(model)

    # ri_grow = RIGrow()
    ri_grow = None
    ri_recorder = Recorder()

    analysis_recorder = AnalysisRecorder(optimizer, 'AdamW', ri_recorder)

    ri_grow_controller = RIController(model, optimizer, criterion, mode, trainloader, testloader, path, batch_timer,
                                      ri_gate, ri_adaption, ri_recorder, ri_grow, analysis_recorder)

    for i in range(param['epoch']):
        ri_grow_controller.step(i + 1)

    analysis_recorder.json_dump(path['path_to_recorder'])
    analysis_recorder.write_txt(path['path_to_log'])
    epoch_timer.dump_json(path['path_to_timer'])
    ri_gate.save_gate_dic_to_json(path['path_to_gate'])
    # with open(path['path_to_json1'], 'w') as f:
    #     json.dump(analysis_recorder.get_recorder_dic(), f, indent=4)
