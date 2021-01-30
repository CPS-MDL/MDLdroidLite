import torch
from torch.autograd import Variable
from torchvision import models
import numpy as np
import torch.nn as nn
import argparse
from operator import itemgetter
from heapq import nsmallest
from prune_filter.prune import prune_vgg16_conv_layer
from utils import test, write_log
from main import train_test
import os


class ModifiedVGG16Model(torch.nn.Module):
    def __init__(self):
        super(ModifiedVGG16Model, self).__init__()

        model = models.vgg16(pretrained=True)
        self.features = model.features

        for param in self.features.parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 2))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class FilterPrunner:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __init__(self, model):
        self.model = model
        self.reset()

    def reset(self):
        self.filter_ranks = {}

    def forward(self, x):
        self.activations = []
        self.gradients = []
        self.grad_index = 0
        self.activation_to_layer = {}

        activation_index = 0
        for layer, (name, module) in enumerate(self.model.features._modules.items()):
            x = module(x)
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                x.register_hook(self.compute_rank)
                self.activations.append(x)
                self.activation_to_layer[activation_index] = layer
                activation_index += 1

        return self.model.classifier(x.view(x.size(0), -1))

    def compute_rank(self, grad):
        activation_index = len(self.activations) - self.grad_index - 1
        activation = self.activations[activation_index]

        taylor = activation * grad
        # Get the average value for every filter, 
        # accross all the other dimensions
        taylor = taylor.mean(dim=(0, 2, 3)).data

        if activation_index not in self.filter_ranks:
            self.filter_ranks[activation_index] = \
                torch.FloatTensor(activation.size(1)).zero_()
            self.filter_ranks[activation_index] = self.filter_ranks[activation_index].to(self.device)

        self.filter_ranks[activation_index] += taylor
        self.grad_index += 1

    def lowest_ranking_filters(self, num):
        data = []
        for i in sorted(self.filter_ranks.keys()):
            for j in range(self.filter_ranks[i].size(0)):
                data.append((self.activation_to_layer[i], j, self.filter_ranks[i][j]))

        return nsmallest(num, data, itemgetter(2))

    def normalize_ranks_per_layer(self):
        for i in self.filter_ranks:
            v = torch.abs(self.filter_ranks[i])
            v = v.cpu()
            v = v / np.sqrt(torch.sum(v * v))
            self.filter_ranks[i] = v.cpu()

    def get_prunning_plan(self, num_filters_to_prune):
        filters_to_prune = self.lowest_ranking_filters(num_filters_to_prune)

        # After each of the k filters are prunned,
        # the filter index of the next filters change since the model is smaller.
        filters_to_prune_per_layer = {}
        for (l, f, _) in filters_to_prune:
            if l not in filters_to_prune_per_layer:
                filters_to_prune_per_layer[l] = []
            filters_to_prune_per_layer[l].append(f)

        for l in filters_to_prune_per_layer:
            filters_to_prune_per_layer[l] = sorted(filters_to_prune_per_layer[l])
            for i in range(len(filters_to_prune_per_layer[l])):
                filters_to_prune_per_layer[l][i] = filters_to_prune_per_layer[l][i] - i

        filters_to_prune = []
        for l in filters_to_prune_per_layer:
            for i in filters_to_prune_per_layer[l]:
                filters_to_prune.append((l, i))

        return filters_to_prune


class PrunningFineTuner_VGG16:
    def __init__(self, train_loader, train2_loader, test_loader, test2_loader, model, optimizer, dic_path, iteration=1,
                 num_to_prune=50,
                 learning_rate=0.0005):
        self.train_data_loader = train_loader
        self.validate_data_loader = train2_loader
        self.test_data_loader = test_loader
        self.test2_data_loader = test2_loader

        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss()
        self.prunner = FilterPrunner(self.model)
        self.model.train()
        self.optimizer = optimizer
        self.dic_path = dic_path
        self.iteration = iteration
        self.num_to_prune = num_to_prune
        self.lr = learning_rate
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def test(self):
        # return
        self.model.eval()
        correct = 0
        total = 0

        for i, (batch, label) in enumerate(self.test_data_loader):
            batch = batch.to(self.device)
            output = self.model(Variable(batch))
            pred = output.data.max(1)[1]
            correct += pred.cpu().eq(label).sum()
            total += label.size(0)

        print("Accuracy :", float(correct) / total)

        self.model.train()

    def train(self, optimizer=None, epoches=10):
        if optimizer is None:
            optimizer = self.optimizer

        for i in range(epoches):
            print("Epoch: ", i)
            self.train_epoch(optimizer)
            self.test()
        print("Finished fine tuning.")

    def train_batch(self, optimizer, batch, label, rank_filters):

        batch = batch.to(self.device)
        label = label.to(self.device)

        self.model.zero_grad()
        input = Variable(batch)

        if rank_filters:
            output = self.prunner.forward(input)
            self.criterion(output, Variable(label)).backward()
        else:
            self.criterion(self.model(input), Variable(label)).backward()
            optimizer.step()

    def train_epoch(self, optimizer=None, rank_filters=False):
        for i, (batch, label) in enumerate(self.train_data_loader):
            self.train_batch(optimizer, batch, label, rank_filters)

    def get_candidates_to_prune(self, num_filters_to_prune):
        self.prunner.reset()
        self.train_epoch(rank_filters=True)
        self.prunner.normalize_ranks_per_layer()
        return self.prunner.get_prunning_plan(num_filters_to_prune)

    def total_num_filters(self):
        filters = 0
        for name, module in self.model.features._modules.items():
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                filters = filters + module.out_channels
        return filters

    def prune(self):
        # Get the accuracy before prunning
        _, acc1 = test(self.model, self.test_data_loader, self.criterion, self.dic_path['path_to_log'])
        self.model.train()

        # Make sure all the layers are trainable
        for param in self.model.features.parameters():
            param.requires_grad = True

        number_of_filters = self.total_num_filters()

        num_filters_to_prune_per_iteration = self.num_to_prune
        iterations = self.iteration

        # num_filters_to_prune_per_iteration = 45
        # iterations = int(float(number_of_filters) / num_filters_to_prune_per_iteration)
        #
        # iterations = int(iterations * 2.0 / 3)
        format_str = 'Number of prunning iterations to reduce 67% filters is : {}\n'.format(iterations)
        print(format_str)
        write_log(format_str, self.dic_path['path_to_log'])
        while True:
            print('Ranking filters.. \n')
            prune_targets = self.get_candidates_to_prune(num_filters_to_prune_per_iteration)
            layers_prunned = {}
            for layer_index, filter_index in prune_targets:
                if layer_index not in layers_prunned:
                    layers_prunned[layer_index] = 0
                layers_prunned[layer_index] = layers_prunned[layer_index] + 1

            print("Layers that will be prunned", layers_prunned)
            write_log('the number of filter in the layer will be prunned is {} \n'.format(layers_prunned),
                      self.dic_path['path_to_log'])
            print("Prunning filters.. ")
            model = self.model.cpu()
            temp_model = self.model.cpu()
            for layer_index, filter_index in prune_targets:
                model = prune_vgg16_conv_layer(model, layer_index, filter_index, use_cuda=False)

            self.model = model.to(self.device)

            message = str(100 * float(self.total_num_filters()) / number_of_filters) + "%"
            print("Filters prunned", str(message))
            write_log('{} filters are remained\n'.format(message), self.dic_path['path_to_log'])
            write_log('after pruning, test1:\n', self.dic_path['path_to_log'])
            _, acc1 = test(self.model, self.test_data_loader, self.criterion, self.dic_path['path_to_log'])
            write_log('after pruning, test2:\n', self.dic_path['path_to_log'])
            _, acc2 = test(self.model, self.test2_data_loader, self.criterion, self.dic_path['path_to_log'])
            print("Fine tuning to recover from prunning iteration.\nAnd save prunned model.")
            # model_path = os.path.join(self.dic_path['res_dir'], 'pruned_model.ckpt')
            # torch.save(model.state_dict(), model_path)
            optimizer = self.optimizer
            # self.train(optimizer, epoches=5)
            if acc1 < 60.00:
                self.model = temp_model
                break

        print("Finished. Going to fine tune the model a bit more")
        write_log('after pruning, test1:\n', self.dic_path['path_to_log'])
        _, acc1 = test(self.model, self.test_data_loader, self.criterion, self.dic_path['path_to_log'])
        write_log('after pruning, test2:\n', self.dic_path['path_to_log'])
        _, acc2 = test(self.model, self.test2_data_loader, self.criterion, self.dic_path['path_to_log'])
        # self.train(optimizer, epoches=15)
        # finetune the model by retraining the model in 15 epochs
        write_log('finetune the model by retraining the model.\n', self.dic_path['path_to_log'])
        train_test(model=self.model, trainloader=self.validate_data_loader, testloader=self.test2_data_loader,
                   learning_rate=self.lr, epoch=15, dic_path=self.dic_path, str_channel='prunned_train')
        write_log('finetune is finished.\n', self.dic_path['path_to_log'])
        write_log('after finetune, test1:\n', self.dic_path['path_to_log'])
        _, acc1 = test(self.model, self.test_data_loader, self.criterion, self.dic_path['path_to_log'])
        write_log('after finetune, test2:\n', self.dic_path['path_to_log'])
        _, acc2 = test(self.model, self.test2_data_loader, self.criterion, self.dic_path['path_to_log'])

        dir = os.path.join(self.dic_path['res_dir'], 'pruned_model.ckpt')
        torch.save(self.model.state_dict(), dir)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", dest="train", action="store_true")
    parser.add_argument("--prune", dest="prune", action="store_true")
    parser.add_argument("--train_path", type=str, default="train")
    parser.add_argument("--test_path", type=str, default="test")
    parser.add_argument('--use-cuda', action='store_true', default=False, help='Use NVIDIA GPU acceleration')
    parser.set_defaults(train=False)
    parser.set_defaults(prune=False)
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()

    return args


if __name__ == '__main__':
    args = get_args()

    if args.train:
        model = ModifiedVGG16Model()
    elif args.prune:
        model = torch.load("model", map_location=lambda storage, loc: storage)

    if args.use_cuda:
        model = model.cuda()

    fine_tuner = PrunningFineTuner_VGG16(args.train_path, args.test_path, model)

    if args.train:
        fine_tuner.train(epoches=10)
        torch.save(model, "model")

    elif args.prune:
        fine_tuner.prune()
