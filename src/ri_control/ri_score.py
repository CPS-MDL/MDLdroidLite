from operator import itemgetter
from heapq import nsmallest
import torch
import torch.nn.functional as F

from utils import device
from grow.growth_utils import find_modules_short, find_modules

class RIScore:
    def __init__(self, model):
        self.model = model
        self.reset()

    def reset(self):
        self.filter_ranks = {}

    def set_model(self, model):
        self.model = model

    def forward(self, x):
        self.reset()
        self.activations = []
        self.gradients = []
        self.grad_index = 0
        self.activation_to_layer = {}

        key_moduels = find_modules_short(self.model)
        activation_index = 0

        for layer, (name, module) in enumerate(self.model.features._modules.items()):
            x = module(x)
            # if isinstance(module, torch.nn.modules.conv.Conv2d):

            if module in key_moduels:
                x.register_hook(self.compute_rank)
                self.activations.append(x)
                self.activation_to_layer[activation_index] = layer
                activation_index += 1

        x = x.view(x.size(0), -1)

        for layer, (name, module) in enumerate(self.model.classifier._modules.items()):
            x = module(x)
            # if isinstance(module, torch.nn.modules.conv.Conv2d):

            if module in key_moduels:
                x.register_hook(self.compute_rank)
                self.activations.append(x)
                self.activation_to_layer[activation_index] = layer
                activation_index += 1

        # return self.model.classifier(x.view(x.size(0), -1))
        return F.softmax(x, dim=1)

    def compute_rank(self, grad):
        activation_index = len(self.activations) - self.grad_index - 1
        activation = self.activations[activation_index]

        taylor = activation * grad
        taylor = torch.abs(taylor)
        # Get the average value for every filter,
        # accross all the other dimensions
        if grad.dim() > 2:
            taylor = taylor.mean(dim=(0, 2, 3)).data

        else:
            taylor = taylor.mean(dim=(0)).data

        if activation_index not in self.filter_ranks:
            self.filter_ranks[activation_index] = torch.FloatTensor(activation.size(1)).zero_()

            self.filter_ranks[activation_index] = self.filter_ranks[activation_index].to(device)

        self.filter_ranks[activation_index] += taylor
        self.grad_index += 1

    def get_s_score(self):
        return self.filter_ranks

    def lowest_ranking_filters(self, num):
        data = []
        for i in sorted(self.filter_ranks.keys()):
            for j in range(self.filter_ranks[i].size(0)):
                data.append((self.activation_to_layer[i], j, self.filter_ranks[i][j]))

        return nsmallest(num, data, itemgetter(2))

    def normalize_ranks_per_layer(self):
        for i in self.filter_ranks:
            v = torch.abs(self.filter_ranks[i])
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
