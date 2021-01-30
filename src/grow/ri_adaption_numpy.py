from sklearn.metrics.pairwise import cosine_similarity
import torch.nn as nn
import torch
import numpy as np

# np.random.seed(1535)
# torch.manual_seed(1535)

## ##
from utils import test
from grow.growth_utils import get_filter_rank, get_rank, target_std_scale, put_to_ndarray
from optmizer.ours_adam import AdamW

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def find_modules(model):
    modules = []
    for index_str, module in model.features._modules.items():
        if isinstance(module, torch.nn.Conv2d):
            modules.append(module)

    for index_str, module in model.classifier._modules.items():
        if isinstance(module, torch.nn.Linear):
            modules.append(module)
    return modules


def structure_model(model):
    struct = []
    for index_str, module in model.features._modules.items():
        if isinstance(module, torch.nn.Conv2d):
            struct.append((model.features, int(index_str)))

    for index_str, module in model.classifier._modules.items():
        if isinstance(module, torch.nn.Linear):
            struct.append((model.classifier, int(index_str)))
    return struct


def find_layer_index(model):
    num_features = len(model.features)
    num_classifier = len(model.classifier)
    return num_features, num_classifier


def get_layer_finder(model, index):
    struct = structure_model(model)
    finder = struct[index]
    return finder


def replace_multiple_layers(model, first_layer, first_index, second_layer, second_index):
    if isinstance(first_layer, torch.nn.Conv2d) and isinstance(second_layer, torch.nn.Conv2d):
        features = torch.nn.Sequential(
            *(replace_layers(model.features, i, [first_index, second_index],
                             [first_layer, second_layer]) for i, _ in enumerate(model.features)))
        model.features = features
        return model
    elif isinstance(first_layer, torch.nn.Conv2d) and isinstance(second_layer, torch.nn.Linear):
        features = torch.nn.Sequential(
            *(replace_layers(model.features, i, [first_index],
                             [first_layer]) for i, _ in enumerate(model.features)))
        classifier = torch.nn.Sequential(
            *(replace_layers(model.classifier, i, [second_index],
                             [second_layer]) for i, _ in enumerate(model.classifier)))
        model.features = features
        model.classifier = classifier
    elif isinstance(first_layer, torch.nn.Linear) and isinstance(second_layer, torch.nn.Linear):
        classifier = torch.nn.Sequential(
            *(replace_layers(model.classifier, i, [first_index, second_index],
                             [first_layer, second_layer]) for i, _ in enumerate(model.classifier)))
        model.classifier = classifier

    return model


# def replace_multiple_layers_new(model, layer, index):
#     if isinstance(layer, torch.nn.Conv2d):
#         features = torch.nn.Sequential(
#             *(replace_layers(model.features, i, [index],
#                              [layer]) for i, _ in enumerate(model.features)))
#         model.features = features
#
#     elif isinstance(layer, torch.nn.Linear):
#         classifier = torch.nn.Sequential(
#             *(replace_layers(model.classifier, i, [index],
#                              [layer]) for i, _ in enumerate(model.classifier)))
#         model.classifier = classifier
#
#     return model


def replace_layers(model_classifer, layer_index, layer_index_change, layers_change):
    # if the layer in the layer_index is needed to changed, then use layers_change to replace,
    # otherwise return the original ones.
    if layer_index in layer_index_change:
        return layers_change[layer_index_change.index(layer_index)]
    return model_classifer[layer_index]


def model_size(layers):
    size = []
    for module in layers:
        size.append(module.bias.data.shape[0])
    return size


def start_end(start, length):
    return start, start + length


class RIAdaption:
    def __init__(self, mode, all_scale=True, new_bias=True):
        self.old_size = []
        self.new_size = []
        self.flatten_factor = 16
        self.project_index = None
        self.mode = mode
        self.scale_list = []
        self.optimizer_mode = 'AdamW'
        self.is_r = False
        self.all_scale = all_scale
        self.bias_new = new_bias

        # initialization
        # modules
        # self._find_modules()
        # struct
        # self._structure_model()

    def _structure_model(self, model):
        struct = []
        for index_str, module in model.features._modules.items():
            if isinstance(module, torch.nn.Conv2d):
                struct.append((model.features, int(index_str)))

        for index_str, module in model.classifier._modules.items():
            if isinstance(module, torch.nn.Linear):
                struct.append((model.classifier, int(index_str)))
        self.struct = struct

    def _find_modules(self, model):
        modules = []
        for index_str, module in model.features._modules.items():
            if isinstance(module, torch.nn.Conv2d):
                modules.append(module)

        for index_str, module in model.classifier._modules.items():
            if isinstance(module, torch.nn.Linear):
                modules.append(module)
        self.modules = modules

    def get_layer_finder(self, index):
        finder = self.struct[index]
        return finder

    def grow(self, model, growth_list, inputs=None, target=None, criterion=None, testloader=None):
        # reset scale_list
        self.scale_list = []

        # memories old_size
        self._find_modules(model)
        self._structure_model(model)
        self.old_size = model_size(self.modules[:-1])
        test(model, test_loader=testloader, criterion=criterion, to_log=None, is_two_out=False)

        for i, grow_list in enumerate(growth_list):
            model.eval()
            # if i > 0 and growth_list[i - 1][1] > 0:
            #     self.is_r = True
            # else:
            #     self.is_r = False
            first = grow_list[0]
            second = first + 1
            num_growth = grow_list[1]
            # if num_growth == 3:
            #     print('', end='')

            # if inputs is not None:
            #     output = model(inputs)
            #     loss = criterion(output, target)
            #     loss.backward()

            model = self._generator(model, first, second, num_growth)
            # test
            test(model, test_loader=testloader, criterion=criterion, to_log=None, is_two_out=False)

            # if inputs is not None:
            #     output = model(inputs)
            #     loss = criterion(output, target)
            #     loss.backward()

        # self.model = self.model.to(device)
        self._find_modules(model)
        self.new_size = model_size(self.modules[:-1])

        return model

    def _generator(self, model, first=0, second=1, num_growth=0):
        if num_growth == 0:
            if self.mode == 'ours':
                self.scale_list.append(1)
            return 0
        self._find_modules(model)
        m1 = self.modules[first]
        m2 = self.modules[second]
        is_flatten = False if type(m1) == type(m2) else True

        new_m1 = self.create_module(m1, True, num_growth, is_flatten=is_flatten)
        new_m2 = self.create_module(m2, False, num_growth, is_flatten=is_flatten)

        # adaption
        scale = self._adaption(m1, new_m1, True, is_flatten, num_growth)
        # self._adaption(m1, new_m1, True, is_flatten, num_growth)
        self._adaption(m2, new_m2, False, is_flatten, num_growth)

        # replace_multiple_layers(self.model, new_m1, new_m2, second_layer, second_index)

        seq1, index1 = self.get_layer_finder(index=first)
        seq2, index2 = self.get_layer_finder(index=second)

        # self.model = replace_multiple_layers_new(self.model)

        model = replace_multiple_layers(model, new_m1, index1, new_m2,  index2)

        # seq1[index1] = new_m1
        # seq2[index2] = new_m2

        if self.mode == 'ours':
            self.scale_list.append(scale)
        return model

    def optimizer_adapt(self, model, optimizer):
        lr = optimizer.defaults['lr']
        if self.optimizer_mode == 'AdamW':
            new_optimizer = AdamW(model.parameters(), lr=lr)
            new_optimizer.vs = optimizer.vs
            new_optimizer.ms = optimizer.ms
            new_optimizer.gs = optimizer.gs
            new_optimizer.grads = optimizer.grads
            new_optimizer.vg = optimizer.vg

        for current_group, new_group in zip(optimizer.param_groups, new_optimizer.param_groups):
            layer_index = 0
            for current_p, new_p in zip(current_group['params'], new_group['params']):
                # if is_scale and scale_list is not None and layer_index / 2 < 3:
                #     scale = scale_list[int(layer_index / 2)]
                #     scale = 1 / scale
                # else:
                #     scale = 1

                current_state = optimizer.state[current_p]
                state = new_optimizer.state[new_p]

                # State initialization
                state['step'] = current_state['step']

                # Exponential moving average of gradient values
                state['exp_avg'] = torch.zeros_like(new_p.data, memory_format=torch.preserve_format)
                new_exp_avg = state['exp_avg'].data
                current_exp_avg = current_state['exp_avg'].data
                new_exp_avg = put_to_ndarray(current_exp_avg, new_exp_avg)

                # Exponential moving average of squared gradient values
                state['exp_avg_sq'] = torch.zeros_like(new_p.data, memory_format=torch.preserve_format)
                new_exp_avg_sq = state['exp_avg_sq'].data
                current_exp_avg_sq = current_state['exp_avg_sq'].data
                new_exp_avg_sq = put_to_ndarray(current_exp_avg_sq, new_exp_avg_sq)

                # if avg:
                #     mean_exp_avg = torch.mean(current_exp_avg)
                #     mean_exp_avg_sq = torch.mean(current_exp_avg_sq)
                #     new_exp_avg[new_exp_avg == 0.0] = mean_exp_avg
                #     new_exp_avg_sq[new_exp_avg_sq == 0.0] = mean_exp_avg_sq

                # state['exp_avg'].data = torch.from_numpy(new_exp_avg.astype(np.float32))
                # state['exp_avg'].data = state['exp_avg'].data.to(device)
                # state['exp_avg_sq'].data = torch.from_numpy(new_exp_avg_sq.astype(np.float32))
                # state['exp_avg_sq'].data = state['exp_avg_sq'].data.to(device)

                state['exp_avg'].data = state['exp_avg']
                state['exp_avg_sq'].data = state['exp_avg_sq']

                layer_index += 1
        return new_optimizer

    def _copy_paras(self, old, new):
        # weights
        old_weight = old.weight.data.cpu().numpy()
        new_weight = new.weight.data.cpu().numpy()
        put_to_ndarray(old_weight, new_weight)
        # put_to_tensor(old.weight.data, new.weight.data)
        # bias
        old_bias = old.bias.data.cpu().numpy()
        new_bias = new.bias.data.cpu().numpy()
        # put_to_tensor(old.bias.data, new.bias.data)
        put_to_ndarray(old_bias, new_bias)
        # grad
        old_grad = old.weight.grad.data.cpu().numpy()
        new_grad = np.zeros(new_weight.shape)
        put_to_ndarray(old_grad, new_grad)
        # new.weight.grad = new.weight.data.clone().fill_(0)
        # put_to_tensor(old.weight.grad.data, new.weight.grad.data)

        # new.weight.data = torch.from_numpy(new_weight.astype(np.float32)).to(device)
        new.weight.data = torch.from_numpy(new_weight.astype(np.float32))
        # new.bias.data = torch.from_numpy(new_bias.astype(np.float32)).to(device)
        new.bias.data = torch.from_numpy(new_bias.astype(np.float32))
        grad = old.weight.grad
        grad.data = torch.from_numpy(new_grad.astype(np.float32))
        # grad.data = torch.from_numpy(new_grad.astype(np.float32)).to(device)
        new.weight.grad = grad

    def _project_weights_ours_1(self, num_growth, old=torch.nn.Conv2d):

        rank = get_filter_rank(old, ascending=True)[0]
        cos = nn.CosineSimilarity(dim=0)

        cosine_array = []
        old_weights = old.weight.data
        old_size = old_weights.shape[0]

        x_weight = old_weights[rank].flatten()
        for index in range(old_size):
            y_weight = old_weights[index].flatten()
            cosine_array.append(cos(x_weight, y_weight).tolist())
        c_rank = get_rank(cosine_array)

        return c_rank[:num_growth]

    def _project_weights_ours(self, num_growth, old=torch.nn.Conv2d):

        rank = get_filter_rank(old, ascending=True)[0]
        # cos = nn.CosineSimilarity(dim=0)

        cosine_array = []
        old_weights = old.weight.data
        c_out = old_weights.shape[0]
        c_in = old_weights.shape[1]
        # dim = old_weights.dim()
        # old_size = old_weights.shape[0]

        x_weight = old_weights[rank].reshape(c_in, -1).cpu().numpy()

        # x_weight = old_weights[rank].flatten()
        if old_weights.dim() == 4:
            for index in range(c_out):
                y_weight = old_weights[index].reshape(c_in, -1).cpu().numpy()
                cosine_array.append(cosine_similarity(x_weight, y_weight)[0][0])
                # cosine_array.append(cos(x_weight, y_weight).tolist())

        elif old_weights.dim() == 2:
            for index in range(c_out):
                y_weight = old_weights[index].reshape(c_in, -1).cpu().numpy()
                top = np.sum(x_weight * y_weight)
                bottom = np.sqrt(np.sum(x_weight ** 2)) * np.sqrt(np.sum(x_weight ** 2))
                cosine_array.append(top / bottom)
        c_rank = get_rank(cosine_array)

        return c_rank

    def _project_weights_rank(self, old, num_growth):
        rank = get_filter_rank(old, ascending=True)
        return rank[:num_growth]

    def _project_weights_random(self, old, new):
        # new.weight.data.shape
        pass

    def _conv_grow(self, conv, is_first, num_growth, mode='ours', is_flatten=False):
        new_conv = self.create_module(conv, is_first, num_growth, is_flatten)

        # the difference between first and second

        # new weights copy from old or random
        # copy grads from old
        # old weights scale

        pass
        # return first_layer, second_layer

    def _dense_grow(self, first_layer, second_layer, mode='ours'):
        pass
        # return first_layer, second_layer

    def _dw_grow(self, first_layer, second_layer, mode='ours'):
        pass

    def _bn_grow(self):
        pass

    def create_module(self, old_layer, is_first, num_growth, is_flatten=False):
        num_in = 0
        num_out = 0
        if is_first:
            num_out = num_growth
        else:
            num_in = num_growth * self.flatten_factor if is_flatten else num_growth

        if isinstance(old_layer, torch.nn.Conv2d):
            new_layer = self._cov(old_layer, num_in, num_out)
        elif isinstance(old_layer, torch.nn.Linear):
            new_layer = self._linear(old_layer, num_in, num_out)
        else:
            raise Exception('The type of module is not supported yet!')
        return new_layer

    #         isinstance(module, torch.nn.Linear):

    def _cov(self, old_layer, num_in, num_out):
        new_layer = torch.nn.Conv2d(in_channels=old_layer.in_channels + num_in,
                                    out_channels=old_layer.out_channels + num_out,
                                    kernel_size=old_layer.kernel_size,
                                    stride=old_layer.stride,
                                    padding=old_layer.padding,
                                    dilation=old_layer.dilation,
                                    groups=old_layer.groups,
                                    bias=(old_layer.bias is not None))
        return new_layer

    def _linear(self, old_layer, num_in, num_out):
        new_layer = torch.nn.Linear(in_features=old_layer.in_features + num_in,
                                    out_features=old_layer.out_features + num_out,
                                    bias=(old_layer.bias is not None))
        return new_layer

    def _adaption(self, old, new, is_first, is_flatten, num_growth):
        # adaption
        # old_tensor = old.weight.data
        # new_tensor = new.weight.data
        old_tensor = old.weight.data.cpu().numpy()
        new_tensor = new.weight.data.cpu().numpy()
        tensor = new.weight.data
        first_scale = ['rank']
        second_scale = ['rank', 'copy']
        project_index = None
        scale = None
        dim = 0 if is_first else 1
        old_c = old_tensor.shape[dim]
        new_c = new_tensor.shape[dim]
        r = 1 - (new_c - old_c) / new_c

        # scale and project index
        if self.mode == 'ours':
            if is_first:
                project_index = self._project_weights_ours(num_growth, old)
                self.project_index = project_index
                if self.is_r:
                    target = target_std_scale(tensor=tensor, ratio=r)
                else:
                    target = target_std_scale(tensor=tensor, ratio=r)
            else:
                target = target_std_scale(tensor=tensor, ratio=r)
                project_index = self.project_index
            # target = target_std_scale(tensor=new_tensor, ratio=r)
            # std = torch.std(old_tensor)
            std = np.std(old_tensor)
            scale = target / std

            print(self.project_index)

        elif self.mode == 'rank':
            if is_first:
                project_index = self._project_weights_rank(num_growth, old)
            else:
                project_index = self.project_index
            scale = 1 / 2

        # new weight and bias
        new_weight, new_bias = self._gen_new_weights(old, num_growth, scale, project_index, is_first, is_flatten)
        # if scale is not None and is_first:
        #     new_weight, new_bias = new_weight * scale, new_bias * scale
        # elif scale is not None and not is_first:
        #     new_weight = new_weight * scale

        # copy parameters
        self._copy_paras(old, new)
        self._copy_weight_bias(new, old_c, new_weight, new_bias, num_growth, is_first, is_flatten)

        # old weight scale
        # if is_first:
        #     if self.mode in first_scale:
        #         self._weight_scale(new, scale, project_index, is_first)
        #     elif self.mode == 'ours':
        #         scale = self._weight_scale(new, scale, None, is_first, r, num_growth)
        # else:
        #     if self.mode in second_scale:
        #         self._weight_scale(new, scale, project_index, is_first)
        #
        # if is_first:
        #     return scale

    def _copy_weight_bias(self, new, old_c, new_weight, new_bias, num_growth, is_first, is_flatten):
        weight = new.weight.data.cpu().numpy()
        bias = new.bias.data.cpu().numpy()
        if is_flatten and not is_first:
            length = self.flatten_factor
        else:
            length = 1
        s, e = start_end(old_c, num_growth * length)
        if is_first:
            weight[s:e] = new_weight
            bias[s:e] = new_bias
        else:
            weight[:, s:e] = new_weight

        # new.weight.data = torch.from_numpy(weight.astype(np.float32)).to(device)
        # new.bias.data = torch.from_numpy(bias.astype(np.float32)).to(device)

        new.weight.data = torch.from_numpy(weight.astype(np.float32))
        new.bias.data = torch.from_numpy(bias.astype(np.float32))

    def _weight_scale(self, module, scale, project_index=None, is_first=True, r=1, num_growth=0):
        if self.mode == 'ours':
            new_tensor = module.weight.data
            if self.all_scale:
                if self.is_r:
                    target = target_std_scale(tensor=new_tensor, ratio=r)
                else:
                    target = target_std_scale(tensor=new_tensor, ratio=r)
            else:
                target = target_std_scale(tensor=new_tensor, ratio=r)
            std = torch.std(new_tensor)
            # std = np.std(new_tensor.cpu().numpy())
            scale = target / std
            if num_growth > 0:
                module.weight.data[:-num_growth] = module.weight.data[:-num_growth] * scale
                module.bias.data[:-num_growth] = module.bias.data[:-num_growth] * scale
        else:
            for index in project_index:
                if is_first:
                    module.weight.data[index] = module.weight.data[index] * scale
                else:
                    module.weight.data[:, index] = module.weight.data[:, index] * scale
        return scale

    # def _gen_new_weights_old(self, old, num_growth, project_list=None, is_first=True, is_flatten=False, min_value=-0.1,
    #                          max_value=0.1):
    #
    #     if is_flatten and not is_first:
    #         length = self.flatten_factor
    #     else:
    #         length = 1
    #
    #     dim = 0 if is_first else 1
    #     old_weight = old.weight.data
    #     old_bias = old.bias.data
    #     new_bias = torch.zeros(num_growth).uniform_(min_value, max_value)
    #     new_weight = torch.zeros_like(old_weight.narrow(dim, -num_growth * length, num_growth * length))
    #     noise = torch.zeros_like(old_weight.narrow(dim, -num_growth * length, num_growth * length)).uniform_(min_value, max_value)
    #     # noise.uniform_(min_value, max_value)
    #
    #     # if is_first:
    #     #     if project_list is not None:
    #     #         new_bias = torch.zeros(num_growth)
    #     #     else:
    #     #         new_bias = torch.zeros(num_growth).uniform_(min_value, max_value)
    #
    #     if project_list is None:
    #         new_weight.uniform_(min_value, max_value)
    #     else:
    #         for i in range(num_growth):
    #             n_s, n_e = start_end(i, length)
    #             o_s, o_e = start_end(project_list[i], length)
    #             if is_first:
    #                 new_weight[n_s:n_e] = old_weight[o_s:o_e]
    #                 if not self.bias_new:
    #                     new_bias[n_s:n_e] = old_bias[o_s:o_e]
    #             else:
    #                 new_weight[:, n_s:n_e] = old_weight[:, o_s:o_e]
    #         new_weight = new_weight + noise
    #     return new_weight, new_bias

    def _gen_new_weights(self, old, num_growth, scale, project_list=None, is_first=True, is_flatten=False, min_value=-0.1,
                         max_value=0.1):
        new_bias = None

        if is_flatten and not is_first:
            length = self.flatten_factor
        else:
            length = 1

        dim = 0 if is_first else 1
        old_weight = old.weight.data.cpu().numpy()
        dimension = old_weight.ndim
        old_bias = old.bias.data.cpu().numpy()
        # new_bias = torch.zeros(num_growth).uniform_(min_value, max_value)
        if is_first:
            # new_bias = torch.from_numpy(np.random.uniform(min_value, max_value, num_growth).astype(np.float32)).to(device)
            new_bias = np.random.uniform(min_value, max_value, num_growth)

        new_weight_shape = torch.zeros_like(old.weight.data.narrow(dim, -num_growth * length, num_growth * length)).shape
        new_weight = np.zeros(new_weight_shape)
        t_noise = torch.zeros_like(old.weight.data.narrow(dim, -num_growth * length, num_growth * length)).uniform_(min_value, max_value)
        noise_shape = t_noise.shape
        noise = np.random.uniform(min_value, max_value, noise_shape)

        # noise.uniform_(min_value, max_value)

        # if is_first:
        #     if project_list is not None:
        #         new_bias = torch.zeros(num_growth)
        #     else:
        #         new_bias = torch.zeros(num_growth).uniform_(min_value, max_value)

        if project_list is None:
            new_weight.uniform_(min_value, max_value)
        else:
            for i in range(num_growth):
                n_s, n_e = start_end(i * length, length)
                o_s, o_e = start_end(project_list[i] * length, length)

                # if not is_first and dimension == 2:
                #     o_s, o_e = start_end(project_list[i] * length, length)
                # else:
                #     # o_s, o_e = start_end(project_list.index(i), length)
                #     o_s, o_e = start_end(project_list.index(i) * length, length)

                if is_first:
                    new_weight[n_s:n_e] = old_weight[o_s:o_e]
                    if not self.bias_new:
                        new_bias[n_s:n_e] = old_bias[o_s:o_e]
                else:
                    new_weight[:, n_s:n_e] = old_weight[:, o_s:o_e]
        if self.mode == 'ours':
            new_weight = new_weight * scale
            noise = noise * scale
            if new_bias is not None:
                new_bias = new_bias * scale
            new_weight = new_weight + noise
        return new_weight, new_bias

    def get_scale(self):
        if self.mode == "ours":
            new_scale = []
            for scale in self.scale_list:
                new_scale.append(1 / scale)
            return new_scale
        else:
            return None

    def get_old_size(self):
        return self.old_size

    def get_new_size(self):
        return self.new_size

    def print_class(self):
        print(self.old_size )
        print(self.new_size)
        print(self.flatten_factor)
        print(self.project_index)
        print(self.mode)
        print(self.scale_list)
        print(self.optimizer_mode)
        print(self.is_r)
        print(self.all_scale)
        print(self.bias_new)