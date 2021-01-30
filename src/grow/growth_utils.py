from torch.nn.init import _calculate_fan_in_and_fan_out
import math
import torch
from optmizer.ours_adam import AdamW
from torch.optim import Adam
import numpy as np

mode = 'LeNet'


def set_mode(mode_name):
    global mode
    mode = mode_name


def model_size(layers):
    size = []
    for module in layers:
        size.append(module.weight.data.shape[0])
    return size


# def find_modules(model):
#     modules = []
#     for index_str, module in model.features._modules.items():
#         if isinstance(module, torch.nn.Conv2d):
#             modules.append(module)
#
#     for index_str, module in model.classifier._modules.items():
#         if isinstance(module, torch.nn.Linear):
#             modules.append(module)
#     return modules


def find_modules(model):
    modules = []
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            modules.append(module)
    return modules


def find_modules_short(model):
    global mode
    if mode == 'MobileNet':
        return find_modules_recursive_short(model)
    else:
        return find_modules(model)[:-1]


def find_modules_recursive_short(model):
    short_size_list = [0, 2, 4, 6]
    modules = []
    modules_short = []
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            modules.append(module)

    for i, m in enumerate(modules):
        if i in short_size_list:
            modules_short.append(m)

    return modules_short


def structure_model(model):
    struct = []
    for index_str, module in model.features._modules.items():
        if isinstance(module, torch.nn.Conv2d):
            struct.append((model.features, int(index_str)))

    for index_str, module in model.classifier._modules.items():
        if isinstance(module, torch.nn.Linear):
            struct.append((model.classifier, int(index_str)))
    return struct


def get_rank(result, ascending=False):
    result = np.asarray(result)
    if not ascending:
        result = result * -1
    order = result.argsort()
    rank = order.tolist()
    return rank


def target_std_scale(tensor, mode='Kaiming', ratio=0.5):
    if mode == 'Kaiming':
        fan_in, _ = _calculate_fan_in_and_fan_out(tensor)
        scale = 1 / math.sqrt(fan_in)
        std = scale * (1 / math.sqrt(3))
        if tensor.dim() == 2:
            std = std * ratio
    elif mode == 'Xavier':
        fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
        scale = math.sqrt(6) / math.sqrt(fan_in + fan_out)
        std = scale * (1 / math.sqrt(3))
    return std


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


def put_to_tensor(small, large):
    if small.dim() == 4:
        s_out, s_in, k_h, k_w = small.shape
        large[:s_out, :s_in, :k_h, :k_w] = small
    elif small.ndim == 2:
        s_out, s_in = small.shape
        large[:s_out, :s_in] = small
    elif small.ndim == 1:
        s_out = small.shape[0]
        large[:s_out] = small
    return large


def get_filter_rank(old_layer, is_first=True, ascending=False, is_vg=False):
    global rank_count
    global optimizer
    weights_tensor = old_layer.weight.data.cpu().numpy()
    grad_tensor = old_layer.weight.grad.data.cpu().numpy()
    s_tensor = grad_tensor * weights_tensor
    if isinstance(old_layer, torch.nn.Conv2d):
        if is_first:
            result = np.sqrt(np.sum(np.abs(s_tensor), axis=(1, 2, 3)))
        else:
            result = np.sqrt(np.sum(np.abs(s_tensor), axis=(0, 2, 3)))
    elif isinstance(old_layer, torch.nn.Linear):
        if is_first:
            result = np.sqrt(np.sum(np.abs(s_tensor), axis=1))
        else:
            result = np.sqrt(np.sum(np.abs(s_tensor), axis=0))
    rank = get_rank(result, ascending=ascending)
    return rank


def adapt_optimizer(model, optimizer, learning_rate, avg=False, scale_list=None, is_scale=False,
                    optimizer_mode='Adam'):
    # optimizer = optimizer
    if optimizer_mode == 'Adam':
        new_optimizer = Adam(model.parameters(), lr=learning_rate)
    if optimizer_mode == 'AdamW':
        new_optimizer = AdamW(model.parameters(), lr=learning_rate)
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


def gen_gradient(model, inputs, target, criterion):
    output = model(inputs)
    loss = criterion(output, target)
    loss.backward()
