import torch
import math

from utils import target_std_scale_calculator
from utils import target_scale_calculator
from utils import device
from grow.growth_utils import find_modules_short
################################


def weight_decay(model, ex_index, growth_list, mode='std'):
    layers = find_modules_short(model)
    scale_list = []
    for i, layer in enumerate(layers):
        # if iterator layer is not grow then, give the scale as 1
        if growth_list[i][1] == 0:
            scale_list.append(torch.empty(1).fill_(1).to(device))
            continue

        weight = layer.weight.data.to(device)

        if mode == 'weight':
            mx_weight = torch.max(torch.abs(torch.min(weight)), torch.abs(torch.max(weight)))
            scale_factor = target_scale_calculator(weight)
        elif mode == 'std':
            # mx_weight = torch.std(weight[:ex_index[i]])
            mx_weight = torch.std(weight)
            r = 1 - ((weight.shape[0] - ex_index[i]) / weight.shape[0])
            scale_factor = target_std_scale_calculator(weight, ratio=r)
        scale = scale_factor / mx_weight
        t_factor = torch.empty(ex_index[i]).fill_(scale)
        t_factor = _reshape_tensor(weight.dim(), t_factor).to(device)

        # down scale weights and bias
        weight[:ex_index[i]] = weight[:ex_index[i]].mul(t_factor)
        # weight = weight.mul(scale)
        if layer.bias is not None:
            bias = layer.bias.data.to(device)
            bias[:ex_index[i]] = bias[:ex_index[i]].mul(scale)
        scale_list.append(1 / scale)

        # if weight.dim() == 4:
        #     c_out, c_in, k_h, k_w = weight.shape
        #     scale_factor = 1 / math.sqrt(c_in * k_h * k_w)
        #
        # elif weight.dim() == 2:
        #     c_out, c_in = weight.shape
        #     scale_factor = 1 / math.sqrt(c_in)

        # left_step = total_step - step
        # factor = math.pow(scale, 1/left_step)
        # if i == len(layers) - 1:
        #     t_factor = torch.empty(10).fill_(scale)
        #     t_factor = _reshape_tensor(weight.dim(), t_factor).to(device)
        #     weight[:10] = weight[:10].mul(t_factor)
        # else:
        #     t_factor = torch.empty(ex_index[i]).fill_(scale)
        #     t_factor = _reshape_tensor(weight.dim(), t_factor).to(device)
        #     weight[:ex_index[i]] = weight[:ex_index[i]].mul(t_factor)
    print('scale list is {}'.format(scale_list))
    return scale_list


def weight_decay_select(model, ex_index, lambda_list):
    layers = find_modules_short(model)
    scale_list = []
    for i, layer in enumerate(layers):
        weight = layer.weight.data.to(device)
        scale = lambda_list[i]
        scale_list.append(1 / scale)
        t_factor = torch.empty(ex_index[i]).fill_(scale)
        t_factor = _reshape_tensor(weight.dim(), t_factor).to(device)
        weight[:ex_index[i]] = weight[:ex_index[i]].mul(t_factor)
    return scale_list


def _reshape_tensor(dim, t_list, mode='out'):
    d = {2: {'out': (-1, 1), 'in': (1, -1)}, 4: {'out': (-1, 1, 1, 1), 'in': (1, -1, 1, 1)}}
    if dim > 1:
        return t_list.reshape(d[dim][mode])
    else:
        return t_list
