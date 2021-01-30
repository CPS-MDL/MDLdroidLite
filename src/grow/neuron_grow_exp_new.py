import torch
import torch.nn as nn
from model.CNN import LeNet5_GROW1, LeNet5_GROW_BN, LeNet5_GROW2, LeNet5_GROW_P
from main import generate_data_loader
from grow.weight_decay import weight_decay
import numpy as np
from plot.plot_weight_hist import gen_hist_from_dic, save_weight_to_dic
from torch.optim import Adam
# from torch.optim import AdamW
from optmizer.ours_adam import AdamW
from grow.gate_hook import register_hook, remove_hook, record_gate, save_gate_dic_to_json, gradient_lambda_decay, \
    shape_hook_tensor_auto_converge, remove_gate_max
from grow.channel_wise_lr import init_channel_lr, record_lr, save_lr_dic_to_json, auto_decay_lr_exist
from grow.activation_hook import register_acv_hook, retrun_acv_dic, remove_acv_hook, save_acv
import time
import math
import os
from utils import write_log, AverageMeter, accuracy, test, save_model, dir_path, count_parameters, calculate_l1, \
    calculate_l2, calculate_s_score, calculate_s_score_new, calculate_weight, calculate_norm, \
    calculate_sparsity, calculate_cosine_similarity, str_to_float, Timer, calculate_vg_score_new, \
    fill_zeros_to_diff, target_scale_calculator
from regularization.regularizer import L2Regularizer
from sklearn.metrics.pairwise import cosine_similarity
import json
from torch.optim import lr_scheduler

from grow.neuron_grow_lr import lr_size_each_layer

from ri_control.recorder import Recorder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# copy baseline
random_num = []
# ours approach
cosine_rank = []
# rank baseline
importance_rank = []
# activation rank
activation_rank = []
# cumulative rank
cumulative_rank = []
# existing_score
s_score = {0: [], 1: [], 2: [], 3: []}
s_score_in = {0: [], 1: [], 2: [], 3: []}

# existing_index
existing_index = []
existing_index_in = []

# batch_time
batch_time_list = []

# global_start
start_time = 0

# rank_scale
rank_scale = 0

# rank_count
rank_count = 0

# mode
mode_name = 'avg'
dic = {}
# cumuative approach
meter_list = [AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()]
divide_seed = 2
divide_num = 2
res_dir = None
dic_score = {}
# dic_change = {0: {'out': []}, 1: {'out': [], 'og': [], 'in': []}, 2: {'out': [], 'og': [], 'in': []}, 3: {'in': []}}
dic_change = {}
dic_exist = {0: {'out': [], 'layer': 0}, 1: {'out': [], 'in': [], 'layer': 0},
             2: {'out': [], 'in': [], 'layer': 0}, 3: {'in': [], 'layer': 0}}
hook_dic = {}
cumulative_layer_index = 0
optimizer_mode = 'adam'
loss_l2_R = False
is_lr = True
is_gate = True
save_image = True
is_grow = False

# shape
old_shape = []
batch_test_acc = []

lambda_list = []


# is_cumulative = False

def reset_dic():
    global dic_change
    dic_change = {0: {'out': []}, 1: {'out': [], 'og': [], 'in': []}, 2: {'out': [], 'og': [], 'in': []}, 3: {'in': []}}


def set_dic(epoch):
    global dic_change
    dic_change = 0
    pass


def insert_rank(rank):
    if not hasattr(insert_rank, "list"):
        insert_rank.list = []
    if len(insert_rank.list) == 3:
        insert_rank.list = []
    insert_rank.list.append(rank)


def rank_meter(layer_index, ascending=False):
    meter = meter_list[layer_index]
    beta = 0.999
    # g * cumulative + (1-g) current
    g = beta ** meter.count
    # summary_val = g * meter.avg + (1 - g) * meter.val
    summary_val = (1 - g) * meter.avg + g * meter.val
    rank = get_rank(summary_val, ascending=ascending)
    return rank


# return layers Gradient according to index, and if index is not given return 3 layers Gradient
def get_Ltwo(model):
    dic = {}
    for index, module in enumerate(model.layers):
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            grad_tensor = module.weight.grad.data.cpu().numpy()
            result = np.sqrt(np.sum(grad_tensor ** 2))
            if index == 0:
                dic[0] = result
            if index == 3:
                dic[3] = result
            if index == 7:
                dic[7] = result
        return dic


def get_filter_rank_old(old_layer, is_first=True, ascending=False):
    grad_tensor = old_layer.weight.grad.data.cpu().numpy()
    if isinstance(old_layer, torch.nn.Conv2d):
        if is_first:
            result = np.sqrt(np.sum(grad_tensor ** 2, axis=(1, 2, 3)))
        else:
            result = np.sqrt(np.sum(grad_tensor ** 2, axis=(0, 2, 3)))
    elif isinstance(old_layer, torch.nn.Linear):
        if is_first:
            result = np.sqrt(np.sum(grad_tensor ** 2, axis=1))
        else:
            result = np.sqrt(np.sum(grad_tensor ** 2, axis=0))
    rank = get_rank(result, ascending=ascending)
    # order = result.argsort()
    # rank = order.argsort()
    # rank = rank.tolist()
    return rank


def get_filter_rank(old_layer, is_first=True, ascending=False, is_vg=False):
    global rank_count
    global optimizer
    weights_tensor = old_layer.weight.data.cpu().numpy()
    if is_vg:
        grad_tensor = optimizer.vg[rank_count % 3].numpy()
        grad_tensor = fill_zeros_to_diff(weights_tensor, grad_tensor)
        rank_count += 1
    else:
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


def get_rank(result, ascending=False):
    result = np.asarray(result)
    if not ascending:
        result = result * -1
    order = result.argsort()
    # rank = order.argsort()
    rank = order.tolist()
    return rank


def find_bn_layers(model):
    layers = {}
    for index_str, module in model.features._modules.items():
        if isinstance(module, torch.nn.BatchNorm2d):
            layers[int(index_str)] = module
    return layers


def find_layers(model):
    layers = []
    for index_str, module in model.features._modules.items():
        if isinstance(module, torch.nn.Conv2d):
            layers.append(module)

    for index_str, module in model.classifier._modules.items():
        if isinstance(module, torch.nn.Linear):
            layers.append(module)
    return layers


def model_size(layers):
    size = []
    for module in layers:
        size.append(module.bias.data.shape[0])
    return size


def record_score(model, batch, epoch, optimizer=None):
    global dic_score
    global dic_change
    global dic_exist
    global existing_index
    global s_score
    global s_score_in
    global optimizer_mode
    global recorder
    # layers = find_layers(model)
    # size = model_size(layers)
    # criterion = nn.CrossEntropyLoss()
    # feature_out, output = model(inputs)
    # loss = criterion(output, target)
    # loss.backward()
    acv_dic = retrun_acv_dic()
    layers = find_layers(model)
    if batch == 0:
        dic_score[epoch] = {}
    dic_score[epoch][batch] = {}
    # save values in to dic
    for i, layer in enumerate(layers):
        dic_score[epoch][batch][i] = {'W': {}, 'L1': {}, 'L2': {}, 'S': {}, 'VG': {}, 'ACV': {}, 'Sparsity': {},
                                      'Delta_S': [], 'Cosine': [], 'V': 0, 'M': 0, 'G': 0, 'Grad': 0}
        # dic_score[epoch][batch][i] = {}
        # dic_score[epoch][batch][i]['W'] = {}
        # dic_score[epoch][batch][i]['L1'] = {}
        # dic_score[epoch][batch][i]['L2'] = {}
        # dic_score[epoch][batch][i]['S'] = {}
        # for key in dic_change[i]:
        #     if not dic_change[i][key]:  # list is empty
        #         continue
        #     m, s = calculate_weight(layer, mode=key, index_list=dic_change[i][key])
        #     channel_s, _ = calculate_s_score_new(layer, mode=key, index_list=dic_change[i][key])
        #     check_dic_key(dic_score[epoch][batch][i]['W'], key)
        #     check_dic_key(dic_score[epoch][batch][i]['S'], key)
        #     dic_score[epoch][batch][i]['W'][key]['mean'] = str(m)
        #     dic_score[epoch][batch][i]['W'][key]['std'] = str(s)
        #     dic_score[epoch][batch][i]['S'][key] = str(channel_s)

        # V, M, G score
        if optimizer is not None and optimizer_mode == 'AdamW' and optimizer.vs:
            dic_score[epoch][batch][i]['V'] = optimizer.vs[i]
            dic_score[epoch][batch][i]['M'] = optimizer.ms[i]
            dic_score[epoch][batch][i]['G'] = optimizer.gs[i]
            dic_score[epoch][batch][i]['Grad'] = optimizer.grads[i]

        # ACV
        if i < len(layers) - 1:
            dic_score[epoch][batch][i]['ACV']['L1'] = str(acv_dic[str(i)]['L1'])
            dic_score[epoch][batch][i]['ACV']['L2'] = str(acv_dic[str(i)]['L2'])

        # Delta S
        if recorder.get_pre_score() and recorder.get_score() and False:
            pre = recorder.get_pre_score()[i]
            cur = recorder.get_score()[i]
            if len(cur) == len(pre):
                delta_s = np.asarray(cur) - np.asarray(pre)
                dic_score[epoch][batch][i]['Delta_S'] = str(delta_s.tolist())

        for key in dic_exist[i]:
            l1, s = calculate_weight(layer, mode=key)
            s_result = calculate_s_score_new(layer, mode=key)
            l1_result = calculate_norm(layer, mode=key, order=1)
            l2_result = calculate_norm(layer, mode=key, order=2)
            sparsity = calculate_sparsity(layer, mode=key)
            dic_score[epoch][batch][i]['W'][key] = {}
            dic_score[epoch][batch][i]['W'][key]['L1'] = str(l1)
            dic_score[epoch][batch][i]['W'][key]['std'] = str(s)
            dic_score[epoch][batch][i]['L1'][key] = str(l1_result)
            dic_score[epoch][batch][i]['L2'][key] = str(l2_result)
            dic_score[epoch][batch][i]['S'][key] = str(s_result)
            dic_score[epoch][batch][i]['Sparsity'][key] = str(sparsity)

            if optimizer_mode == 'AdamW' and optimizer.vg:
                vg_result = calculate_vg_score_new(layer, optimizer.vg[i], mode=key)
                dic_score[epoch][batch][i]['VG'][key] = str(vg_result)
            else:
                dic_score[epoch][batch][i]['VG'][key] = []

    # existing_index_out
    if len(existing_index) > 0:
        for i, layer in enumerate(layers):
            # cosine
            if i < 3:
                s_exist = dic_score[epoch][batch][i]['S']['out']
                s_exist = str_to_float(s_exist)
                s_exist = s_exist[:existing_index[i]]
                s_rank = get_rank(s_exist)
                cos = calculate_cosine_similarity(layer, exist_index=existing_index[i], rank=s_rank)
                dic_score[epoch][batch][i]['Cosine'] = cos

            # existing_index_out
            s_result = calculate_s_score_new(layer, mode='out')
            s_score[i] = s_result

            # existing_index_in
            s_result_in = calculate_s_score_new(layer, mode='in')
            s_score_in[i] = s_result_in


def caluclulate_activation(model, inputs, next_layer_index, epoch, channel, print_weights=False):
    global dic
    layers = find_layers(model)
    size = model_size(layers)
    act_model = LeNet5_GROW1(in_channel=1, out_channel=10, out1=size[0], out2=size[1], fc1=size[2])
    act_model.conv1 = layers[0]
    act_model.conv2 = layers[1]
    act_model.fc1 = layers[2]
    act_model.fc2 = layers[3]
    ac_tuple = act_model(inputs)
    ac_list = []
    ac_mean = []
    for i, ac in enumerate(ac_tuple):
        ac_list.append(ac.data.cpu().numpy())
        if ac_list[i].ndim == 4:
            ac_mean.append(np.mean(ac_list[i], axis=(0, 2, 3)))
        elif ac_list[i].ndim == 2:
            ac_mean.append(np.mean(ac_list[i], axis=(0)))
    if next_layer_index == 0:
        return ac_mean

    for i, ac in enumerate(ac_list):
        dic[epoch][channel][i] = {}
        dic[epoch][channel][i]['A'] = {}
        if i < 2:
            mean = np.mean(ac, axis=(0, 2, 3)).astype(np.float16).tolist()
            std = np.std(ac, axis=(0, 2, 3)).astype(np.float16).tolist()
        else:
            mean = np.mean(ac, axis=0).astype(np.float16).tolist()
            std = np.std(ac, axis=0).astype(np.float16).tolist()
        # np round
        mean = np.around(mean, 5).tolist()
        std = np.around(std, 5).tolist()
        # str_activation = '\nLayer-{} activation:\n\tmean: {}({}), \n\tstd:{}({})\n'.format(i, mean, np.mean(mean), std,
        #                                                                                    np.mean(std))
        # print(str_activation)
        # write_log(str_activation, to_log[i])
        dic[epoch][channel][i]['A']['mean'] = str(mean)
        dic[epoch][channel][i]['A']['std'] = str(std)
        dic[epoch][channel][i]['A']['M_mean'] = str(np.mean(mean))
        dic[epoch][channel][i]['A']['M_std'] = str(np.mean(std))

    # if print_weights:
    #     for i, layer in enumerate(layers):
    #         weight = layer.weight.data.cpu().numpy()
    #         mean_list = []
    #         std_list = []
    #         for channel_weight in weight:
    #             mean_list.append(np.mean(channel_weight))
    #             std_list.append(np.std(channel_weight))
    #         mean_list = np.round(mean_list, 5)
    #         std_list = np.round(std_list, 5)
    #         str_weight = '\nLayer-{} weight:\n\tmean: {}({}), \n\tstd:{}({})\n'.format(i, mean_list, np.mean(mean_list),
    #                                                                                    std_list,
    #                                                                                    np.mean(std_list))
    #         print(str_weight)
    #         write_log(str_weight, to_log[i])

    return ac_mean[next_layer_index]


def put_cumulative_score(model, mode='L1'):
    global meter_list
    layers = find_layers(model)
    for meter, module in zip(meter_list, layers):
        if mode == 'L1':
            channel_result, _ = calculate_l1(module)
        if mode == 'L2':
            channel_result, _ = calculate_l2(module)
        if mode == 'S':
            channel_result, _ = calculate_s_score(module)
        meter.update(np.asarray(channel_result))


def reset_cumulative_score():
    global meter_list
    for meter in meter_list:
        meter.reset()


def calculate_score(model, inputs, target, epoch, channel, print_weights=True):
    global dic
    criterion = nn.CrossEntropyLoss()
    feature_out, output = model(inputs)
    loss = criterion(output, target)
    loss.backward()
    layers = find_layers(model)
    channel = (channel + '.')[:-1]
    if print_weights:
        for i, layer in enumerate(layers):
            dic[epoch][channel][i]['W'] = {}
            dic[epoch][channel][i]['L1'] = {}
            dic[epoch][channel][i]['L2'] = {}
            dic[epoch][channel][i]['S'] = {}
            weight = layer.weight.data.cpu().numpy()
            mean_list = []
            std_list = []
            for channel_weight in weight:
                mean_list.append(np.mean(channel_weight))
                std_list.append(np.std(channel_weight))
            mean_list = np.around(mean_list, 5).tolist()
            std_list = np.around(std_list, 5).tolist()
            # str_weight = '\nLayer-{} weight:\n\tmean: {}({}), \n\tstd:{}({})\n'.format(i, mean_list, np.mean(mean_list),
            #                                                                            std_list, np.mean(std_list))
            # print(str_weight)
            # write_log(str_weight, to_log[i])
            # save to dic
            dic[epoch][channel][i]['W']['mean'] = str(mean_list)
            dic[epoch][channel][i]['W']['std'] = str(std_list)
            dic[epoch][channel][i]['W']['M_mean'] = str(np.mean(mean_list))
            dic[epoch][channel][i]['W']['M_std'] = str(np.mean(std_list))
            l1_channel, l1_layer = calculate_l1(layer)
            dic[epoch][channel][i]['L1']['channel'] = str(l1_channel)
            dic[epoch][channel][i]['L1']['layer'] = str(l1_layer)
            l2_channel, l2_layer = calculate_l2(layer)
            dic[epoch][channel][i]['L2']['channel'] = str(l2_channel)
            dic[epoch][channel][i]['L2']['layer'] = str(l2_layer)
            score_channel, score_layer = calculate_s_score(layer)
            dic[epoch][channel][i]['S']['channel'] = str(score_channel)
            dic[epoch][channel][i]['S']['layer'] = str(score_layer)


def caluclulate_activation1(model, inputs, to_log, epoch, channel, print_weights=False):
    layers = find_layers(model)
    bn_layers = find_bn_layers(model)
    size = model_size(layers)
    act_model = LeNet5_GROW2(in_channel=1, out_channel=10, out1=size[0], out2=size[1], fc1=size[2])
    act_model.conv1 = layers[0]
    act_model.bn1 = bn_layers[1]
    act_model.conv2 = layers[1]
    act_model.bn2 = bn_layers[5]
    act_model.fc1 = layers[2]
    act_model.fc2 = layers[3]
    act_model = act_model.to(device)
    ac1, ac2, ac3, ac4, out = act_model(inputs)

    for i, ac in enumerate(
            [ac1.data.cpu().numpy(), ac2.data.cpu().numpy(), ac3.data.cpu().numpy(), ac4.data.cpu().numpy()]):
        dic[epoch][channel][i] = {}
        dic[epoch][channel][i]['A'] = {}
        if i < 2:
            mean = np.mean(ac, axis=(0, 2, 3)).astype(np.float16).tolist()
            std = np.std(ac, axis=(0, 2, 3)).astype(np.float16).tolist()
        else:
            mean = np.mean(ac, axis=0).astype(np.float16).tolist()
            std = np.std(ac, axis=0).astype(np.float16).tolist()
        # np round
        mean = np.around(mean, 5).tolist()
        std = np.around(std, 5).tolist()
        # str_activation = '\nLayer-{} activation:\n\tmean: {}({}), \n\tstd:{}({})\n'.format(i, mean, np.mean(mean), std,
        #                                                                                    np.mean(std))
        # print(str_activation)
        # write_log(str_activation, to_log[i])
        dic[epoch][channel][i]['A']['mean'] = str(mean)
        dic[epoch][channel][i]['A']['std'] = str(std)
        dic[epoch][channel][i]['A']['M_mean'] = str(np.mean(mean))
        dic[epoch][channel][i]['A']['M_std'] = str(np.mean(std))

    # for i, ac in enumerate(
    #         [ac1.data.cpu().numpy(), ac2.data.cpu().numpy(), ac3.data.cpu().numpy(), ac4.data.cpu().numpy()]):
    #     if i < 2:
    #         mean = np.mean(ac, axis=(0, 2, 3)).tolist()
    #         std = np.std(ac, axis=(0, 2, 3)).tolist()
    #     else:
    #         mean = np.mean(ac, axis=0).tolist()
    #         std = np.std(ac, axis=0).tolist()
    #     str_activation = 'Layer-{} activation:\n\tmean: {}({}), std:{}({})'.format(i, mean, np.mean(mean), std,
    #                                                                                np.mean(std))
    #     write_log(str_activation, to_log[i])


def weight_adaption_avg(old_layer, incremental_num, is_first=True):
    old_weights = old_layer.weight.data.cpu().numpy()
    new_bias_needed = None
    if isinstance(old_layer, torch.nn.Conv2d):
        old_c_out, old_c_in, k_h, k_w = old_weights.shape
        old_bias = old_layer.bias.data.cpu().numpy()
        if is_first:
            avg_weight = np.mean(old_weights, axis=0)
            shape = avg_weight.shape
            avg_weight = np.reshape(avg_weight, (1, shape[0], shape[1], shape[2]))
            size_needed = np.random.uniform(np.min(old_weights) / 10, np.max(old_weights) / 10,
                                            (incremental_num, old_c_in, k_h, k_w))
            new_bias_needed = np.random.uniform(np.min(old_bias) / 10, np.max(old_bias) / 10,
                                                incremental_num)
        else:
            avg_weight = np.mean(old_weights, axis=1)
            shape = avg_weight.shape
            avg_weight = np.reshape(avg_weight, (shape[0], 1, shape[1], shape[2]))
            size_needed = np.random.uniform(np.min(old_weights) / 10, np.max(old_weights) / 10,
                                            (old_c_out, incremental_num, k_h, k_w))

    elif isinstance(old_layer, torch.nn.Linear):
        old_c_out, old_c_in = old_weights.shape
        old_bias = old_layer.bias.data.cpu().numpy()
        if is_first:
            avg_weight = np.mean(old_weights, axis=0)
            shape = avg_weight.shape
            avg_weight = np.reshape(avg_weight, (1, shape[0]))
            size_needed = np.random.uniform(np.min(old_weights) / 10, np.max(old_weights) / 10,
                                            (incremental_num, old_c_in))
            new_bias_needed = np.random.uniform(np.min(old_bias) / 10, np.max(old_bias) / 10,
                                                incremental_num)
        else:
            avg_weight = np.mean(old_weights, axis=1)
            shape = avg_weight.shape
            avg_weight = np.reshape(avg_weight, (shape[0], 1))
            size_needed = np.random.uniform(np.min(old_weights) / 10, np.max(old_weights) / 10,
                                            (old_c_out, incremental_num))

    base_weight = multi_weights(avg_weight, incremental_num, is_first)
    base_weight = size_needed + base_weight
    return base_weight, new_bias_needed


def replace_bn_layers(model, index, incremental_num):
    dic_layers = find_bn_layers(model)
    current_layer = dic_layers[index]
    new_num_features = current_layer.num_features + incremental_num
    current_bias = current_layer.bias.data.cpu().numpy()
    current_weight = current_layer.weight.data.cpu().numpy()
    new_layer = nn.BatchNorm2d(
        num_features=new_num_features,
        eps=current_layer.eps,
        momentum=current_layer.momentum,
        affine=current_layer.affine,
        track_running_stats=current_layer.track_running_stats
    )
    new_weight = np.ones((new_num_features))
    new_bias = np.zeros((new_num_features))
    new_weight[0:current_layer.num_features] = current_weight
    new_bias[0:current_layer.num_features] = current_bias
    new_layer.bias.data = torch.from_numpy(new_bias.astype(np.float32))
    new_layer.weight.data = torch.from_numpy(new_weight.astype(np.float32))
    new_layer.bias.data = new_layer.bias.data.to(device)
    new_layer.weight.data = new_layer.weight.data.to(device)
    model.features[index] = new_layer
    return model


def weight_adaption_random_select(old_layer, incremental_num, is_first=True):
    old_weights = old_layer.weight.data.cpu().numpy()
    new_bias_needed = None
    if isinstance(old_layer, torch.nn.Conv2d):
        old_c_out, old_c_in, k_h, k_w = old_weights.shape
        old_bias = old_layer.bias.data.cpu().numpy()
        if is_first:
            weights = np.zeros((incremental_num, old_c_in, k_h, k_w))
            random_list = np.random.choice(old_c_out, incremental_num)
            for i, r in enumerate(random_list):
                weights[i] = old_weights[r]
            size_needed = np.random.uniform(np.min(old_weights) / 10, np.max(old_weights) / 10,
                                            (incremental_num, old_c_in, k_h, k_w))
            new_bias_needed = np.random.uniform(np.min(old_bias) / 10, np.max(old_bias) / 10,
                                                incremental_num)
        else:
            weights = np.zeros((old_c_out, incremental_num, k_h, k_w))
            random_list = np.random.choice(old_c_in, incremental_num)
            for i, r in enumerate(random_list):
                weights[:, i, :, :] = old_weights[:, r, :, :]
            size_needed = np.random.uniform(np.min(old_weights) / 10, np.max(old_weights) / 10,
                                            (old_c_out, incremental_num, k_h, k_w))

    elif isinstance(old_layer, torch.nn.Linear):
        old_c_out, old_c_in = old_weights.shape
        old_bias = old_layer.bias.data.cpu().numpy()
        if is_first:
            weights = np.zeros((incremental_num, old_c_in))
            random_list = np.random.choice(old_c_out, incremental_num)
            for i, r in enumerate(random_list):
                weights[i] = old_weights[r]
            size_needed = np.random.uniform(np.min(old_weights) / 10, np.max(old_weights) / 10,
                                            (incremental_num, old_c_in))
            new_bias_needed = np.random.uniform(np.min(old_bias) / 10, np.max(old_bias) / 10,
                                                incremental_num)
        else:
            weights = np.zeros((old_c_out, incremental_num))
            random_list = np.random.choice(old_c_in, incremental_num)
            for i, r in enumerate(random_list):
                weights[:, i] = old_weights[:, r]
            size_needed = np.random.uniform(np.min(old_weights) / 10, np.max(old_weights) / 10,
                                            (old_c_out, incremental_num))

    base_weight = size_needed + weights
    return base_weight, new_bias_needed


def weight_adaption_copy_one(old_layer, incremental_num, is_first=True, is_multi=False, multi=16):
    global divide_num
    global random_num
    old_weights = old_layer.weight.data.cpu().numpy()

    # if divide_num == 'non-scale':
    #     numerator = 1
    # else:
    #     numerator = incremental_num if divide_num != 2 else divide_num

    numerator = incremental_num
    new_bias_needed = None

    if isinstance(old_layer, torch.nn.Conv2d):
        old_c_out, old_c_in, k_h, k_w = old_weights.shape
        if is_first:
            weights = np.zeros((incremental_num, old_c_in, k_h, k_w))
            random = np.random.choice(old_c_out, 1)[0]  # incremental_num to 1
            for i in range(incremental_num):
                weights[i] = old_weights[random]
            size_needed = np.random.uniform(np.min(old_weights) / 10, np.max(old_weights) / 10,
                                            (incremental_num, old_c_in, k_h, k_w))
            new_bias_needed = np.random.uniform(np.min(old_weights) / 10, np.max(old_weights) / 10,
                                                incremental_num)
            random_num = random
        else:
            weights = np.zeros((old_c_out, incremental_num, k_h, k_w))
            random = random_num
            for i in range(incremental_num):
                weights[:, i, :, :] = old_weights[:, random, :, :] / (numerator + 1)
            size_needed = np.random.uniform(np.min(old_weights) / 10, np.max(old_weights) / 10,
                                            (old_c_out, incremental_num, k_h, k_w))

    elif isinstance(old_layer, torch.nn.Linear):
        old_c_out, old_c_in = old_weights.shape
        if is_first:
            weights = np.zeros((incremental_num, old_c_in))
            random = np.random.choice(old_c_out, 1)[0]  # incremental_num to 1
            for i in range(incremental_num):
                weights[i] = old_weights[random]
            size_needed = np.random.uniform(np.min(old_weights) / 10, np.max(old_weights) / 10,
                                            (incremental_num, old_c_in))
            new_bias_needed = np.random.uniform(np.min(old_weights) / 10, np.max(old_weights) / 10,
                                                incremental_num)
            random_num = random
        else:
            if not is_multi:
                multi = 1
            weights = np.zeros((old_c_out, incremental_num))
            random = random_num
            for i in range(incremental_num // multi):
                start = random * multi
                end = start + multi
                start_i = i * multi
                end_i = start_i + multi
                weights[:, start_i:end_i] = old_weights[:, start:end] / ((numerator + multi) // multi)
            size_needed = np.random.uniform(np.min(old_weights) / 10, np.max(old_weights) / 10,
                                            (old_c_out, incremental_num))
    base_weight = size_needed + weights
    return base_weight, new_bias_needed


def weight_adaption_copy_n(old_layer, incremental_num, is_first=True, is_multi=False, multi=16):
    global divide_num
    global random_num
    old_weights = old_layer.weight.data.cpu().numpy()
    new_bias_needed = None

    # if divide_num == 'non-scale':
    #     numerator = 1
    # else:
    #     numerator = incremental_num if divide_num != 2 else divide_num

    numerator = 2

    if isinstance(old_layer, torch.nn.Conv2d):
        old_c_out, old_c_in, k_h, k_w = old_weights.shape
        if is_first:
            weights = np.zeros((incremental_num, old_c_in, k_h, k_w))
            random_list = np.random.choice(old_c_out, incremental_num)  # incremental_num to 1
            for i, r in enumerate(random_list):
                weights[i] = old_weights[r]
            size_needed = np.random.uniform(-0.1, 0.1, (incremental_num, old_c_in, k_h, k_w))
            new_bias_needed = np.random.uniform(-0.1, 0.1, incremental_num)
            random_num = random_list
        else:
            weights = np.zeros((old_c_out, incremental_num, k_h, k_w))
            random_list = random_num
            for i, r in enumerate(random_list):
                weights[:, i] = old_weights[:, r] / numerator
            size_needed = np.random.uniform(-0.1, 0.1, (old_c_out, incremental_num, k_h, k_w))

    elif isinstance(old_layer, torch.nn.Linear):
        old_c_out, old_c_in = old_weights.shape
        if is_first:
            weights = np.zeros((incremental_num, old_c_in))
            random_list = np.random.choice(old_c_out, incremental_num)
            for i, r in enumerate(random_list):
                weights[i] = old_weights[r]
            new_bias_needed = np.random.uniform(-0.1, 0.1, incremental_num)
            random_num = random_list
            size_needed = np.random.uniform(-0.1, 0.1, (incremental_num, old_c_in))
        else:
            if not is_multi:
                multi = 1
            weights = np.zeros((old_c_out, incremental_num))
            random_list = random_num
            for i, r in enumerate(random_list):
                start = r * multi
                end = start + multi
                start_i = i * multi
                end_i = start_i + multi
                weights[:, start_i:end_i] = old_weights[:, start:end] / numerator
            size_needed = np.random.uniform(-0.1, 0.1, (old_c_out, incremental_num))

    base_weight = size_needed + weights
    return base_weight, new_bias_needed


def weight_adaption_rank(old_layer, incremental_num, is_first=True):
    old_weights = old_layer.weight.data.cpu().numpy()
    new_bias_needed = None
    if isinstance(old_layer, torch.nn.Conv2d):
        old_c_out, old_c_in, k_h, k_w = old_weights.shape
        old_bias = old_layer.bias.data.cpu().numpy()
        if is_first:
            rank = get_filter_rank(old_layer)
            weights = np.reshape(old_weights[rank[0], :, :, :], (1, old_c_in, k_h, k_w))
            for i in range(1, incremental_num):
                index = rank[0]
                weights = np.vstack((weights, np.reshape(old_weights[index, :, :, :], (1, old_c_in, k_h, k_w))))
            size_needed = np.random.uniform(np.min(old_weights) / 10, np.max(old_weights) / 10,
                                            (incremental_num, old_c_in, k_h, k_w))
            new_bias_needed = np.random.uniform(np.min(old_bias) / 10, np.max(old_bias) / 10,
                                                incremental_num)
        else:
            rank = get_filter_rank(old_layer, is_first=False)
            weights = np.reshape(old_weights[:, rank[0], :, :], (old_c_out, 1, k_h, k_w))
            for i in range(1, incremental_num):
                index = rank[0]
                weights = np.hstack((weights, np.reshape(old_weights[:, index, :, :], (old_c_out, 1, k_h, k_w))))
            size_needed = np.random.uniform(np.min(old_weights) / 10, np.max(old_weights) / 10,
                                            (old_c_out, incremental_num, k_h, k_w))

    elif isinstance(old_layer, torch.nn.Linear):
        old_c_out, old_c_in = old_weights.shape
        old_bias = old_layer.bias.data.cpu().numpy()
        if is_first:
            rank = get_filter_rank(old_layer)
            weights = old_weights[rank[0], :]
            for i in range(1, incremental_num):
                index = rank[0]
                weights = np.vstack((weights, old_weights[index, :]))
            size_needed = np.random.uniform(np.min(old_weights) / 10, np.max(old_weights) / 10,
                                            (incremental_num, old_c_in))
            new_bias_needed = np.random.uniform(np.min(old_bias) / 10, np.max(old_bias) / 10,
                                                incremental_num)
        else:
            rank = get_filter_rank(old_layer, is_first=False)
            weights = np.reshape(old_weights[:, rank[0]], (old_c_out, 1))
            for i in range(1, incremental_num):
                index = rank[0]
                weights = np.hstack((weights, np.reshape(old_weights[:, index], (old_c_out, 1))))
            size_needed = np.random.uniform(np.min(old_weights) / 10, np.max(old_weights) / 10,
                                            (old_c_out, incremental_num))
    base_weight = size_needed + weights
    # base_weight = weights
    return base_weight, new_bias_needed


def weight_adaption_rank_connect(old_layer, incremental_num, is_first=True, is_multi=False, m_seed=16):
    global divide_num
    global importance_rank
    old_weights = old_layer.weight.data.cpu().numpy()
    new_bias_needed = None

    if divide_num == 'non-scale':
        numerator = 1
    else:
        numerator = incremental_num if divide_num != 2 else divide_num

    if isinstance(old_layer, torch.nn.Conv2d):
        old_c_out, old_c_in, k_h, k_w = old_weights.shape
        if is_first:
            weights = np.zeros((incremental_num, old_c_in, k_h, k_w))
            rank = get_filter_rank(old_layer)
            importance_rank = rank
            for i in range(0, incremental_num):
                index = rank[i]
                weights[i] = old_weights[index]
            size_needed = np.random.uniform(-0.1, 0.1, (incremental_num, old_c_in, k_h, k_w))
            new_bias_needed = np.random.uniform(-0.1, 0.1, incremental_num)
        else:
            rank = importance_rank
            weights = np.zeros((old_c_out, incremental_num, k_h, k_w))
            for i in range(1, incremental_num):
                index = rank[i]
                weights[:, i] = old_weights[:, index] / numerator
            size_needed = np.random.uniform(-0.1, 0.1, (old_c_out, incremental_num, k_h, k_w))

    elif isinstance(old_layer, torch.nn.Linear):
        old_c_out, old_c_in = old_weights.shape
        if is_first:
            weights = np.zeros((incremental_num, old_c_in))
            rank = get_filter_rank(old_layer)
            importance_rank = rank
            for i in range(0, incremental_num):
                index = rank[i]
                weights[i] = old_weights[index]
            size_needed = np.random.uniform(-0.1, 0.1, (incremental_num, old_c_in))
            new_bias_needed = np.random.uniform(-0.1, 0.1, incremental_num)
        else:
            rank = importance_rank
            weights = np.zeros((old_c_out, incremental_num))
            if is_multi:
                multiplier = m_seed
            else:
                multiplier = 1
            for i in range(0, incremental_num // multiplier):
                index = rank[i]
                start = index * multiplier
                end = start + multiplier
                start_i = i * multiplier
                end_i = start_i + multiplier
                weights[:, start_i:end_i] = old_weights[:, start:end] / (numerator * multiplier)
            size_needed = np.random.uniform(-0.1, 0.1, (old_c_out, incremental_num))
    base_weight = size_needed + weights
    return base_weight, new_bias_needed


def weight_adaption_rank_baseline(old_layer, incremental_num, is_first=True, is_multi=False, ascending=False):
    global divide_num
    global importance_rank
    old_weights = old_layer.weight.data.cpu().numpy()
    new_bias_needed = None
    # if divide_num == 'non-scale':
    #     numerator = 1
    # else:
    #     numerator = incremental_num if divide_num != 2 else divide_num
    numerator = 2

    if isinstance(old_layer, torch.nn.Conv2d):
        old_c_out, old_c_in, k_h, k_w = old_weights.shape
        if is_first:
            weights = np.zeros((incremental_num, old_c_in, k_h, k_w))
            rank = get_filter_rank(old_layer, ascending=ascending)
            importance_rank = rank
            for i in range(0, incremental_num):
                index = rank[i]
                weights[i] = old_weights[index] / numerator
            size_needed = np.random.uniform(-0.1, 0.1, (incremental_num, old_c_in, k_h, k_w))
            new_bias_needed = np.random.uniform(-0.1, 0.1, incremental_num)
        else:
            rank = importance_rank
            weights = np.zeros((old_c_out, incremental_num, k_h, k_w))
            for i in range(1, incremental_num):
                index = rank[i]
                weights[:, i] = old_weights[:, index] / numerator
            size_needed = np.random.uniform(-0.1, 0.1, (old_c_out, incremental_num, k_h, k_w))

    elif isinstance(old_layer, torch.nn.Linear):
        old_c_out, old_c_in = old_weights.shape
        if is_first:
            weights = np.zeros((incremental_num, old_c_in))
            rank = get_filter_rank(old_layer, ascending=ascending)
            importance_rank = rank
            for i in range(0, incremental_num):
                index = rank[i]
                weights[i] = old_weights[index] / numerator
            size_needed = np.random.uniform(-0.1, 0.1, (incremental_num, old_c_in))
            new_bias_needed = np.random.uniform(-0.1, 0.1, incremental_num)
        else:
            rank = importance_rank
            weights = np.zeros((old_c_out, incremental_num))
            if is_multi:
                multiplier = 16
            else:
                multiplier = 1
            for i in range(0, incremental_num // multiplier):
                index = rank[i]
                start = index * multiplier
                end = start + multiplier
                start_i = i * multiplier
                end_i = start_i + multiplier
                weights[:, start_i:end_i] = old_weights[:, start:end] / numerator
            size_needed = np.random.uniform(-0.1, 0.1, (old_c_out, incremental_num))
    base_weight = size_needed + weights
    return base_weight, new_bias_needed


def weight_adaption_rank_baseline_scale(old_layer, incremental_num, is_first=True, is_multi=False, ascending=False):
    global divide_num
    global importance_rank
    global rank_scale
    old_weights = old_layer.weight.data.cpu().numpy()
    new_bias_needed = None
    # if divide_num == 'non-scale':
    #     numerator = 1
    # else:
    #     numerator = incremental_num if divide_num != 2 else divide_num
    # numerator = 2
    old_scale = max(np.abs(np.min(old_weights)), np.abs(np.max(old_weights)))

    if isinstance(old_layer, torch.nn.Conv2d):
        old_c_out, old_c_in, k_h, k_w = old_weights.shape
        if is_first:
            scale_factor = 1 / math.sqrt(old_c_in * k_h * k_w)
            numerator = scale_factor / old_scale
            print('old={:2.5f}| new={:2.5f}| scale={:2.5f}'.format(old_scale, scale_factor, numerator))
            weights = np.zeros((incremental_num, old_c_in, k_h, k_w))
            rank = get_filter_rank(old_layer, ascending=ascending)
            importance_rank = rank
            for i in range(0, incremental_num):
                index = rank[i]
                weights[i] = old_weights[index] * numerator
            # size_needed = np.random.uniform(-0.1, 0.1, (incremental_num, old_c_in, k_h, k_w))
            new_bias_needed = np.random.uniform(-0.1, 0.1, incremental_num)
        else:
            scale_factor = 1 / math.sqrt((old_c_in + incremental_num) * k_h * k_w)
            numerator = scale_factor / old_scale
            print('old={:2.5f}| new={:2.5f}| scale={:2.5f}'.format(old_scale, scale_factor, numerator))
            rank = importance_rank
            weights = np.zeros((old_c_out, incremental_num, k_h, k_w))
            for i in range(1, incremental_num):
                index = rank[i]
                weights[:, i] = old_weights[:, index] * numerator
            # size_needed = np.random.uniform(-0.1, 0.1, (old_c_out, incremental_num, k_h, k_w))

    elif isinstance(old_layer, torch.nn.Linear):
        old_c_out, old_c_in = old_weights.shape
        if is_first:
            scale_factor = 1 / math.sqrt(old_c_in)
            numerator = scale_factor / old_scale
            print('old={:2.5f}| new={:2.5f}| scale={:2.5f}'.format(old_scale, scale_factor, numerator))
            weights = np.zeros((incremental_num, old_c_in))
            rank = get_filter_rank(old_layer, ascending=ascending)
            importance_rank = rank
            for i in range(0, incremental_num):
                index = rank[i]
                weights[i] = old_weights[index] * numerator
            # size_needed = np.random.uniform(-0.1, 0.1, (incremental_num, old_c_in))
            new_bias_needed = np.random.uniform(-0.1, 0.1, incremental_num)
        else:
            scale_factor = 1 / math.sqrt(old_c_in + incremental_num)
            numerator = scale_factor / old_scale
            print('old={:2.5f}| new={:2.5f}| scale={:2.5f}'.format(old_scale, scale_factor, numerator))
            rank = importance_rank
            weights = np.zeros((old_c_out, incremental_num))
            if is_multi:
                multiplier = 16
            else:
                multiplier = 1
            for i in range(0, incremental_num // multiplier):
                index = rank[i]
                start = index * multiplier
                end = start + multiplier
                start_i = i * multiplier
                end_i = start_i + multiplier
                weights[:, start_i:end_i] = old_weights[:, start:end] * numerator
    size_needed = np.random.uniform(np.min(weights) / 10, np.max(weights) / 10, weights.shape)
    base_weight = size_needed + weights
    # rank_scale record
    rank_scale = numerator
    return base_weight, new_bias_needed


def weight_adaption_rank_one(old_layer, incremental_num, is_first=True, is_multi=False):
    global divide_num
    global importance_rank
    old_weights = old_layer.weight.data.cpu().numpy()
    new_bias_needed = None
    # if divide_num == 'non-scale':
    #     numerator = 1
    # else:
    #     numerator = incremental_num if divide_num != 2 else divide_num
    # numerator = incremental_num

    # s_index = 0 if is_first else 1
    # numerator = 1 - (incremental_num / (old_weights.shape[s_index] + incremental_num))
    # numerator /= 2
    numerator = 1 / 2

    if isinstance(old_layer, torch.nn.Conv2d):
        old_c_out, old_c_in, k_h, k_w = old_weights.shape
        if is_first:
            weights = np.zeros((incremental_num, old_c_in, k_h, k_w))
            rank = get_filter_rank(old_layer)
            importance_rank = rank
            for i in range(0, incremental_num):
                index = rank[0]
                weights[i] = old_weights[index] * numerator
            size_needed = np.random.uniform(-0.1, 0.1, (incremental_num, old_c_in, k_h, k_w))
            new_bias_needed = np.random.uniform(-0.1, 0.1, incremental_num)
        else:
            rank = importance_rank
            weights = np.zeros((old_c_out, incremental_num, k_h, k_w))
            for i in range(1, incremental_num):
                index = rank[0]
                weights[:, i] = old_weights[:, index] * numerator
            size_needed = np.random.uniform(-0.1, 0.1, (old_c_out, incremental_num, k_h, k_w))

    elif isinstance(old_layer, torch.nn.Linear):
        old_c_out, old_c_in = old_weights.shape
        if is_first:
            weights = np.zeros((incremental_num, old_c_in))
            rank = get_filter_rank(old_layer)
            importance_rank = rank
            for i in range(0, incremental_num):
                index = rank[0]
                weights[i] = old_weights[index] * numerator
            size_needed = np.random.uniform(-0.1, 0.1, (incremental_num, old_c_in))
            new_bias_needed = np.random.uniform(-0.1, 0.1, incremental_num)
        else:
            rank = importance_rank
            weights = np.zeros((old_c_out, incremental_num))
            if is_multi:
                multiplier = 16
            else:
                multiplier = 1
            for i in range(0, incremental_num // multiplier):
                index = rank[0]
                start = index * multiplier
                end = start + multiplier
                start_i = i * multiplier
                end_i = start_i + multiplier
                weights[:, start_i:end_i] = old_weights[:, start:end] * numerator
            size_needed = np.random.uniform(-0.1, 0.1, (old_c_out, incremental_num))
    base_weight = size_needed + weights
    return base_weight, new_bias_needed


def weight_adaption_random(old_layer, incremental_num, is_first=True, val_min=-0.1, val_max=0.1):
    old_weights = old_layer.weight.data.cpu().numpy()
    new_bias_needed = None

    if isinstance(old_layer, torch.nn.Conv2d):
        old_c_out, old_c_in, k_h, k_w = old_weights.shape
        if is_first:
            weights = torch.empty(incremental_num, old_c_in, k_h, k_w)
            nn.init.uniform_(weights, val_min, val_max)
            new_bias_needed = torch.empty(incremental_num)
            nn.init.uniform_(new_bias_needed, val_min, val_max)
        else:
            weights = torch.empty(old_c_out, incremental_num, k_h, k_w)
            nn.init.uniform_(weights, val_min, val_max)

    elif isinstance(old_layer, torch.nn.Linear):
        old_c_out, old_c_in = old_weights.shape
        if is_first:
            weights = torch.empty(incremental_num, old_c_in)
            nn.init.uniform_(weights, val_min, val_max)
            new_bias_needed = torch.empty(incremental_num)
            nn.init.uniform_(new_bias_needed, val_min, val_max)
        else:
            weights = torch.empty(old_c_out, incremental_num)
            nn.init.uniform_(weights, val_min, val_max)
    return weights, new_bias_needed


def weight_adaption_rank_ours(old_layer, incremental_num, is_first=True, is_multi=False):
    global divide_num
    global importance_rank
    old_weights = old_layer.weight.data.cpu().numpy()
    new_bias = None

    s_index = 0 if is_first else 1
    r = 1 - (incremental_num / (old_weights.shape[s_index] + incremental_num))

    if isinstance(old_layer, torch.nn.Conv2d):
        old_c_out, old_c_in, k_h, k_w = old_weights.shape
        if is_first:
            scale_factor = 1 / math.sqrt(old_c_in * k_h * k_w)
            weights = np.zeros((incremental_num, old_c_in, k_h, k_w))
            rank = get_filter_rank(old_layer, ascending=True)
            importance_rank = rank
            for i in range(0, incremental_num):
                index = rank[i]
                weights[i] = old_weights[index]
            new_bias = np.random.uniform(-0.1, 0.1, incremental_num)
        else:
            scale_factor = 1 / math.sqrt((old_c_in + incremental_num) * k_h * k_w)
            rank = importance_rank
            weights = np.zeros((old_c_out, incremental_num, k_h, k_w))
            for i in range(1, incremental_num):
                index = rank[i]
                weights[:, i] = old_weights[:, index]

    elif isinstance(old_layer, torch.nn.Linear):
        old_c_out, old_c_in = old_weights.shape
        if is_first:
            scale_factor = 1 / math.sqrt(old_c_in)
            weights = np.zeros((incremental_num, old_c_in))
            rank = get_filter_rank(old_layer, ascending=True)
            importance_rank = rank
            for i in range(0, incremental_num):
                index = rank[i]
                weights[i] = old_weights[index]
            new_bias = np.random.uniform(-0.1, 0.1, incremental_num)
        else:
            scale_factor = 1 / math.sqrt(old_c_in + incremental_num)
            rank = importance_rank
            weights = np.zeros((old_c_out, incremental_num))
            if is_multi:
                multiplier = 16
            else:
                multiplier = 1
            for i in range(0, incremental_num // multiplier):
                index = rank[i]
                start = index * multiplier
                end = start + multiplier
                start_i = i * multiplier
                end_i = start_i + multiplier
                weights[:, start_i:end_i] = old_weights[:, start:end]

    old_scale = max(np.abs(np.min(old_weights)), np.abs(np.max(old_weights)))
    numerator = scale_factor / old_scale
    weights = weights * numerator
    # print('old={:2.10f}| new={:2.10f}| scale={:2.10f}'.format(old_scale, scale_factor, numerator))

    size_needed = np.random.uniform(-0.1, 0.1, weights.shape) * numerator
    base_weight = size_needed + weights

    if new_bias is not None:
        new_bias = new_bias * numerator
    return base_weight, new_bias


def weight_adaption_rank_connect_noscale(old_layer, incremental_num, is_first=True, is_multi=False):
    old_weights = old_layer.weight.data.cpu().numpy()
    new_bias_needed = None
    global divide_num
    numerator = 1
    global importance_rank
    if isinstance(old_layer, torch.nn.Conv2d):
        old_c_out, old_c_in, k_h, k_w = old_weights.shape
        # old_bias = old_layer.bias.data.cpu().numpy()
        if is_first:
            weights = np.zeros((incremental_num, old_c_in, k_h, k_w))
            rank = get_filter_rank(old_layer)
            importance_rank = rank
            for i in range(0, incremental_num):
                index = rank[i]
                weights[i] = old_weights[index]
            size_needed = np.random.uniform(-0.1, 0.1, (incremental_num, old_c_in, k_h, k_w))
            new_bias_needed = np.random.uniform(-0.1, 0.1, incremental_num)
        else:
            rank = importance_rank
            weights = np.zeros((old_c_out, incremental_num, k_h, k_w))
            for i in range(1, incremental_num):
                index = rank[i]
                weights[:, i] = old_weights[:, index] / numerator
            size_needed = np.random.uniform(-0.1, 0.1, (old_c_out, incremental_num, k_h, k_w))

    elif isinstance(old_layer, torch.nn.Linear):
        old_c_out, old_c_in = old_weights.shape
        # old_bias = old_layer.bias.data.cpu().numpy()
        if is_first:
            weights = np.zeros((incremental_num, old_c_in))
            rank = get_filter_rank(old_layer)
            importance_rank = rank
            for i in range(0, incremental_num):
                index = rank[i]
                weights[i] = old_weights[index]
            size_needed = np.random.uniform(-0.1, 0.1, (incremental_num, old_c_in))
            new_bias_needed = np.random.uniform(-0.1, 0.1, incremental_num)
        else:
            rank = importance_rank
            weights = np.zeros((old_c_out, incremental_num))
            if is_multi:
                multiplier = 16
            else:
                multiplier = 1
            for i in range(0, incremental_num // multiplier):
                index = rank[i]
                start = index * multiplier
                end = start + multiplier
                start_i = i * multiplier
                end_i = start_i + multiplier
                weights[:, start_i:end_i] = old_weights[:, start:end] / (numerator * multiplier)
            size_needed = np.random.uniform(-0.1, 0.1, (old_c_out, incremental_num))
    base_weight = size_needed + weights
    return base_weight, new_bias_needed


def weight_adaption_rank_cumulative(old_layer, incremental_num, is_first=True, is_multi=False):
    old_weights = old_layer.weight.data.cpu().numpy()
    new_bias_needed = None
    global divide_num
    # numerator = 2  # incremental_num
    # s_index = 0 if is_first else 1
    # numerator = 1 - (incremental_num / (old_weights.shape[s_index] + incremental_num))
    # numerator /= 2

    numerator = 1 / 2
    global importance_rank
    global cumulative_layer_index
    if isinstance(old_layer, torch.nn.Conv2d):
        old_c_out, old_c_in, k_h, k_w = old_weights.shape
        # old_bias = old_layer.bias.data.cpu().numpy()
        if is_first:
            weights = np.zeros((incremental_num, old_c_in, k_h, k_w))
            rank = rank_meter(cumulative_layer_index)
            importance_rank = rank
            for i in range(0, incremental_num):
                index = rank[i]
                # weights[i] = old_weights[index] / numerator  # numerator
                weights[i] = old_weights[index] * numerator  # numerator
            size_needed = np.random.uniform(-0.1, 0.1, (incremental_num, old_c_in, k_h, k_w))
            new_bias_needed = np.random.uniform(-0.1, 0.1, incremental_num)
        else:
            rank = importance_rank
            weights = np.zeros((old_c_out, incremental_num, k_h, k_w))
            for i in range(1, incremental_num):
                index = rank[i]
                # weights[:, i] = old_weights[:, index] / numerator  # * numerator
                weights[:, i] = old_weights[:, index] * numerator
            size_needed = np.random.uniform(-0.1, 0.1, (old_c_out, incremental_num, k_h, k_w))

    elif isinstance(old_layer, torch.nn.Linear):
        old_c_out, old_c_in = old_weights.shape
        # old_bias = old_layer.bias.data.cpu().numpy()
        if is_first:
            weights = np.zeros((incremental_num, old_c_in))
            rank = rank_meter(cumulative_layer_index)
            importance_rank = rank
            for i in range(0, incremental_num):
                index = rank[i]
                # weights[i] = old_weights[index] / numerator
                weights[i] = old_weights[index] * numerator
            size_needed = np.random.uniform(-0.1, 0.1, (incremental_num, old_c_in))
            new_bias_needed = np.random.uniform(-0.1, 0.1, incremental_num)
        else:
            rank = importance_rank
            weights = np.zeros((old_c_out, incremental_num))
            if is_multi:
                multiplier = 16
            else:
                multiplier = 1
            for i in range(0, incremental_num // multiplier):
                index = rank[i]
                start = index * multiplier
                end = start + multiplier
                start_i = i * multiplier
                end_i = start_i + multiplier
                if multiplier == 1:
                    weights[:, start_i:end_i] = old_weights[:, start:end]
                else:
                    weights[:, start_i:end_i] = old_weights[:, start:end] * numerator
                # weights[:, start_i:end_i] = old_weights[:, start:end] / numerator  #(numerator // multiplier)
                # weights[:, start_i:end_i] = old_weights[:, start:end] #* numerator  # (numerator // multiplier)
            size_needed = np.random.uniform(-0.1, 0.1, (old_c_out, incremental_num))
    base_weight = size_needed + weights
    return base_weight, new_bias_needed


def weight_adaption_ranklow_one(old_layer, incremental_num, is_first=True, is_multi=False):
    old_weights = old_layer.weight.data.cpu().numpy()
    new_bias_needed = None
    numerator = incremental_num
    global importance_rank
    if isinstance(old_layer, torch.nn.Conv2d):
        old_c_out, old_c_in, k_h, k_w = old_weights.shape
        # old_bias = old_layer.bias.data.cpu().numpy()
        if is_first:
            weights = np.zeros((incremental_num, old_c_in, k_h, k_w))
            rank = get_filter_rank(old_layer, ascending=True)
            importance_rank = rank
            for i in range(0, incremental_num):
                index = rank[0]
                weights[i] = old_weights[index]
            size_needed = np.random.uniform(-0.1, 0.1, (incremental_num, old_c_in, k_h, k_w))
            new_bias_needed = np.random.uniform(-0.1, 0.1, incremental_num)
        else:
            rank = importance_rank
            weights = np.zeros((old_c_out, incremental_num, k_h, k_w))
            for i in range(1, incremental_num):
                index = rank[0]
                weights[:, i] = old_weights[:, index] / (numerator + 1)
            size_needed = np.random.uniform(-0.1, 0.1, (old_c_out, incremental_num, k_h, k_w))

    elif isinstance(old_layer, torch.nn.Linear):
        old_c_out, old_c_in = old_weights.shape
        # old_bias = old_layer.bias.data.cpu().numpy()
        if is_first:
            weights = np.zeros((incremental_num, old_c_in))
            rank = get_filter_rank(old_layer, ascending=True)
            importance_rank = rank
            for i in range(0, incremental_num):
                index = rank[0]
                weights[i] = old_weights[index]
            size_needed = np.random.uniform(-0.1, 0.1, (incremental_num, old_c_in))
            new_bias_needed = np.random.uniform(-0.1, 0.1, incremental_num)
        else:
            rank = importance_rank
            weights = np.zeros((old_c_out, incremental_num))
            if is_multi:
                multiplier = 16
            else:
                multiplier = 1
            for i in range(0, incremental_num // multiplier):
                index = rank[0]
                start = index * multiplier
                end = start + multiplier
                start_i = i * multiplier
                end_i = start_i + multiplier
                weights[:, start_i:end_i] = old_weights[:, start:end] / ((numerator + multiplier) // multiplier)
            size_needed = np.random.uniform(-0.1, 0.1, (old_c_out, incremental_num))
    base_weight = size_needed + weights
    return base_weight, new_bias_needed


def weight_adaption_ranklow_n(old_layer, incremental_num, is_first=True, is_multi=False):
    old_weights = old_layer.weight.data.cpu().numpy()
    new_bias_needed = None
    numerator = incremental_num
    global importance_rank
    if isinstance(old_layer, torch.nn.Conv2d):
        old_c_out, old_c_in, k_h, k_w = old_weights.shape
        # old_bias = old_layer.bias.data.cpu().numpy()
        if is_first:
            weights = np.zeros((incremental_num, old_c_in, k_h, k_w))
            rank = get_filter_rank(old_layer, ascending=True)
            importance_rank = rank
            for i in range(0, incremental_num):
                index = rank[i]
                weights[i] = old_weights[index] / numerator
            size_needed = np.random.uniform(-0.1, 0.1, (incremental_num, old_c_in, k_h, k_w))
            new_bias_needed = np.random.uniform(-0.1, 0.1, incremental_num)
        else:
            rank = importance_rank
            weights = np.zeros((old_c_out, incremental_num, k_h, k_w))
            for i in range(1, incremental_num):
                index = rank[i]
                weights[:, i] = old_weights[:, index] / numerator
            size_needed = np.random.uniform(-0.1, 0.1, (old_c_out, incremental_num, k_h, k_w))

    elif isinstance(old_layer, torch.nn.Linear):
        old_c_out, old_c_in = old_weights.shape
        # old_bias = old_layer.bias.data.cpu().numpy()
        if is_first:
            weights = np.zeros((incremental_num, old_c_in))
            rank = get_filter_rank(old_layer, ascending=True)
            importance_rank = rank
            for i in range(0, incremental_num):
                index = rank[i]
                weights[i] = old_weights[index] / numerator
            size_needed = np.random.uniform(-0.1, 0.1, (incremental_num, old_c_in))
            new_bias_needed = np.random.uniform(-0.1, 0.1, incremental_num)
        else:
            rank = importance_rank
            weights = np.zeros((old_c_out, incremental_num))
            if is_multi:
                multiplier = 16
            else:
                multiplier = 1
            for i in range(0, incremental_num // multiplier):
                index = rank[i]
                start = index * multiplier
                end = start + multiplier
                start_i = i * multiplier
                end_i = start_i + multiplier
                weights[:, start_i:end_i] = old_weights[:, start:end] / numerator
            size_needed = np.random.uniform(-0.1, 0.1, (old_c_out, incremental_num))
    base_weight = size_needed + weights
    return base_weight, new_bias_needed


def weight_adaption_activation_low(old_layer, incremental_num, is_first=True, is_multi=False):
    old_weights = old_layer.weight.data.cpu().numpy()
    new_bias_needed = None
    numerator = incremental_num
    global activation_rank
    if isinstance(old_layer, torch.nn.Conv2d):
        old_c_out, old_c_in, k_h, k_w = old_weights.shape
        # old_bias = old_layer.bias.data.cpu().numpy()
        if is_first:
            weights = np.zeros((incremental_num, old_c_in, k_h, k_w))
            rank = get_rank(activation_rank, ascending=True)
            activation_rank = rank
            for i in range(0, incremental_num):
                index = rank[i]
                weights[i] = old_weights[index] / numerator
            size_needed = np.random.uniform(-0.1, 0.1, (incremental_num, old_c_in, k_h, k_w))
            new_bias_needed = np.random.uniform(-0.1, 0.1, incremental_num)
        else:
            rank = activation_rank
            weights = np.zeros((old_c_out, incremental_num, k_h, k_w))
            for i in range(1, incremental_num):
                index = rank[i]
                weights[:, i] = old_weights[:, index] / numerator
            size_needed = np.random.uniform(-0.1, 0.1, (old_c_out, incremental_num, k_h, k_w))

    elif isinstance(old_layer, torch.nn.Linear):
        old_c_out, old_c_in = old_weights.shape
        # old_bias = old_layer.bias.data.cpu().numpy()
        if is_first:
            weights = np.zeros((incremental_num, old_c_in))
            rank = get_rank(activation_rank, ascending=True)
            activation_rank = rank
            for i in range(0, incremental_num):
                index = rank[i]
                weights[i] = old_weights[index] / numerator
            size_needed = np.random.uniform(-0.1, 0.1, (incremental_num, old_c_in))
            new_bias_needed = np.random.uniform(-0.1, 0.1, incremental_num)
        else:
            rank = activation_rank
            weights = np.zeros((old_c_out, incremental_num))
            if is_multi:
                multiplier = 16
            else:
                multiplier = 1
            for i in range(0, incremental_num // multiplier):
                index = rank[i]
                start = index * multiplier
                end = start + multiplier
                start_i = i * multiplier
                end_i = start_i + multiplier
                weights[:, start_i:end_i] = old_weights[:, start:end] / numerator
            size_needed = np.random.uniform(-0.1, 0.1, (old_c_out, incremental_num))
    base_weight = size_needed + weights
    return base_weight, new_bias_needed


def weight_adaption_rank_connect1(old_layer, incremental_num, is_first=True, rank=None, is_multi=False):
    old_weights = old_layer.weight.data.cpu().numpy()
    new_bias_needed = None
    if isinstance(old_layer, torch.nn.Conv2d):
        old_c_out, old_c_in, k_h, k_w = old_weights.shape
        # old_bias = old_layer.bias.data.cpu().numpy()
        if is_first:
            rank = get_filter_rank(old_layer)
            weights = np.reshape(old_weights[rank[0], :, :, :], (1, old_c_in, k_h, k_w))
            for i in range(1, incremental_num):
                index = rank[i]
                weights = np.vstack((weights, np.reshape(old_weights[index, :, :, :], (1, old_c_in, k_h, k_w))))
            size_needed = np.random.uniform(-0.1, 0.1,
                                            (incremental_num, old_c_in, k_h, k_w))
            new_bias_needed = np.random.uniform(-0.1, 0.1,
                                                incremental_num)
        else:
            weights = np.reshape(old_weights[:, rank[0], :, :], (old_c_out, 1, k_h, k_w)) / 2
            for i in range(1, incremental_num):
                index = rank[0]
                weights = np.hstack((weights, np.reshape(old_weights[:, index, :, :], (old_c_out, 1, k_h, k_w)) / 2))
            size_needed = np.random.uniform(-0.1, 0.1,
                                            (old_c_out, incremental_num, k_h, k_w))

    elif isinstance(old_layer, torch.nn.Linear):
        old_c_out, old_c_in = old_weights.shape
        old_bias = old_layer.bias.data.cpu().numpy()
        if is_first:
            rank = get_filter_rank(old_layer)
            weights = old_weights[rank[0], :]
            for i in range(1, incremental_num):
                index = rank[0]
                weights = np.vstack((weights, old_weights[index, :]))
            size_needed = np.random.uniform(-0.1, 0.1,
                                            (incremental_num, old_c_in))
            new_bias_needed = np.random.uniform(-0.1, 0.1,
                                                incremental_num)
        else:
            weights = np.zeros((old_c_out, incremental_num))
            if is_multi:
                multiplier = 16
            else:
                multiplier = 1
            for i in range(0, incremental_num // multiplier):
                index = rank[0]
                start = index * multiplier
                end = start + multiplier
                start_i = i * multiplier
                end_i = start_i + multiplier
                weights[:, start_i:end_i] = old_weights[:, start:end] / 2
            size_needed = np.random.uniform(np.mean(old_weights) * -1, np.mean(old_weights),
                                            (old_c_out, incremental_num))

    base_weight = size_needed + weights
    return base_weight, new_bias_needed


def group_filters(num_filters, rank, num_group):
    n = num_filters // num_group
    if n == 0:
        n = 1
    start = 0
    groups = []
    while True:
        group = []
        for i in range(start, start + n):
            group.append(rank[i])
            start += 1
        groups.append(group)
        if start == num_group * n or n < 2:
            break
    return groups


def weight_adaption_rank_group2(old_layer, incremental_num, is_first=True, rank=None, multi=16, is_multi=False):
    old_weights = old_layer.weight.data.cpu().numpy()
    new_bias_needed = None
    cosine_array = []
    global cosine_rank
    if isinstance(old_layer, torch.nn.Conv2d):
        old_c_out, old_c_in, k_h, k_w = old_weights.shape
        old_bias = old_layer.bias.data.cpu().numpy()
        if is_first:
            weights = np.zeros((incremental_num, old_c_in, k_h, k_w))
            rank = get_filter_rank(old_layer)
            rank1_index = rank[0]
            x_weight = old_weights[rank1_index, :, :, :]
            for index in range(old_c_out):
                y_weight = old_weights[index, :, :, :]
                top = np.sum(x_weight * y_weight)
                bottom = np.sqrt(np.sum(x_weight ** 2)) * np.sqrt(np.sum(x_weight ** 2))
                cosine_array.append(top / bottom)
            c_rank = get_rank(cosine_array)
            for order in range(incremental_num):
                weights[order] = old_weights[c_rank.index(order)]
            mean_weight = np.reshape(np.mean(weights, axis=0), (1, old_c_in, k_h, k_w))
            weights = multi_weights(mean_weight, incremental_num, is_first=True)
            size_needed = np.random.uniform(np.min(old_weights) / 10, np.max(old_weights) / 10,
                                            (incremental_num, old_c_in, k_h, k_w))
            new_bias_needed = np.random.uniform(np.min(old_bias) / 10, np.max(old_bias) / 10,
                                                incremental_num)
            cosine_rank = c_rank
        else:
            weights = np.zeros((old_c_out, incremental_num, k_h, k_w))
            for order in range(incremental_num):
                weights[:, order, :, :] = old_weights[:, cosine_rank.index(order), :, :]
            mean_weight = np.reshape(np.mean(weights, axis=1), (old_c_out, 1, k_h, k_w)) / incremental_num
            weights = multi_weights(mean_weight, incremental_num, is_first=False)
            size_needed = np.random.uniform(np.min(old_weights) / 10, np.max(old_weights) / 10,
                                            (old_c_out, incremental_num, k_h, k_w))

    elif isinstance(old_layer, torch.nn.Linear):
        old_c_out, old_c_in = old_weights.shape
        old_bias = old_layer.bias.data.cpu().numpy()
        if is_first:
            weights = np.zeros((incremental_num, old_c_in))
            rank = get_filter_rank(old_layer)
            rank1_index = rank[0]
            x_weight = old_weights[rank1_index, :]
            for index in range(old_c_out):
                y_weight = old_weights[index, :]
                top = np.sum(x_weight * y_weight)
                bottom = np.sqrt(np.sum(x_weight ** 2)) * np.sqrt(np.sum(x_weight ** 2))
                cosine_array.append(top / bottom)
            c_rank = get_rank(cosine_array)
            for order in range(incremental_num):
                weights[order] = old_weights[c_rank.index(order)]
            mean_weight = np.reshape(np.mean(weights, axis=0), (1, old_c_in))
            weights = multi_weights(mean_weight, incremental_num, is_first=True)
            size_needed = np.random.uniform(np.min(old_weights) / 10, np.max(old_weights) / 10,
                                            (incremental_num, old_c_in))
            new_bias_needed = np.random.uniform(np.min(old_bias) / 10, np.max(old_bias) / 10,
                                                incremental_num)
            cosine_rank = c_rank
        else:
            if not is_multi:
                multi = 1
            weights = np.zeros((old_c_out, incremental_num))
            num_needed = incremental_num // multi
            for i in range(num_needed):
                start = cosine_rank[i] * multi
                end = start + multi
                start_i = i * multi
                end_i = start_i + multi
                weights[:, start_i:end_i] = old_weights[:, start:end]
            mean_weight = np.reshape(np.mean(weights, axis=1), (old_c_out, 1)) / incremental_num
            weights = multi_weights(mean_weight, incremental_num=incremental_num, is_first=False)
            size_needed = np.random.uniform(np.min(old_weights) / 10, np.max(old_weights) / 10,
                                            (old_c_out, incremental_num))

    base_weight = size_needed + weights
    return base_weight, new_bias_needed


def weight_adaption_rank_cosine_all(old_layer, incremental_num, is_first=True, multi=16, is_multi=False, asc=True):
    old_weights = old_layer.weight.data.cpu().numpy()
    new_bias = None
    cosine_array = []
    global cosine_rank
    global divide_num

    if isinstance(old_layer, torch.nn.Conv2d):
        old_c_out, old_c_in, k_h, k_w = old_weights.shape
        if is_first:
            scale_factor = 1 / math.sqrt(old_c_in * k_h * k_w)
            weights = np.zeros((incremental_num, old_c_in, k_h, k_w))
            rank = get_filter_rank(old_layer, ascending=asc)
            rank1_index = rank[0]
            x_weight = old_weights[rank1_index, :, :, :]
            x_weight = np.reshape(x_weight, (old_c_in, k_h * k_w))
            for index in range(old_c_out):
                y_weight = old_weights[index, :, :, :]
                y_weight = np.reshape(y_weight, (old_c_in, k_h * k_w))
                cosine_array.append(cosine_similarity(x_weight, y_weight)[0][0])
            c_rank = get_rank(cosine_array)
            for order in range(incremental_num):
                weights[order] = old_weights[c_rank.index(order)]
            new_bias = np.random.uniform(-0.1, 0.1, incremental_num)
            cosine_rank = c_rank
            insert_rank(c_rank)
        else:
            scale_factor = 1 / math.sqrt((old_c_in + incremental_num) * k_h * k_w)
            weights = np.zeros((old_c_out, incremental_num, k_h, k_w))
            for order in range(incremental_num):
                weights[:, order, :, :] = old_weights[:, cosine_rank.index(order), :, :]

    elif isinstance(old_layer, torch.nn.Linear):
        old_c_out, old_c_in = old_weights.shape
        if is_first:
            scale_factor = 1 / math.sqrt(old_c_in)
            weights = np.zeros((incremental_num, old_c_in))
            rank = get_filter_rank(old_layer, ascending=asc)
            rank1_index = rank[0]
            x_weight = old_weights[rank1_index, :]
            for index in range(old_c_out):
                y_weight = old_weights[index, :]
                top = np.sum(x_weight * y_weight)
                bottom = np.sqrt(np.sum(x_weight ** 2)) * np.sqrt(np.sum(x_weight ** 2))
                cosine_array.append(top / bottom)
            c_rank = get_rank(cosine_array)
            for order in range(incremental_num):
                weights[order] = old_weights[c_rank.index(order)]
            new_bias = np.random.uniform(-0.1, 0.1, incremental_num)
            cosine_rank = c_rank
            insert_rank(c_rank)
        else:
            scale_factor = 1 / math.sqrt(old_c_in + incremental_num)
            if not is_multi:
                multi = 1
            weights = np.zeros((old_c_out, incremental_num))
            num_needed = incremental_num // multi
            for i in range(num_needed):
                start = cosine_rank[i] * multi
                end = start + multi
                start_i = i * multi
                end_i = start_i + multi
                weights[:, start_i:end_i] = old_weights[:, start:end]

    old_scale = max(np.abs(np.min(old_weights)), np.abs(np.max(old_weights)))
    numerator = scale_factor / old_scale
    weights = weights * numerator
    size_needed = np.random.uniform(-0.1, 0.1, weights.shape) * numerator
    base_weight = size_needed + weights
    # print('old={:2.10f}| new={:2.10f}| scale={:2.10f}'.format(old_scale, scale_factor, numerator))

    if new_bias is not None:
        new_bias = new_bias * numerator
    return base_weight, new_bias


def weight_adaption_rank_cosine(old_layer, incremental_num, is_first=True, multi=16, is_multi=False, asc=True):
    old_weights = old_layer.weight.data.cpu().numpy()
    new_bias = None
    cosine_array = []
    global cosine_rank
    global lambda_list

    if isinstance(old_layer, torch.nn.Conv2d):
        old_c_out, old_c_in, k_h, k_w = old_weights.shape
        if is_first:
            n_tensor = torch.zeros(incremental_num + old_c_out, old_c_in, k_h, k_w)
            weights = np.zeros((incremental_num, old_c_in, k_h, k_w))
            rank = get_filter_rank(old_layer, ascending=asc)
            rank1_index = rank[0]
            x_weight = old_weights[rank1_index, :, :, :]
            x_weight = np.reshape(x_weight, (old_c_in, k_h * k_w))
            for index in range(old_c_out):
                y_weight = old_weights[index, :, :, :]
                y_weight = np.reshape(y_weight, (old_c_in, k_h * k_w))
                cosine_array.append(cosine_similarity(x_weight, y_weight)[0][0])
            c_rank = get_rank(cosine_array)
            for order in range(incremental_num):
                weights[order] = old_weights[c_rank.index(order)]
            new_bias = np.random.uniform(-0.1, 0.1, incremental_num)
            cosine_rank = c_rank
            insert_rank(c_rank)
        else:
            n_tensor = torch.zeros(old_c_out, incremental_num + old_c_in, k_h, k_w)
            weights = np.zeros((old_c_out, incremental_num, k_h, k_w))
            for order in range(incremental_num):
                weights[:, order, :, :] = old_weights[:, cosine_rank.index(order), :, :]

    elif isinstance(old_layer, torch.nn.Linear):
        old_c_out, old_c_in = old_weights.shape
        if is_first:
            n_tensor = torch.zeros(incremental_num + old_c_out, old_c_in)
            weights = np.zeros((incremental_num, old_c_in))
            rank = get_filter_rank(old_layer, ascending=asc)
            rank1_index = rank[0]
            x_weight = old_weights[rank1_index, :]
            for index in range(old_c_out):
                y_weight = old_weights[index, :]
                top = np.sum(x_weight * y_weight)
                bottom = np.sqrt(np.sum(x_weight ** 2)) * np.sqrt(np.sum(x_weight ** 2))
                cosine_array.append(top / bottom)
            c_rank = get_rank(cosine_array)
            for order in range(incremental_num):
                weights[order] = old_weights[c_rank.index(order)]
            new_bias = np.random.uniform(-0.1, 0.1, incremental_num)
            cosine_rank = c_rank
            insert_rank(c_rank)
        else:
            n_tensor = torch.zeros(old_c_out, incremental_num + old_c_in)
            if not is_multi:
                multi = 1
            weights = np.zeros((old_c_out, incremental_num))
            num_needed = incremental_num // multi
            for i in range(num_needed):
                start = cosine_rank[i] * multi
                end = start + multi
                start_i = i * multi
                end_i = start_i + multi
                weights[:, start_i:end_i] = old_weights[:, start:end]

    scale = target_scale_calculator(n_tensor)
    old_scale = max(np.abs(np.min(weights)), np.abs(np.max(weights)))
    numerator = scale / old_scale
    if is_first:
        lambda_list.append(numerator)
    weights = weights * numerator
    size_needed = np.random.uniform(-0.1, 0.1, weights.shape) * numerator
    base_weight = size_needed + weights
    # print('old={:2.10f}| new={:2.10f}| scale={:2.10f}'.format(old_scale, scale_factor, numerator))

    if new_bias is not None:
        new_bias = new_bias * numerator
    return base_weight, new_bias


def weight_adaption_rank_cosine1(old_layer, incremental_num, is_first=True, multi=16, is_multi=False):
    old_weights = old_layer.weight.data.cpu().numpy()
    new_bias_needed = None
    cosine_array = []
    global cosine_rank
    global divide_num
    s_index = 0 if is_first else 1
    numerator = 1 - (incremental_num / (old_weights.shape[s_index] + incremental_num))
    numerator /= 2
    if isinstance(old_layer, torch.nn.Conv2d):
        old_c_out, old_c_in, k_h, k_w = old_weights.shape
        old_bias = old_layer.bias.data.cpu().numpy()
        if is_first:
            weights = np.zeros((incremental_num, old_c_in, k_h, k_w))
            rank = get_filter_rank(old_layer)
            rank1_index = rank[0]
            x_weight = old_weights[rank1_index, :, :, :]
            x_weight = np.reshape(x_weight, (old_c_in, k_h * k_w))
            for index in range(old_c_out):
                y_weight = old_weights[index, :, :, :]
                y_weight = np.reshape(y_weight, (old_c_in, k_h * k_w))
                cosine_array.append(cosine_similarity(x_weight, y_weight)[0][0])
            c_rank = get_rank(cosine_array)
            for order in range(incremental_num):
                weights[order] = old_weights[c_rank.index(order)] * numerator
            # mean_weight = np.reshape(np.mean(weights, axis=0), (1, old_c_in, k_h, k_w))
            # weights = multi_weights(mean_weight, incremental_num, is_first=True)
            size_needed = np.random.uniform(np.min(old_weights) / 10, np.max(old_weights) / 10,
                                            (incremental_num, old_c_in, k_h, k_w))
            new_bias_needed = np.random.uniform(np.min(old_bias) / 10, np.max(old_bias) / 10,
                                                incremental_num)
            cosine_rank = c_rank
            insert_rank(c_rank)
        else:
            weights = np.zeros((old_c_out, incremental_num, k_h, k_w))
            for order in range(incremental_num):
                weights[:, order, :, :] = old_weights[:, cosine_rank.index(order), :, :] * numerator
            # mean_weight = np.reshape(np.mean(weights, axis=1), (old_c_out, 1, k_h, k_w)) / (incremental_num + 1)
            # weights = multi_weights(mean_weight, incremental_num, is_first=False)
            size_needed = np.random.uniform(np.min(old_weights) / 10, np.max(old_weights) / 10,
                                            (old_c_out, incremental_num, k_h, k_w))

    elif isinstance(old_layer, torch.nn.Linear):
        old_c_out, old_c_in = old_weights.shape
        old_bias = old_layer.bias.data.cpu().numpy()
        if is_first:
            weights = np.zeros((incremental_num, old_c_in))
            rank = get_filter_rank(old_layer)
            rank1_index = rank[0]
            x_weight = old_weights[rank1_index, :]
            for index in range(old_c_out):
                y_weight = old_weights[index, :]
                top = np.sum(x_weight * y_weight)
                bottom = np.sqrt(np.sum(x_weight ** 2)) * np.sqrt(np.sum(x_weight ** 2))
                cosine_array.append(top / bottom)
            c_rank = get_rank(cosine_array)
            for order in range(incremental_num):
                weights[order] = old_weights[c_rank.index(order)] * numerator
            # mean_weight = np.reshape(np.mean(weights, axis=0), (1, old_c_in))
            # weights = multi_weights(mean_weight, incremental_num, is_first=True)
            size_needed = np.random.uniform(np.min(old_weights) / 10, np.max(old_weights) / 10,
                                            (incremental_num, old_c_in))
            new_bias_needed = np.random.uniform(np.min(old_bias) / 10, np.max(old_bias) / 10,
                                                incremental_num)
            cosine_rank = c_rank
            insert_rank(c_rank)
        else:
            if not is_multi:
                multi = 1
            weights = np.zeros((old_c_out, incremental_num))
            num_needed = incremental_num // multi
            for i in range(num_needed):
                start = cosine_rank[i] * multi
                end = start + multi
                start_i = i * multi
                end_i = start_i + multi
                weights[:, start_i:end_i] = old_weights[:, start:end] * numerator
            # mean_weight = np.reshape(np.mean(weights, axis=1), (old_c_out, 1)) / (incremental_num + 1)
            # weights = multi_weights(mean_weight, incremental_num=incremental_num, is_first=False)
            size_needed = np.random.uniform(np.min(old_weights) / 10, np.max(old_weights) / 10,
                                            (old_c_out, incremental_num))

    base_weight = size_needed + weights
    return base_weight, new_bias_needed


def weight_adaption_rank_cumulative_cosine(old_layer, incremental_num, is_first=True, multi=16, is_multi=False,
                                           ascending=False):
    old_weights = old_layer.weight.data.cpu().numpy()
    new_bias_needed = None
    cosine_array = []
    global cosine_rank
    global cumulative_rank
    global divide_num
    global meter_list
    global cumulative_layer_index
    s_index = 0 if is_first else 1
    numerator = 1 - (incremental_num / (old_weights.shape[s_index] + incremental_num))
    numerator /= 2
    if isinstance(old_layer, torch.nn.Conv2d):
        old_c_out, old_c_in, k_h, k_w = old_weights.shape
        old_bias = old_layer.bias.data.cpu().numpy()
        if is_first:
            weights = np.zeros((incremental_num, old_c_in, k_h, k_w))
            rank = rank_meter(cumulative_layer_index, ascending=ascending)
            cumulative_rank = rank
            rank1_index = rank[0]
            x_weight = old_weights[rank1_index, :, :, :]
            x_weight = np.reshape(x_weight, (old_c_in, k_h * k_w))
            for index in range(old_c_out):
                y_weight = old_weights[index, :, :, :]
                y_weight = np.reshape(y_weight, (old_c_in, k_h * k_w))
                cosine_array.append(cosine_similarity(x_weight, y_weight)[0][0])
            c_rank = get_rank(cosine_array)
            for order in range(incremental_num):
                weights[order] = old_weights[c_rank.index(order)] * numerator
            # mean_weight = np.reshape(np.mean(weights, axis=0), (1, old_c_in, k_h, k_w))
            # weights = multi_weights(mean_weight, incremental_num, is_first=True)
            size_needed = np.random.uniform(np.min(old_weights) / 10, np.max(old_weights) / 10,
                                            (incremental_num, old_c_in, k_h, k_w))
            new_bias_needed = np.random.uniform(np.min(old_bias) / 10, np.max(old_bias) / 10,
                                                incremental_num)
            cosine_rank = c_rank
            # insert_rank(c_rank)
        else:
            weights = np.zeros((old_c_out, incremental_num, k_h, k_w))
            for order in range(incremental_num):
                weights[:, order, :, :] = old_weights[:, cosine_rank.index(order), :, :] * numerator
            # mean_weight = np.reshape(np.mean(weights, axis=1), (old_c_out, 1, k_h, k_w)) / (incremental_num + 1)
            # weights = multi_weights(mean_weight, incremental_num, is_first=False)
            size_needed = np.random.uniform(np.min(old_weights) / 10, np.max(old_weights) / 10,
                                            (old_c_out, incremental_num, k_h, k_w))

    elif isinstance(old_layer, torch.nn.Linear):
        old_c_out, old_c_in = old_weights.shape
        old_bias = old_layer.bias.data.cpu().numpy()
        if is_first:
            weights = np.zeros((incremental_num, old_c_in))
            rank = rank_meter(cumulative_layer_index, ascending=ascending)
            cumulative_rank = rank
            rank1_index = rank[0]
            x_weight = old_weights[rank1_index, :]
            for index in range(old_c_out):
                y_weight = old_weights[index, :]
                top = np.sum(x_weight * y_weight)
                bottom = np.sqrt(np.sum(x_weight ** 2)) * np.sqrt(np.sum(x_weight ** 2))
                cosine_array.append(top / bottom)
            c_rank = get_rank(cosine_array)
            for order in range(incremental_num):
                weights[order] = old_weights[c_rank.index(order)] * numerator
            # mean_weight = np.reshape(np.mean(weights, axis=0), (1, old_c_in))
            # weights = multi_weights(mean_weight, incremental_num, is_first=True)
            size_needed = np.random.uniform(np.min(old_weights) / 10, np.max(old_weights) / 10,
                                            (incremental_num, old_c_in))
            new_bias_needed = np.random.uniform(np.min(old_bias) / 10, np.max(old_bias) / 10,
                                                incremental_num)
            cosine_rank = c_rank
            # insert_rank(c_rank)
        else:
            if not is_multi:
                multi = 1
            weights = np.zeros((old_c_out, incremental_num))
            num_needed = incremental_num // multi
            for i in range(num_needed):
                start = cosine_rank[i] * multi
                end = start + multi
                start_i = i * multi
                end_i = start_i + multi
                weights[:, start_i:end_i] = old_weights[:, start:end] * numerator
            # mean_weight = np.reshape(np.mean(weights, axis=1), (old_c_out, 1)) / (incremental_num + 1)
            # weights = multi_weights(mean_weight, incremental_num=incremental_num, is_first=False)
            size_needed = np.random.uniform(np.min(old_weights) / 10, np.max(old_weights) / 10,
                                            (old_c_out, incremental_num))

    base_weight = size_needed + weights
    return base_weight, new_bias_needed


def weight_adaption_rank_group3(old_layer, incremental_num, is_first=True, multi=16, is_multi=False):
    old_weights = old_layer.weight.data.cpu().numpy()
    new_bias_needed = None
    cosine_array = []
    global cosine_rank
    global divide_num
    global meter_list
    global cumulative_layer_index
    numerator = (incremental_num) if divide_num != 2 else divide_num
    if isinstance(old_layer, torch.nn.Conv2d):
        old_c_out, old_c_in, k_h, k_w = old_weights.shape
        old_bias = old_layer.bias.data.cpu().numpy()
        if is_first:
            weights = np.zeros((incremental_num, old_c_in, k_h, k_w))
            rank = get_filter_rank(old_layer)
            rank1_index = rank[0]
            x_weight = old_weights[rank1_index, :, :, :]
            x_weight = np.reshape(x_weight, (old_c_in, k_h * k_w))
            for index in range(old_c_out):
                y_weight = old_weights[index, :, :, :]
                y_weight = np.reshape(y_weight, (old_c_in, k_h * k_w))
                cosine_array.append(cosine_similarity(x_weight, y_weight)[0][0])
            c_rank = get_rank(cosine_array)
            for order in range(incremental_num):
                weights[order] = old_weights[c_rank.index(order)] / numerator
            # mean_weight = np.reshape(np.mean(weights, axis=0), (1, old_c_in, k_h, k_w))
            # weights = multi_weights(mean_weight, incremental_num, is_first=True)
            size_needed = np.random.uniform(np.min(old_weights) / 10, np.max(old_weights) / 10,
                                            (incremental_num, old_c_in, k_h, k_w))
            new_bias_needed = np.random.uniform(np.min(old_bias) / 10, np.max(old_bias) / 10,
                                                incremental_num)
            cosine_rank = c_rank
            insert_rank(c_rank)
        else:
            weights = np.zeros((old_c_out, incremental_num, k_h, k_w))
            for order in range(incremental_num):
                weights[:, order, :, :] = old_weights[:, cosine_rank.index(order), :, :] / numerator
            # mean_weight = np.reshape(np.mean(weights, axis=1), (old_c_out, 1, k_h, k_w)) / (incremental_num + 1)
            # weights = multi_weights(mean_weight, incremental_num, is_first=False)
            size_needed = np.random.uniform(np.min(old_weights) / 10, np.max(old_weights) / 10,
                                            (old_c_out, incremental_num, k_h, k_w))

    elif isinstance(old_layer, torch.nn.Linear):
        old_c_out, old_c_in = old_weights.shape
        old_bias = old_layer.bias.data.cpu().numpy()
        if is_first:
            weights = np.zeros((incremental_num, old_c_in))
            rank = get_filter_rank(old_layer)
            rank1_index = rank[0]
            x_weight = old_weights[rank1_index, :]
            for index in range(old_c_out):
                y_weight = old_weights[index, :]
                top = np.sum(x_weight * y_weight)
                bottom = np.sqrt(np.sum(x_weight ** 2)) * np.sqrt(np.sum(x_weight ** 2))
                cosine_array.append(top / bottom)
            c_rank = get_rank(cosine_array)
            for order in range(incremental_num):
                weights[order] = old_weights[c_rank.index(order)] / numerator
            # mean_weight = np.reshape(np.mean(weights, axis=0), (1, old_c_in))
            # weights = multi_weights(mean_weight, incremental_num, is_first=True)
            size_needed = np.random.uniform(np.min(old_weights) / 10, np.max(old_weights) / 10,
                                            (incremental_num, old_c_in))
            new_bias_needed = np.random.uniform(np.min(old_bias) / 10, np.max(old_bias) / 10,
                                                incremental_num)
            cosine_rank = c_rank
            insert_rank(c_rank)
        else:
            if not is_multi:
                multi = 1
            weights = np.zeros((old_c_out, incremental_num))
            num_needed = incremental_num // multi
            for i in range(num_needed):
                start = cosine_rank[i] * multi
                end = start + multi
                start_i = i * multi
                end_i = start_i + multi
                weights[:, start_i:end_i] = old_weights[:, start:end] / ((numerator + multi) // multi)
            # mean_weight = np.reshape(np.mean(weights, axis=1), (old_c_out, 1)) / (incremental_num + 1)
            # weights = multi_weights(mean_weight, incremental_num=incremental_num, is_first=False)
            size_needed = np.random.uniform(np.min(old_weights) / 10, np.max(old_weights) / 10,
                                            (old_c_out, incremental_num))

    base_weight = size_needed + weights
    return base_weight, new_bias_needed


def weight_adaption_rank_group1(old_layer, incremental_num, is_first=True, rank=None, multi=16, is_multi=False):
    old_weights = old_layer.weight.data.cpu().numpy()
    new_bias_needed = None
    if isinstance(old_layer, torch.nn.Conv2d):
        old_c_out, old_c_in, k_h, k_w = old_weights.shape
        old_bias = old_layer.bias.data.cpu().numpy()
        if is_first:
            weights = np.zeros((incremental_num, old_c_in, k_h, k_w))
            rank = get_filter_rank(old_layer)
            group = group_filters(len(rank), rank, incremental_num)
            for w, index in enumerate(group):
                group_weight = np.zeros((len(index), old_c_in, k_h, k_w))
                for i, item in enumerate(group[0]):
                    group_weight[i] = old_weights[item]
                weights[w] = np.mean(group_weight, axis=0)
            size_needed = np.random.uniform(np.min(old_weights) / 10, np.max(old_weights) / 10,
                                            (incremental_num, old_c_in, k_h, k_w))
            new_bias_needed = np.random.uniform(np.min(old_bias) / 10, np.max(old_bias) / 10,
                                                incremental_num)
        else:
            weights = np.zeros((old_c_out, incremental_num, k_h, k_w))
            group = group_filters(len(rank), rank, incremental_num)
            for w, index in enumerate(group):
                group_weight = np.zeros((old_c_out, len(index), k_h, k_w))
                for i, item in enumerate(group[0]):
                    group_weight[:, i, :, :] = old_weights[:, item, :, :]
                weights[:, w, :, :] = np.mean(group_weight, axis=1) / incremental_num
            size_needed = np.random.uniform(np.min(old_weights) / 10, np.max(old_weights) / 10,
                                            (old_c_out, incremental_num, k_h, k_w))

    elif isinstance(old_layer, torch.nn.Linear):
        old_c_out, old_c_in = old_weights.shape
        old_bias = old_layer.bias.data.cpu().numpy()
        if is_first:
            weights = np.zeros((incremental_num, old_c_in))
            rank = get_filter_rank(old_layer)
            group = group_filters(len(rank), rank, incremental_num)
            for w, index in enumerate(group):
                group_weight = np.zeros((len(index), old_c_in))
                for i, item in enumerate(group[0]):
                    group_weight[i] = old_weights[item]
                weights[w] = np.mean(group_weight, axis=0)
            size_needed = np.random.uniform(np.min(old_weights) / 10, np.max(old_weights) / 10,
                                            (incremental_num, old_c_in))
            new_bias_needed = np.random.uniform(np.min(old_bias) / 10, np.max(old_bias) / 10,
                                                incremental_num)
        else:
            if not is_multi:
                multi = 1
            weights = np.zeros((old_c_out, incremental_num))
            group = group_filters(len(rank), rank, incremental_num // multi)
            for w, index in enumerate(group):
                group_weight = np.zeros((old_c_out, len(index) * multi))
                for i, item in enumerate(group[0]):
                    start = item * multi
                    end = start + multi
                    start_i = i * multi
                    end_i = start_i + multi
                    group_weight[:, start_i:end_i] = old_weights[:, start:end]
                start_w = w * multi
                end_w = start_w + multi
                reshaped_weight = np.reshape(np.mean(group_weight, axis=1), (old_c_out, 1)) / incremental_num
                weights[:, start_w:end_w] = multi_weights(reshaped_weight, incremental_num=multi, is_first=False)
            size_needed = np.random.uniform(np.min(old_weights) / 10, np.max(old_weights) / 10,
                                            (old_c_out, incremental_num))

    base_weight = size_needed + weights
    return base_weight, new_bias_needed


def change_og_based_scale(model, ac_ratio, next_layer_index, add_num=None):
    layers = find_layers(model)
    current_layer = layers[next_layer_index]
    out_layer = layers[next_layer_index - 1]
    # weight = current_layer.weight.data
    ac_ratio = torch.tensor(ac_ratio.astype(np.float32)).to(device)
    # dim = weight.ndim
    # if dim == 2:
    #     ac_ratio = torch.tensor(np.reshape(ac_ratio, (-1, 1)).astype(np.float32)).to(device)
    # elif dim == 4:
    #     ac_ratio = torch.tensor(np.reshape(ac_ratio, (-1, 1, 1, 1)).astype(np.float32)).to(device)
    # else:
    #     raise Exception('the shape is not correct for weight')
    if add_num is not None:
        current_layer.weight.data[:, -add_num:] = current_layer.weight.data[:, -add_num:] * ac_ratio
        current_layer.weight.data[-add_num:, :] = current_layer.weight.data[-add_num:, :] * ac_ratio


# weights adaption
def weight_adaption(old_layer, incremental_num, generate_mode='random', is_first=True, is_multi=False):
    global divide_num
    if generate_mode == 'avg':
        weight, bias = weight_adaption_avg(old_layer, incremental_num, is_first=is_first)

    elif generate_mode == 'random':
        weight, bias = weight_adaption_random(old_layer, incremental_num, is_first=is_first)

    elif generate_mode == 'rank':
        weight, bias = weight_adaption_rank(old_layer, incremental_num, is_first=is_first)

    elif generate_mode == 'rank_baseline':
        weight, bias = weight_adaption_rank_baseline(old_layer, incremental_num, is_first=is_first, is_multi=is_multi)

    elif generate_mode == 'rank_baseline_scale':
        weight, bias = weight_adaption_rank_baseline_scale(old_layer, incremental_num, is_first=is_first,
                                                           is_multi=is_multi)

    elif generate_mode == 'rank_ours':
        weight, bias = weight_adaption_rank_ours(old_layer, incremental_num, is_first=is_first, is_multi=is_multi)

    elif generate_mode == 'rankconnect':
        weight, bias = weight_adaption_rank_connect(old_layer, incremental_num, is_first=is_first,
                                                    is_multi=is_multi)
    elif generate_mode == 'copy_one':
        weight, bias = weight_adaption_copy_one(old_layer, incremental_num, is_first=is_first, is_multi=is_multi)

    elif generate_mode == 'copy_n':
        weight, bias = weight_adaption_copy_n(old_layer, incremental_num, is_first=is_first, is_multi=is_multi)

    elif generate_mode == 'ranklow_n':
        weight, bias = weight_adaption_ranklow_n(old_layer, incremental_num, is_first=is_first, is_multi=is_multi)

    elif generate_mode == 'ranklow_one':
        weight, bias = weight_adaption_ranklow_one(old_layer, incremental_num, is_first=is_first, is_multi=is_multi)

    elif generate_mode == 'rank_cumulative_cosine':
        # weight_adaption_rank_baseline(old_layer, incremental_num, is_first=is_first,
        #                               is_multi=is_multi, ascending=True)
        weight, bias = weight_adaption_rank_cumulative_cosine(old_layer, incremental_num, is_first=is_first,
                                                              is_multi=is_multi)
    elif generate_mode == 'rank_cumulative':
        weight, bias = weight_adaption_rank_cumulative(old_layer, incremental_num, is_first=is_first, is_multi=is_multi)

    elif generate_mode == 'rank_cosine':
        # weight, bias = weight_adaption_rank_group(old_layer, incremental_num, is_first=is_first,
        #                                           is_multi=is_multi)
        weight, bias = weight_adaption_rank_cosine(old_layer, incremental_num, is_first=is_first,
                                                   is_multi=is_multi)
    # elif generate_mode == 'random':
    #     weight, bias = weight_adaption_random(old_layer, incremental_num, is_first=is_first)

    elif generate_mode == 'activation_low':
        weight, bias = weight_adaption_activation_low(old_layer, incremental_num, is_first=is_first, is_multi=is_multi)

    elif generate_mode == 'rank_one':
        weight, bias = weight_adaption_rank_one(old_layer, incremental_num, is_first=is_first, is_multi=is_multi)

    else:
        raise Exception('The generation mode is not found')

    return weight, bias


def multi_weights(weights, incremental_num, is_first=True):
    new_weights = weights
    if is_first:
        dim = 0
    else:
        dim = 1
    for i in range(incremental_num - 1):
        new_weights = np.concatenate((new_weights, weights), axis=dim)
    return new_weights


def test_batch(model, data, target):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    model = model.to(device)
    with torch.no_grad():
        _, output = model(data)
        test_loss = criterion(output, target)  # sum up batch loss
    return test_loss.item()


def replace_layers(model_classifer, layer_index, layer_index_change, layers_change):
    # if the layer in the layer_index is needed to changed, then use layers_change to replace,
    # otherwise return the original ones.
    if layer_index in layer_index_change:
        return layers_change[layer_index_change.index(layer_index)]
    return model_classifer[layer_index]


def weight_in_out(model, output, incremental_num, name_of_param='classifier.2.weight', a=0.3, ):
    # process previous_x, calculate mean without 0 element
    previous = output.data.cpu().numpy()
    # previous[np.where(previous == 0)] = np.nan
    # previous_output = np.nanmean(previous, axis=0)
    previous_output = np.mean(previous, axis=0)
    next_gradient = None
    next_weights = None
    current_weights = None

    for name, p in model.named_parameters():
        # get the gradient from the last layer and average, then it becomes shape 10
        # Note that it mean 10 connections to 10 neurons in last layer
        if name == name_of_param:
            # get the gradient tensor
            grad_tensor = p.grad.data.cpu().numpy()

            # average to 10 gradient
            next_gradient = np.mean(grad_tensor, axis=1)
            # grad_tensor[np.where(grad_tensor == 0)] = np.nan
            # next_gradient = np.nanmean(grad_tensor, axis=1)

            # next_weights
            next_weights = p.data.cpu().numpy()
        elif name == 'classifier.0.weight':
            current_weights = p.data.cpu().numpy()

    n = previous_output.shape[0]
    m = next_gradient.shape[0]
    gradient_matrix = np.zeros((n, m))
    for row in range(n):
        for col in range(m):
            gradient_matrix[row, col] = previous_output[row] * next_gradient[col]
    gradient_matrix_abs = np.sqrt(np.abs(gradient_matrix))
    sgn_matrix = np.sign(gradient_matrix)
    random_matrix = np.random.uniform(-1, 1, n * m)
    random_matrix = np.reshape(random_matrix, (n, m))
    g_matrix = gradient_matrix_abs * random_matrix

    # calculate w_in, w_out
    w_in = np.sum(g_matrix * sgn_matrix, axis=1)
    w_out = np.sum(g_matrix, axis=0)

    # multiply the a
    w_out = w_out * a * numpy_avg(np.abs(next_weights)) / numpy_avg(np.abs(w_out))
    w_in = w_in * a * numpy_avg(np.abs(current_weights)) / numpy_avg(np.abs(w_in))

    w_in = np.reshape(w_in, (1, -1))
    w_out = np.reshape(w_out, (-1, 1))

    # w_in, w_out need to further process
    # w_in and w_out times incremental numbers
    w_in = multi_weights(w_in, incremental_num, is_first=True)
    w_out = multi_weights(w_out, incremental_num, is_first=False)

    return w_in, w_out


def weights_divide(old_weights, incremental_num, is_multi=False, multi=16, twoD=False, mode='rankgroup'):
    global cosine_rank
    global random_num
    global importance_rank
    global divide_seed
    global rank_scale
    # global cumulative_rank

    if divide_seed == "non-scale":
        seed = 1
    else:
        seed = incremental_num if divide_seed != 2 else 2
    if not is_multi:
        multi = 1
    incremental_num = incremental_num // multi

    numerator = 1 - (incremental_num / (old_weights.shape[1] + incremental_num))
    numerator = numerator / 2

    for i in range(incremental_num):
        if mode == 'rank_one':
            break
        if mode == 'rankgroup':
            order = cosine_rank[i]
        # elif mode == 'randomMap':
        #     order = random_num
        #     seed = 2
        elif mode == 'randomMap' or mode == 'copy_n':
            seed = 2
            order = random_num[i]
        elif mode == 'copy_one':
            seed = incremental_num + 1
            order = random_num
            start = order * multi
            end = start + multi
            old_weights[:, start:end] = old_weights[:, start:end] / seed
            break
        elif mode == 'rankconnect':
            order = importance_rank[i]
        elif mode == 'rank_baseline':
            order = importance_rank[i]
            numerator = 1 / 2
        elif mode == 'rank_baseline_scale':
            order = importance_rank[i]
            numerator = rank_scale
        elif mode == 'ranklow_one':
            order = importance_rank[i]
        else:
            raise Exception('the generation mode is not found')
        start = order * multi
        end = start + multi
        # old_weights[:, start:end] = old_weights[:, start:end] / seed
        old_weights[:, start:end] = old_weights[:, start:end] * numerator

    if mode == 'rank_one':
        seed = 2
        order = importance_rank[0]
        start = order * multi
        end = start + multi
        old_weights[:, start:end] = old_weights[:, start:end] / seed

    return old_weights


def weights_divide_first(old_weights, incremental_num, multi=1, mode='rankconnect'):
    global cosine_rank
    global random_num
    global importance_rank
    global divide_seed
    global rank_scale

    if divide_seed == "non-scale":
        seed = 1
    else:
        seed = incremental_num if divide_seed != 2 else 2

    # numerator = 1 - (incremental_num / (old_weights.shape[1] + incremental_num))

    for i in range(incremental_num):
        if mode == 'rank_one':
            break
        if mode == 'rankgroup':
            order = cosine_rank[i]
        elif mode == 'randomMap':
            order = random_num[i]
        elif mode == 'rankconnect':
            order = importance_rank[i]
        elif mode == 'rank_baseline':
            numerator = 1 / 2
            order = importance_rank[i]
        elif mode == 'rank_baseline_scale':
            numerator = rank_scale
            order = importance_rank[i]
        start = order * multi
        end = start + multi
        # old_weights[start:end] = old_weights[start:end] / seed
        old_weights[start:end] = old_weights[start:end] * numerator
    if mode == 'rank_one':
        seed = 2
        order = importance_rank[0]
        start = order * multi
        end = start + multi
        old_weights[start:end] = old_weights[start:end] / seed

    return old_weights


def conv_layer_grow(old_conv_layer, incremental_num, is_first=True, mode='rank'):
    if old_conv_layer is None:
        raise BaseException("No Conv layer found in classifier")
    old_weights = old_conv_layer.weight.data.cpu().numpy()
    old_c_out, old_c_in, k_h, k_w = old_weights.shape
    old_bias = old_conv_layer.bias.data.cpu().numpy()
    old_bias_size = old_bias.shape[0]
    mode_list = ['random']
    # if mode not in mode_list:
    weights, bias = weight_adaption(old_conv_layer, incremental_num, generate_mode=mode, is_first=is_first)
    old_grad = old_conv_layer.weight.grad.data.cpu().numpy()

    # create new_conv_layer
    if is_first:
        # if the old weight is also needed to change
        f_mode_list = ['rank_baseline', 'rank_one', 'rank_baseline_scale']  # , 'rank_cumulative'
        if mode in f_mode_list:
            old_weights = weights_divide_first(old_weights, incremental_num, mode=mode)
        new_conv_layer = torch.nn.Conv2d(in_channels=old_conv_layer.in_channels,
                                         out_channels=old_conv_layer.out_channels + incremental_num,
                                         kernel_size=old_conv_layer.kernel_size,
                                         stride=old_conv_layer.stride,
                                         padding=old_conv_layer.padding,
                                         dilation=old_conv_layer.dilation,
                                         groups=old_conv_layer.groups,
                                         bias=(old_conv_layer.bias is not None))

        # the first layer need to change the bias
        new_bias = new_conv_layer.bias.data.cpu().numpy()
        new_bias[:old_bias_size] = old_bias

        # weights
        new_weights = new_conv_layer.weight.data.cpu().numpy()
        new_weights[:old_c_out, :old_c_in, :, :] = old_weights

        if mode not in mode_list:
            new_bias[old_bias_size:] = bias
            new_weights[old_c_out:, :, :, :] = weights
            new_grad = np.zeros(new_weights.shape).astype(np.float32)
            new_grad[:old_c_out, :old_c_in, :, :] = old_grad

    else:
        # if mode == 'randomMap' or mode == 'rankgroup':
        m_list = ['rank_baseline', 'rank_baseline_scale', 'copy_n']  # , 'rank_one', 'randomMap', 'copy_one',
        # m_list = ['rank_baseline', 'randomMap', 'copy_one', 'copy_n', 'rank_cumulative']
        if mode in m_list:
            old_weights = weights_divide(old_weights, incremental_num, twoD=False, mode=mode)

        new_conv_layer = torch.nn.Conv2d(in_channels=old_conv_layer.in_channels + incremental_num,
                                         out_channels=old_conv_layer.out_channels,
                                         kernel_size=old_conv_layer.kernel_size,
                                         stride=old_conv_layer.stride,
                                         padding=old_conv_layer.padding,
                                         dilation=old_conv_layer.dilation,
                                         groups=old_conv_layer.groups,
                                         bias=(old_conv_layer.bias is not None))
        # copy the weights from old_linear_layer , the new filter use init weight
        new_bias = new_conv_layer.bias.data.cpu().numpy()
        new_bias[:old_bias_size] = old_bias
        new_weights = new_conv_layer.weight.data.cpu().numpy()
        new_weights[:old_c_out, :old_c_in, :, :] = old_weights

        if mode not in mode_list:
            new_weights[:, old_c_in:, :, :] = weights
            new_grad = np.zeros(new_weights.shape).astype(np.float32)
            new_grad[:old_c_out, :old_c_in, :, :] = old_grad

    # copy new bias to new_conv_layer
    new_conv_layer.bias.data = torch.from_numpy(new_bias)
    new_conv_layer.bias.data = new_conv_layer.bias.data.to(device)

    # copy new weights to new_conv_layer
    new_conv_layer.weight.data = torch.from_numpy(new_weights)
    new_conv_layer.weight.data = new_conv_layer.weight.data.to(device)

    # copy old_grad to new_conv_layer
    if mode not in mode_list:
        grad = old_conv_layer.weight.grad
        grad.data = torch.from_numpy(new_grad).to(device)
        new_conv_layer.weight.grad = grad

    return new_conv_layer


def fc_layer_grow(old_linear_layer, incremental_num, weights=None, is_first=True, mode='rank',
                  is_multi=False):
    if old_linear_layer is None:
        raise BaseException("No linear layer found in classifier")

    old_weights = old_linear_layer.weight.data.cpu().numpy()
    old_bias = old_linear_layer.bias.data.cpu().numpy()
    old_bias_size = old_bias.shape[0]
    old_out, old_in = old_weights.shape
    mode_list = ['random', 'bridging']
    if weights is None:
        # if mode not in mode_list:
        weights, bias = weight_adaption(old_linear_layer, incremental_num, generate_mode=mode, is_first=is_first,
                                        is_multi=is_multi)
        old_grad = old_linear_layer.weight.grad.data.cpu().numpy()
    else:
        _, bias = weight_adaption(old_linear_layer, incremental_num, is_first=is_first)
    # create new_linear_layer
    if is_first:
        f_mode_list = ['rank_baseline', 'rank_baseline_scale', 'rank_one']  # , 'rank_cumulative'
        if mode in f_mode_list:
            old_weights = weights_divide_first(old_weights, incremental_num, mode=mode)
        new_linear_layer = torch.nn.Linear(old_linear_layer.in_features,
                                           old_linear_layer.out_features + incremental_num)
        # copy the weights from old_linear_layer and add grow weights
        if weights is not None:
            # if mode not in mode_list:
            new_weights = np.vstack((old_weights, weights)).astype(np.float32)
        else:
            new_weights = new_linear_layer.weight.data.cpu().numpy()
            new_weights[:old_out, :] = old_weights

        # the first layer need to change the bias
        new_bias = new_linear_layer.bias.data.cpu().numpy()
        new_bias[:old_bias_size] = old_bias

        # copy the bias and old grad to new layer
        if mode not in mode_list:
            new_bias[old_bias_size:] = bias
            new_grad = np.zeros(new_weights.shape).astype(np.float32)
            new_grad[:old_out, :] = old_grad
    else:
        new_linear_layer = torch.nn.Linear(old_linear_layer.in_features + incremental_num,
                                           old_linear_layer.out_features)
        # if mode == 'randomMap' or mode == 'rankgroup':
        m_list = ['rank_baseline', 'rank_baseline_scale', 'copy_n']  # , 'rank_one', 'randomMap', 'copy_one'
        # m_list = ['rank_baseline', 'randomMap', 'copy_one', 'copy_n', 'rank_cumulative']
        if mode in m_list:
            old_weights = weights_divide(old_weights, incremental_num, is_multi, twoD=True, mode=mode)
        # copy the weights from old_linear_layer and add grow weights
        if weights is not None:
            # if mode not in mode_list:
            new_weights = np.hstack((old_weights, weights)).astype(np.float32)
        else:
            new_weights = new_linear_layer.weight.data.cpu().numpy()
            new_weights[:, :old_in] = old_weights
        new_bias = new_linear_layer.bias.data.cpu().numpy()
        new_bias[:old_bias_size] = old_bias
        # copy the old grad to new layers
        # if mode not in mode_list and mode != 'bridging':
        if mode not in mode_list:
            new_grad = np.zeros(new_weights.shape).astype(np.float32)
            new_grad[:, :old_in] = old_grad

    out_channel, in_channel = new_linear_layer.weight.data.cpu().numpy().shape

    # assert the new_weights's shape must be same
    assert new_weights.shape[0] == out_channel and new_weights.shape[
        1] == in_channel, 'the new weights shape in linear layer 1 is incorrect'

    # copy new bias to new_linear_layer
    new_linear_layer.bias.data = torch.from_numpy(new_bias)
    new_linear_layer.bias.data = new_linear_layer.bias.data.to(device)
    # copy new weights to new_linear_layer
    new_linear_layer.weight.data = torch.from_numpy(new_weights)
    new_linear_layer.weight.data = new_linear_layer.weight.data.to(device)
    # copy old grad  to new_linear_layer
    # if mode not in mode_list and mode != 'bridging':
    if mode not in mode_list:
        grad = old_linear_layer.weight.grad
        grad.data = torch.from_numpy(new_grad).to(device)
        new_linear_layer.weight.grad = grad

    return new_linear_layer


def numpy_avg(np_array, axis=None):
    np_array[np.where(np_array == 0)] = np.nan
    if axis is not None:
        output = np.nanmean(np_array, axis=axis)
    else:
        output = np.nanmean(np_array)
    return output


def grow_one_neuron(model, output, incremental_num, mode='rank'):
    # the first linear layer need to be changed
    layer_index = 0
    old_linear_layer = None
    # find old Linear layer
    for _, module in model.classifier._modules.items():
        if isinstance(module, torch.nn.Linear):
            old_linear_layer = module
            break
        layer_index = layer_index + 1

    if mode == 'bridging':
        # get w_in, w_out
        w_in, w_out = weight_in_out(model, output, incremental_num)
        new_linear_layer = fc_layer_grow(old_linear_layer, incremental_num, weights=w_in, mode='bridging')
    else:
        # rank = get_filter_rank(old_linear_layer)
        new_linear_layer = fc_layer_grow(old_linear_layer, incremental_num, mode=mode)

    # the second linear layer need to be changed
    layer_index2 = 0
    for layer, module in model.classifier._modules.items():
        if isinstance(module, torch.nn.Linear) and int(layer) > layer_index:
            old_linear_layer2 = module
            layer_index2 = int(layer)

    if mode == 'bridging':
        new_linear_layer2 = fc_layer_grow(old_linear_layer2, incremental_num, weights=w_out, is_first=False,
                                          mode='bridging')
    else:
        new_linear_layer2 = fc_layer_grow(old_linear_layer2, incremental_num, is_first=False, mode=mode)

    # put two new generated layer to classifier
    classifier = torch.nn.Sequential(
        *(replace_layers(model.classifier, i, [layer_index, layer_index2],
                         [new_linear_layer, new_linear_layer2]) for i, _ in enumerate(model.classifier)))
    del model.classifier
    model.classifier = classifier

    return model


def create_layers(first_layer, second_layer, incremental_num):
    new_first_layer = conv_layer_grow(first_layer, incremental_num, mode='random')
    if isinstance(second_layer, torch.nn.Conv2d):
        new_second_layer = conv_layer_grow(second_layer, incremental_num, is_first=False, mode='random')
    else:
        new_second_layer = fc_layer_grow(second_layer, incremental_num * 16, weights=None,
                                         is_first=False, mode='random')
    return new_first_layer, new_second_layer


def replace_multiple_layers(model, first_layer, first_index, second_layer, second_index):
    if isinstance(second_layer, torch.nn.Conv2d):
        features = torch.nn.Sequential(
            *(replace_layers(model.features, i, [first_index, second_index],
                             [first_layer, second_layer]) for i, _ in enumerate(model.features)))
        model.features = features
        return model
    else:
        features = torch.nn.Sequential(
            *(replace_layers(model.features, i, [first_index],
                             [first_layer]) for i, _ in enumerate(model.features)))
        classifier = torch.nn.Sequential(
            *(replace_layers(model.classifier, i, [second_index],
                             [second_layer]) for i, _ in enumerate(model.classifier)))
        model.features = features
        model.classifier = classifier
        return model


def grow_one_filter(model, layer_index, incremental_num, data, target, num=10):
    first_layer_index = 0
    old_conv_layer = None
    # global is_bn
    # find old conv layer
    for index_str, module in model.features._modules.items():
        if isinstance(module, torch.nn.Conv2d) and int(index_str) == layer_index:
            old_conv_layer = module
            break
        first_layer_index = first_layer_index + 1

    # the second conv layer need to be changed
    second_layer_index = 0
    old_conv_layer2 = None
    old_linear_layer = None
    for layer, module in model.features._modules.items():
        if isinstance(module, torch.nn.Conv2d) and int(layer) > first_layer_index:
            old_conv_layer2 = module
            second_layer_index = int(layer)
            break

    if old_conv_layer2 is None:
        for layer, module in model.classifier._modules.items():
            if isinstance(module, torch.nn.Linear):
                old_linear_layer = module
                second_layer_index = int(layer)
                break
    # if is_bn:
    #     model = replace_bn_layers(model, layer_index + 1, incremental_num)
    losses = []
    first_layers = []
    second_layers = []

    # start to compare each model with different layer
    # this function need to random check them ten times to get two best layers by comparing loss values.
    for i in range(num):
        if old_conv_layer2 is not None:
            new_first_layer, new_second_layer = create_layers(old_conv_layer, old_conv_layer2, incremental_num)
        else:
            new_first_layer, new_second_layer = create_layers(old_conv_layer, old_linear_layer, incremental_num)
        model = replace_multiple_layers(model, new_first_layer, first_layer_index, new_second_layer,
                                        second_layer_index)
        loss = test_batch(model, data, target)
        first_layers.append(new_first_layer)
        second_layers.append(new_second_layer)
        losses.append(loss)

    min_loss = losses[0]
    index = 0
    for i, l in enumerate(losses):
        if l < min_loss:
            min_loss = l
            index = i
    new_first_layer = first_layers[index]
    new_second_layer = second_layers[index]
    # then return the model which has the lowest loss value
    model = replace_multiple_layers(model, new_first_layer, first_layer_index, new_second_layer, second_layer_index)

    # if old_conv_layer2 is None
    # create a new fc1 layer

    # above code need to be in a function and return two layers and layer index.

    # this function need to random check them ten times to get two best layers by comparing loss values.

    return model


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


def replace_optimizer(model, optimizer, learning_rate, avg=False, scale_list=None, is_scale=False,
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


def grow_filters(model, layer_index, incremental_num, mode='rank'):
    first_layer_index = 0
    old_conv_layer = None
    # find old conv layer
    for index_str, module in model.features._modules.items():
        if isinstance(module, torch.nn.Conv2d) and int(index_str) == layer_index:
            old_conv_layer = module
            break
        first_layer_index = first_layer_index + 1

    # the second conv layer need to be changed
    second_layer_index = 0
    old_conv_layer2 = None
    old_linear_layer = None
    for layer, module in model.features._modules.items():
        if isinstance(module, torch.nn.Conv2d) and int(layer) > first_layer_index:
            old_conv_layer2 = module
            second_layer_index = int(layer)
            break

    if old_conv_layer2 is None:
        for layer, module in model.classifier._modules.items():
            if isinstance(module, torch.nn.Linear):
                old_linear_layer = module
                second_layer_index = int(layer)
                break

    new_first_layer = conv_layer_grow(old_conv_layer, incremental_num, is_first=True, mode=mode)

    if old_conv_layer2 is not None:
        new_second_layer = conv_layer_grow(old_conv_layer2, incremental_num, is_first=False, mode=mode)
    else:
        new_second_layer = fc_layer_grow(old_linear_layer, incremental_num * 16, is_first=False, mode=mode,
                                         is_multi=True)

    model = replace_multiple_layers(model, new_first_layer, first_layer_index, new_second_layer, second_layer_index)

    # if old_conv_layer2 is None
    # create a new fc1 layer

    # above code need to be in a function and return two layers and layer index.

    # this function need to random check them ten times to get two best layers by comparing loss values.

    return model


def change_model(model, mode='rankgroup', layer_index=0, incremental_num=2, inputs=None, target=None,
                 features_output=None, is_con=True):
    mode_list = ['bridging']
    global is_bn
    if mode not in mode_list and is_con:
        n_model = grow_filters(model, layer_index=layer_index, incremental_num=incremental_num, mode=mode)
        if is_bn:
            n_model = replace_bn_layers(n_model, index=layer_index + 1, incremental_num=incremental_num)
    # elif mode == 'random' and is_con:
    #     if is_bn:
    #         model = replace_bn_layers(model, index=layer_index + 1, incremental_num=incremental_num)
    #     n_model = grow_one_filter(model, layer_index=layer_index, incremental_num=incremental_num, data=inputs,
    #                               target=target, num=1)
    elif mode == 'bridging' and is_con:
        if is_bn:
            model = replace_bn_layers(model, index=layer_index + 1, incremental_num=incremental_num)
        n_model = grow_one_filter(model, layer_index=layer_index, incremental_num=incremental_num, data=inputs,
                                  target=target)

    else:
        n_model = grow_one_neuron(model, features_output, incremental_num=incremental_num, mode=mode)

    return n_model


def loss_learning(new_model, old_loss, inputs, target, criterion, to_log, l=0.3):
    write_log('batch loss before grow: {}\n'.format(old_loss), to_log)
    features_output, output = new_model(inputs)
    new_loss = criterion(output, target)
    write_log('batch loss after grow: {}\n'.format(new_loss), to_log)
    global lr
    optimizer = torch.optim.Adam(new_model.parameters(), lr=lr)
    new_loss_data = np.min(new_loss.data.cpu().numpy()).astype(np.float32)
    old_loss_data = np.min(old_loss.data.cpu().numpy()).astype(np.float32)
    if new_loss_data > old_loss_data:
        new_loss.data = torch.from_numpy(np.asarray([new_loss_data - l * old_loss_data]))
        new_loss.data = new_loss.data.to(device)
        new_loss.backward()
        optimizer.step()
        features_output, output = new_model(inputs)
        learn_loss = criterion(output, target)
        write_log('batch loss after learning: {}\n'.format(learn_loss), to_log)
    return new_model


# def gen_diff_lr_dic(old_m, new_m, new_lr=2):
#     # old_layers = find_layers(old_m)
#     # new_layers = find_layers(new_m)
#     dic = {}
#     for old_p, new_p in zip(list(old_m.parameters()), list(new_m.parameters())):
#         old_t = torch.ones_like(old_p)
#         new_t = torch.empty_like(new_p).fill_(new_lr)
#         if old_t.ndim < 2:
#             new_t[:old_p.shape[0]] = old_t
#         else:
#             new_t[:old_p.shape[0], :old_p.shape[1]] = old_t
#         dic[new_p] = new_t
#     return dic


def gen_diff_lr_dic(old_shape_list, new_m, new_lr=2):
    # old_layers = find_layers(old_m)
    # new_layers = find_layers(new_m)
    dic = {}
    for old_shape, new_p in zip(old_shape_list, list(new_m.parameters())):
        old_t = torch.ones(old_shape)
        new_t = torch.empty_like(new_p).fill_(new_lr)
        if old_t.ndim < 2:
            new_t[:old_shape[0]] = old_t
        else:
            new_t[:old_shape[0], :old_shape[1]] = old_t
        dic[new_p] = new_t
    return dic


def replace_optimizerW(model, optimizer, learning_rate, avg=True, lr_dic=None):
    optimizer = optimizer
    # new_optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    new_optimizer = AdamW(model.parameters(), lr=learning_rate, lr_dic=lr_dic)
    for current_group, new_group in zip(optimizer.param_groups, new_optimizer.param_groups):
        for current_p, new_p in zip(current_group['params'], new_group['params']):
            current_state = optimizer.state[current_p]
            state = new_optimizer.state[new_p]
            # State initialization
            state['step'] = current_state['step']
            # Exponential moving average of gradient values
            state['exp_avg'] = torch.zeros_like(new_p.data, memory_format=torch.preserve_format)
            new_exp_avg = state['exp_avg'].data.cpu().numpy()
            current_exp_avg = current_state['exp_avg'].data.cpu().numpy()
            new_exp_avg = put_to_ndarray(current_exp_avg, new_exp_avg)

            # Exponential moving average of squared gradient values
            state['exp_avg_sq'] = torch.zeros_like(new_p.data, memory_format=torch.preserve_format)
            new_exp_avg_sq = state['exp_avg_sq'].data.cpu().numpy()
            current_exp_avg_sq = current_state['exp_avg_sq'].data.cpu().numpy()
            new_exp_avg_sq = put_to_ndarray(current_exp_avg_sq, new_exp_avg_sq)

            if avg:
                mean_exp_avg = np.mean(current_exp_avg)
                mean_exp_avg_sq = np.mean(current_exp_avg_sq)
                new_exp_avg[new_exp_avg == 0.0] = mean_exp_avg
                new_exp_avg_sq[new_exp_avg_sq == 0.0] = mean_exp_avg_sq

            state['exp_avg'].data = torch.from_numpy(new_exp_avg.astype(np.float32))
            state['exp_avg'].data = state['exp_avg'].data.to(device)
            state['exp_avg_sq'].data = torch.from_numpy(new_exp_avg_sq.astype(np.float32))
            state['exp_avg_sq'].data = state['exp_avg_sq'].data.to(device)

    return new_optimizer


def train(train_loader, model, to_log, to_test, test_loader=None, grow_interval=1, grow_con_interval=300000000,
          print_freq=100, epoch=0):
    global mode_name
    global dic
    global dic_change
    global optimizer_mode
    global is_bn
    global activation_rank
    global is_lr
    global hook_dic
    global is_gate
    global batch_test_acc
    global existing_index
    global existing_index_in
    global s_score
    global s_score_in
    global old_shape
    global gate_step
    global save_image
    global is_grow
    global is_reg
    global batch_time_list
    global start_time
    global grow_epoch
    global layer_index_list
    global is_record_diagnosis
    global optimizer
    global lr
    global recorder
    global lambda_list
    hist_dic = {}

    # first time
    first_time = True

    # create optimizer and critierion
    # initialize critierion
    criterion = nn.CrossEntropyLoss()

    if is_reg:
        regularizer_loss = L2Regularizer(model=model, lambda_reg=1e-05)

    # create multi average meters
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    train_loss = []

    # switch to train mode
    model.train()

    # register acv hook
    register_acv_hook(model)

    # record start time
    batch_timer.start()
    start = time.time()

    for i_batch, (inputs, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - start)

        # put data into available devices
        inputs, target = inputs.to(device), target.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # gradient and do SGD step
        model = model.train()

        features_output, output = model(inputs)
        loss = criterion(output, target)

        if is_reg:
            loss = regularizer_loss.regularized_all_param(reg_loss_function=loss)

        train_loss.append(loss.item())
        loss.backward()

        # gradient decay
        if is_gate and existing_index:
            gradient_lambda_decay(model)
            # gradient_lambda_decay_select(model, lambda_list)

        optimizer.step()

        # cumulative_score
        put_cumulative_score(model)

        # Place is where the grow happens
        # if (epoch - 1) in grow_epoch and first_time:
        if (epoch - 1) % 3 ==0 and first_time:
            # save activation before grow
            save_acv()

            batch_timer.pause()
            epoch_timer.pause()
            # start grow timer
            grow_timer.start()

            # get current weight shape for each layer
            if is_lr:
                old_shape = lr_size_each_layer(model)

            old_size = model_size(find_layers(model))
            existing_index = old_size[:3]
            # existing_index = old_size

            if is_gate:
                remove_hook(hook_dic)
            remove_acv_hook()

            dic[epoch] = {}
            dic_change[epoch] = {}

            # before
            if save_image:
                layers = find_layers(model)
                save_weight_to_dic(layers, hist_dic, key='before')

            ac = caluclulate_activation(model, inputs, 0, epoch=epoch, channel='channel_before',
                                        print_weights=True)

            for i, layer_index in enumerate(layer_index_list):
                model.eval()
                conv = True if i < 2 else False
                # conv = True
                write_log('Epoch {}, Before layer {} grow:\n'.format(epoch, i), to_test)
                channel_before = 'B{}'.format(layer_index[0])
                dic[epoch][channel_before] = {}

                # save histgram figures before growing
                # S score compare new out and new in
                # each epoch S score
                is_scale = False

                grow_timer.pause()
                test(model, test_loader=test_loader, criterion=criterion, to_log=to_test,
                     is_two_out=True)
                grow_timer.resume()

                if is_bn:
                    caluclulate_activation1(model, inputs, to_test, epoch=epoch, channel=channel_before,
                                            print_weights=True)
                else:
                    acac = caluclulate_activation(model, inputs, i + 1, epoch=epoch, channel=channel_before,
                                                  print_weights=True)
                    write_log('\nAc_old in L{} : {} \n'.format(i + 1, acac.tolist()), to_test)
                    ac_old = ac[i + 1]
                    if is_scale:
                        write_log('\nAc_old in L{} : {} \n'.format(i + 1, ac_old.tolist()), to_test)
                calculate_score(model, inputs, target, epoch=epoch, channel=channel_before, print_weights=True)
                features_output, output = model(inputs)

                # record layer size
                layer_size = model_size(find_layers(model))
                global cumulative_layer_index
                cumulative_layer_index = i
                # save the current layer activation to global variable
                activation_rank = ac[i]

                model = change_model(model, mode=mode_name, layer_index=layer_index[0],
                                     incremental_num=layer_index[1], inputs=inputs,
                                     target=target, features_output=features_output, is_con=conv)

                # after model change, save next layer activation
                # get scale ratio for each channel
                # after grow record the in and out index
                record_index(layer=i, grow_epoch=epoch, incremental_num=layer_index[1], layer_size=layer_size)
                model = model.to(device)
                write_log('Epoch {}, After layer {} grow:\n'.format(epoch, i), to_test)
                channel_after = 'A{}'.format(layer_index[0])
                dic[epoch][channel_after] = {}

                grow_timer.pause()
                test(model, test_loader=test_loader, criterion=criterion, to_log=to_test,
                     is_two_out=True)
                grow_timer.resume()

                if is_bn:
                    caluclulate_activation1(model, inputs, to_test, epoch=epoch, channel=channel_after,
                                            print_weights=True)
                    module_names = ['features.0', 'features.5', 'classifier.0']
                else:
                    ac_new = caluclulate_activation(model, inputs, i + 1, epoch=epoch, channel=channel_after,
                                                    print_weights=True)
                    write_log('\nAc_new in L{} : {} \n'.format(i + 1, ac_new.tolist()), to_test)
                    module_names = ['features.0', 'features.3', 'classifier.0']

                calculate_score(model, inputs, target, epoch=epoch, channel=channel_after, print_weights=True)

                if is_scale:
                    ac_ratio = ac_old / (ac_new + 0.00005)
                    ac_ratio[(ac_ratio > 1) | (ac_ratio < 0)] = 1
                    ac_ratio = np.mean(ac_ratio)
                    # change the weight scale
                    write_log('\nScale in L{} : {} \n'.format(i + 1, ac_ratio.tolist()), to_test)
                    change_og_based_scale(model, ac_ratio, i + 1, layer_index[1])
                    ac_scale = caluclulate_activation(model, inputs, i + 1, epoch=epoch, channel=channel_after,
                                                      print_weights=True)
                    write_log('\nAc_scale in L{} : {} \n'.format(i + 1, ac_scale.tolist()), to_test)

            # only add weight_decay
            # scale_list = weight_decay(model, existing_index)

            # weight decay
            if is_gate and existing_index:
                scale_list = weight_decay(model, existing_index)
                # scale_list = weight_decay_select(model, existing_index, lambda_list)
                # lambda_list = []
                test(model, test_loader=test_loader, criterion=criterion, to_log=to_test,
                     is_two_out=True)

            new_size = model_size(find_layers(model))

            # register hook
            register_acv_hook(model)

            if is_gate:
                hook_dic = register_hook(model, module_names, old_size[:3], new_size[:3], scale_list=scale_list)

            # guarantee gradient
            features_output, output = model(inputs)  # save acv after grow
            loss = criterion(output, target)
            loss.backward()

            first_time = False
            model = model.to(device)
            model = model.train()
            reset_cumulative_score()

            # update optimizer as the shape of weights is changed
            if optimizer_mode == 'SGD':
                optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
            else:
                optimizer = replace_optimizer(model, optimizer, lr, avg=False, optimizer_mode=optimizer_mode)
                # optimizer = AdamW(model.parameters(), lr=lr)

            # after
            if save_image:
                layers = find_layers(model)
                save_weight_to_dic(layers, hist_dic, key='after')

            grow_timer.stop()
            # resume tiemrs
            batch_timer.resume()
            epoch_timer.resume()

        # record in and out and og value is in each batch epoch
        recorder.record(model)

        if is_record_diagnosis:
            batch_timer.pause()
            epoch_timer.pause()
            record_score(model, i_batch, epoch, optimizer=optimizer)
            batch_timer.resume()
            epoch_timer.resume()

        if is_gate and not first_time:
            if i_batch == 0:

                # shape_hook_tensor(existing_index, s_score, is_first=True)

                # shape_hook_tensor_max(existing_index, s_score, is_first=True)

                # shape_hook_tensor_auto(existing_index, s_score, is_first=True, step=i_batch)

                # shape_hook_tensor_autoMax(existing_index, s_score, is_first=True, step=i_batch, total_step=gate_step)

                # shape_hook_tensor_auto_add(existing_index, is_first=True, step=i_batch, total_step=gate_step)

                # shape_hook_tensor_auto_exsit(existing_index, s_score, is_first=True, step=i_batch, total_step=gate_step)

                test(model, test_loader=test_loader, criterion=criterion, to_log=to_test,
                     is_two_out=True)
                pass
            else:

                # shape_hook_tensor(existing_index, s_score, is_first=False)

                # shape_hook_tensor_max(existing_index, s_score, is_first=True)

                # shape_hook_tensor_max(existing_index, s_score, is_first=True)

                if i_batch < gate_step:
                    # if i_batch == 149:
                    #     print('', end='')
                    shape_hook_tensor_auto_converge(existing_index, step=i_batch, total_step=gate_step)
                    # shape_hook_tensor_auto(existing_index, s_score, is_first=False, step=i_batch, total_step=gate_step)
                    # shape_hook_tensor_autoMax(existing_index, s_score, is_first=False, step=i_batch, total_step=gate_step)
                    # shape_hook_tensor_auto_add(existing_index, is_first=False, step=i_batch, total_step=gate_step)
                    # shape_hook_tensor_auto_exsit(existing_index, s_score, is_first=False, step=i_batch,
                    #                              total_step=gate_step)
                    pass
                elif gate_step < i_batch < 599:
                    # shape_hook_tensor_auto_decay(existing_index, step=i_batch, total_step=599)
                    pass
                elif i_batch == 599:
                    # print('', end='')
                    # remove_hook(hook_dic)
                    pass
            record_gate(epoch, i_batch)

        if is_lr and not first_time:
            # if is_lr and epoch == 3:
            if i_batch == gate_step:
                lr_dic = init_channel_lr(old_shape, model, fill=2.0)
                # lr_dic = init_channel_lr_exist(old_shape, model)
                # lr_dic = init_channel_lr(old_shape, model, s_score, s_score_in, is_first=True)
                # lr_dic = init_channel_lr_out(old_shape, model, s_score, s_score_in, is_first=True)
                # lr_dic = init_channel_lr_out_exist(old_shape, model, s_score, s_score_in, is_first=True)
                # optimizer = AdamW(model.parameters(), lr=lr, lr_dic=lr_dic)
                # optimizer = replace_optimizer(model, optimizer, lr, avg=False, optimizer_mode='AdamW')
                optimizer.lr_dic = lr_dic
            elif gate_step < i_batch < 599:
                lr_dic = auto_decay_lr_exist(old_shape, step=i_batch, total_step=599)
                # lr_dic = gen_diff_lr_channel_dic(old_shape, model, s_score, s_score_in, step=i_batch,
                #                                  total_step=gate_step)
                # lr_dic = gen_diff_lr_channel_dic_out(old_shape, model, s_score, s_score_in, step=i_batch,
                #                                  total_step=gate_step)
                # lr_dic = gen_diff_lr_channel_dic_out_exist(old_shape, model, s_score, s_score_in, step=i_batch,
                #                                            total_step=gate_step)

                optimizer.lr_dic = lr_dic
                pass
            elif i_batch == 599:
                optimizer.lr_dic = None
            record_lr(epoch, i_batch)

        if save_image and i_batch == gate_step and not first_time:
            layers = find_layers(model)
            save_weight_to_dic(layers, hist_dic, key='finish')
            folder_path = os.path.dirname(to_log)
            gen_hist_from_dic(hist_dic, path=folder_path, epoch=epoch)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - start)
        start = time.time()

        if (i_batch + 1) % print_freq == 0:
            output_str = ('Epoch: [{0}][{1}/{2}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:3.3f} ({top1.avg:3.3f})\t'
                          'Prec@5 {top5.val:3.3f} ({top5.avg:3.3f})'.format(
                epoch, (i_batch + 1), len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
            print(output_str)
            write_log(output_str + '\n', to_log)

        epoch_timer.pause()
        # batch time saved into the list
        if (i_batch + 1) % print_freq == 0:
            b_time = (time.time() - start_time)
            batch_time_list.append(b_time)
            batch_timer.stop()
            # _, b_acc = test(model, test_loader=test_loader, criterion=criterion, is_two_out=True)
            # batch_test_acc.append(b_acc)
            if (i_batch + 1) < 600:
                batch_timer.start()
        epoch_timer.resume()

    return train_loss


def record_index(layer, grow_epoch, incremental_num, layer_size, multi=16):
    global dic_score
    global dic_change
    global mode_name
    global random_num
    global importance_rank
    global cosine_rank
    global activation_rank
    global cumulative_rank

    # if mode_name == 'randomMap' or mode_name == 'copy_one' or mode_name == 'copy_n':
    #     order_list = random_num
    # if mode_name == 'rankconnect' or mode_name == 'rank_baseline' or mode_name == 'ranklow_one' or mode_name == 'rank_ours':
    #     order_list = importance_rank
    # if mode_name == 'rankgroup' or mode_name == 'rank_cumulative_cosine':
    #     order_list = cosine_rank

    # record rank list

    dic_change[grow_epoch][layer] = {}

    if isinstance(random_num, np.int32):
        dic_change[grow_epoch][layer]["random"] = str(random_num)
    else:
        if len(random_num) > 0:
            dic_change[grow_epoch][layer]["random"] = str(random_num)

    if len(importance_rank) > 0:
        dic_change[grow_epoch][layer]["s_rank"] = str(importance_rank)

    if len(cosine_rank) > 0:
        dic_change[grow_epoch][layer]["cosine_rank"] = str(cosine_rank)

    if len(cumulative_rank) > 0:
        dic_change[grow_epoch][layer]["cumulative_rank"] = str(cumulative_rank)

    if not isinstance(activation_rank, list):
        if len(activation_rank) > 0:
            dic_change[grow_epoch][layer]["activation_rank"] = str(activation_rank.tolist())
    else:
        if len(activation_rank) > 0:
            dic_change[grow_epoch][layer]["activation_rank"] = str(activation_rank)

    # for i in range(incremental_num):
    #     if 'out' in dic_change[layer]:
    #         dic_change[layer]["out"].append(layer_size[layer] + i)
    #     if 'in' in dic_change[layer + 1]:
    #         in_order_list = [((layer_size[layer] + i) * multi + ele) for ele in range(multi)]
    #         dic_change[layer + 1]["in"] += in_order_list
    #     if mode_name == 'bridging' or mode_name == 'copy_one' or mode_name == 'copy_n' or mode_name == 'random' or mode_name == 'ranklow_n' \
    #             or mode_name == 'activation_low':
    #         continue
    #     if 'og' in dic_change[layer + 1]:
    #         dic_change[layer + 1]["og"] = {}
    #         dic_change[layer + 1]["og"]["cosin_rank"]
    #         # if layer != 1:  # the num is fixed so far
    #         #     multi = 1
    #         # if mode_name == 'randomMap':
    #         #     seed = order_list[i]
    #         # else:
    #         #     seed = order_list[i]
    #         # og_order_list = [(seed * multi + ele) for ele in range(multi)]
    #         dic_change[layer + 1]["og"] += og_order_list


def update_weights_by_activation(model, current_activation):
    pass


def train_test(model, trainloader, testloader, learning_rate, epoch, dic_path, str_channel):
    global optimizer
    global is_scheduler
    # initialize criterion
    criterion = nn.CrossEntropyLoss()

    # initialize scheduler
    if not is_grow and is_scheduler:
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # start time
    global start_time
    global batch_time_list
    global batch_test_acc
    global hook_dic
    global lr
    start_time = time.time()

    # loss , acc, time
    test_loss = []
    test_acc = []
    batch_loss = []
    train_loss_mean = []
    train_loss_std = []
    time_in_epoch = []

    for e in range(epoch):
        # train
        epoch_timer.start()

        train_loss = train(trainloader, model, test_loader=testloader, epoch=(e + 1), to_log=dic_path['path_to_log'],
                           to_test=dic_path['path_to_test'])

        # write and delete
        with open(path['path_to_json1'], 'a') as f:
            json.dump(dic_score, f, indent=4)

        if (e+1) in dic_score:
            del dic_score[(e+1)]

        if not is_grow and is_scheduler:
            exp_lr_scheduler.step()

        if is_grow and is_scheduler and (e + 1) % 3 == 0:
            lr = lr * 0.8

        epoch_timer.stop()
        batch_loss.append(train_loss)
        train_loss_mean.append(np.mean(batch_loss))
        train_loss_std.append(np.std(batch_loss))

        # test after train
        loss, acc = test(model, test_loader=testloader, criterion=criterion, to_log=dic_path['path_to_log'],
                         is_two_out=True)

        # if e == 2:
        #     remove_hook(hook_dic)
        #     h_dic = register_hook_delete(model)
        #     print('add old hook to temporarily delete the existing channel:')
        #     test(model, test_loader=testloader, criterion=criterion, to_log=dic_path['path_to_log'],
        #          is_two_out=True)
        #     remove_hook(h_dic)
        #
        #     h_dic = register_hook_delete(model, old_fill=1, new_fill=0)
        #     print('add old hook to temporarily delete the new channel:')
        #     test(model, test_loader=testloader, criterion=criterion, to_log=dic_path['path_to_log'],
        #          is_two_out=True)
        #     remove_hook(h_dic)

        # assert acc > 90 or e < 3, "it is not good, restart"

        test_loss.append(loss)
        test_acc.append(acc)
        time_in_epoch.append(time.time() - start_time)

        # save model for each epoch
        # path_to_model_epoch = os.path.join(dic_path['res_dir'], ('model_E{}_{}.pkl'.format(e, str_channel)))
        # save_model(model=model, path_to_model=path_to_model_epoch, mode='entire')

    # total time
    total_time = time.time() - start_time

    # save final model
    save_model(model=model, path_to_model=(dic_path['path_to_model']).format(str=str_channel), mode='part')

    # the size of parameters
    para_str = 'The number of parameters in the model is : {}\n'.format(count_parameters(model))

    # total time
    time_str = 'The total training time is {}\n'.format(total_time)

    # output
    print(time_str)
    print(para_str)

    # save log
    write_log(time_str, dic_path['path_to_log'])
    write_log(para_str, dic_path['path_to_log'])
    write_log('train time in each epoch:{}\n'.format(time_in_epoch), dic_path['path_to_log'])
    write_log('batch time:{}\n'.format(batch_time_list), dic_path['path_to_log'])
    write_log('batch test acc:{}\n'.format(batch_test_acc), dic_path['path_to_log'])
    write_log('Loss:{}\n'.format(test_loss), dic_path['path_to_log'])
    write_log('Accuracy:{}\n'.format(test_acc), dic_path['path_to_log'])
    write_log('Train_loss_mean:{}\n'.format(train_loss_mean), dic_path['path_to_log'])
    write_log('Train_loss_std:{}\n'.format(train_loss_std), dic_path['path_to_log'])
    for index, item in enumerate(batch_loss):
        str = '\n{}_Epoch_{} = {};\n'.format(str_channel, index + 1, item)
        write_log(str, dic_path['path_to_log'])


if __name__ == '__main__':
    param = {
        'epochs': 100,
        'lr': 0.0005,
        'batch_size': 100
    }
    # initialize timers
    batch_timer = Timer(name='batch')
    grow_timer = Timer(name='grow')
    epoch_timer = Timer(name='epoch')

    is_record_diagnosis = True
    is_grow = False
    is_reg = False
    is_bn = False
    is_lr = False
    save_image = False
    is_gate = False
    is_scheduler = False
    optimizer_mode = 'AdamW'
    gate_step = 150
    lr = param['lr']
    # data set name
    dataset = 'MNIST'
    # dataset = 'EMG'
    # mode_list = ['avg','rankconnect','rankgroup','randomMap','bridging']

    if is_grow:
        grow_epoch = [1, 3, 6, 9]
        # grow_epoch = [1]
    else:
        grow_epoch = []

    if is_bn:
        layer_index_list = [(0, 2), (4, 3), (5, 10)]
    else:
        # layer_index_list = [(0, 2), (3, 3)]
        layer_index_list = [(0, 2), (3, 3), (5, 10)]

    args = {'out1': 2, 'out2': 5, 'fc1': 10}
    # args = {'out1': 4, 'out2': 8, 'fc1': 20}
    # args = {'out1': 6, 'out2': 11, 'fc1': 30}
    # args = {'out1': 8, 'out2': 14, 'fc1': 40}
    grow_mode_list = [
        # 'bridging',
        'rank_baseline',
        # 'random',
        # 'copy_n',
        # 'standard',
        # 'rank_ours',
        # 'rank_cosine',
        # 'rank_baseline_scale',
        # 'rank_cumulative',
        # 'random',
        # 'ranklow_n',
        # 'ranklow_one',
        # 'rank_cumulative_cosine',
        # 'rank_cumulative',
    ]
    # seed_list = [2]
    # num_list = [2]

    std_para_list = [
        # {'out1': 2, 'out2': 5, 'fc1': 10},
        # {'out1': 4, 'out2': 8, 'fc1': 20},
        # {'out1': 6, 'out2': 11, 'fc1': 30},
        # {'out1': 8, 'out2': 14, 'fc1': 40},
        # {'out1': 10, 'out2': 17, 'fc1': 50},

        # rank-baseline  Paper
        # {'out1': 8, 'out2': 17, 'fc1': 23},

        # 10% , 30%, 60%, 100%
        {'out1': 2, 'out2': 5, 'fc1': 50},
        {'out1': 6, 'out2': 15, 'fc1': 150},
        {'out1': 12, 'out2': 30, 'fc1': 300},
        {'out1': 20, 'out2': 50, 'fc1': 500},
    ]

    if is_grow:
        loop_list = grow_mode_list
    else:
        loop_list = std_para_list

    for mode in loop_list:
        para = mode
        # path_name`  y
        dic = {}
        # divide_num = num
        # divide_seed = seed
        if is_bn:
            path_name = 'grow_{}_BN_'.format(mode)
        elif not is_grow:
            path_name = 'standard_{}_{}_{}_'.format(para['out1'], para['out2'], para['fc1'])
        else:
            # path_name = 'grow_{}_og{}_new{}_'.format(mode, seed, num)
            path_name = 'grow_{}_'.format(mode)
        mode_name = mode
        # create model
        # Note that only the model returns two outputs is acceptable

        # set torch and numpy random seed
        # torch.manual_seed(1535)
        # np.random.seed(1535)

        # Create model
        if not is_grow:
            model = LeNet5_GROW_P(**para)
        elif is_bn:
            model = LeNet5_GROW_BN()
        else:
            model = LeNet5_GROW_P(**args)

        # init model weights
        # model.apply(weights_init)
        model = model.to(device)

        recorder = Recorder(model)

        # create path
        path = dir_path(path_name)
        res_dir = path['res_dir']

        # optimizer
        if optimizer_mode == 'AdamW':
            optimizer = AdamW(model.parameters(), lr=lr)
        elif optimizer_mode == 'Adam':
            optimizer = Adam(model.parameters(), lr=lr)
        elif optimizer_mode == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        else:
            raise Exception('the optimizer {} is not implemented'.format(optimizer_mode))
        # load data
        trainloader, testloader = generate_data_loader(batch_size=param['batch_size'], dataset=dataset,
                                                       is_main=False)

        try:
            train_test(model, trainloader, testloader, param['lr'], param['epochs'], path,
                       str_channel=path_name)
        except AssertionError as error:
            print(error)
            break

        # remove hook
        remove_hook(hook_dic)
        remove_gate_max()
        existing_index = []
        # dump dic to datafile
        with open(path['path_to_json'], 'w') as f:
            json.dump(dic, f, indent=4)
        # with open(path['path_to_json1'], 'w') as f:
        #     json.dump(dic_score, f, indent=4)
        with open(path['path_to_json_dic'], 'w') as f:
            json.dump(dic_change, f, indent=4)
        with open(path['path_to_timer'], 'w') as f:
            json.dump(batch_timer.timers, f, indent=4)
        save_gate_dic_to_json(path)
        save_lr_dic_to_json(path)
        # need to check the gradient shape and the x output shape in two layers.
        # debugger to see what is inside and to see how pytorch does those things
        meter_list = [AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()]
