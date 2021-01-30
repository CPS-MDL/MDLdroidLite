import torch
import numpy as np
import json

from utils import check_dic_key, find_layers
from utils import device
from grow.activation_hook import return_preacv_dic, retrun_acv_dic

hook_tensor = [None, None, None]
max_gate = [None, None, None]
delete_gate = [None, None, None]
dic_gate = {}

#########################


def hook_maker(shape, index, dim=4, fill=2):
    # global hook_tensor
    global max_gate
    # pre_acv = return_preacv_dic()
    # cur_acv = retrun_acv_dic()

    # old scale, new is 1
    if max_gate[index] is None:
        t_old = torch.empty(shape[0]).fill_(fill)
        t_new = torch.empty(shape[1]).fill_(1)
        t_new[:shape[0]] = t_old
    else:
        t_old = torch.flatten(max_gate[index] * fill)
        t_new = torch.empty(shape[1]).fill_(1)
        t_new[:shape[0]] = t_old

    # new scale, old is 1
    # if max_gate[index] is None:
    #     t_old = torch.empty(shape[0]).fill_(1)
    #     t_new = torch.empty(shape[1]).fill_(fill)
    #     t_new[:shape[0]] = t_old
    # else:
    #     t_old = torch.flatten(max_gate[index] * fill)
    #     t_new = torch.empty(shape[1]).fill_(1)
    #     t_new[:shape[0]] = t_old
    # old_tensor = max_gate[index]
    # old_tensor = torch.ones(shape[0])
    # if old_tensor is not None:
    #     # old tensor times 2
    #     old_tensor = old_tensor * 2
    #     t_ones = torch.ones_like(old_tensor)
    #     old_tensor = torch.where(old_tensor > 1, t_ones, old_tensor)
    # else:
    #     old_tensor = t_old
    # t_new = torch.empty(shape[1]).fill_(fill)
    # old_tensor = torch.reshape(old_tensor, (-1,))
    # t_new[:shape[0]] = old_tensor

    if dim == 4:
        t_new = torch.reshape(t_new, (1, -1, 1, 1))
        t_new = t_new.requires_grad_(False)
    else:
        t_new = torch.reshape(t_new, (1, -1))
        t_new = t_new.requires_grad_(False)

    t_new = t_new.to(device)
    max_gate[index] = t_new

    def hook1(self, input, output):
        output.mul_(max_gate[0])
        # print(output)
        # hook1.t = hook1.t + 0.001
        # x = torch.ones_like(hook1.t)
        # hook1.t = torch.where(hook1.t > 1, x, hook1.t)

    def hook2(self, input, output):
        # if 0 in output:
        #     print('yes')
        output.mul_(max_gate[1])
        # print(output)
        # hook2.t = hook2.t + 0.001
        # x = torch.ones_like(hook2.t)
        # hook2.t = torch.where(hook2.t > 1, x, hook2.t)

    def hook3(self, input, output):
        # if 0 in output:
        #     print('yes')
        output.mul_(max_gate[2])
        # print(output)
        #     hook3.t = t_new
        # output.mul_(hook3.t)
        # hook3.t = hook3.t + 0.001
        # x = torch.ones_like(hook3.t)
        # hook3.t = torch.where(hook3.t > 1, x, hook3.t)

    if index == 0:
        return hook1
    if index == 1:
        return hook2
    if index == 2:
        return hook3


def gradient_lambda_decay(model):
    global max_gate
    layers = find_layers(model)
    for i, layer in enumerate(layers[:-1]):
        # for i, layer in enumerate(layers):
        dim = layer.weight.grad.data.dim()
        if dim == 4:
            grad_lambda = torch.reshape(max_gate[i], (-1, 1, 1, 1))
            # grad_lambda = t_new.requires_grad_(False)
        else:
            grad_lambda = torch.reshape(max_gate[i], (-1, 1))
            # grad_lambda = t_new.requires_grad_(False)
        layer.weight.grad.data = layer.weight.grad.data / grad_lambda


def gradient_lambda_decay_select(model, lambda_list):
    global max_gate
    layers = find_layers(model)
    for i, layer in enumerate(layers[:-1]):
        dim = layer.weight.grad.data.dim()
        if dim == 4:
            grad_lambda = torch.reshape(max_gate[i], (-1, 1, 1, 1))
            # grad_lambda = t_new.requires_grad_(False)
        else:
            grad_lambda = torch.reshape(max_gate[i], (-1, 1))
            # grad_lambda = t_new.requires_grad_(False)
        layer.weight.grad.data = layer.weight.grad.data / grad_lambda


#
# def hook_maker(shape, index, dim=4, fill=0.00):
#     # global hook_tensor
#     global max_gate
#     old_tensor = max_gate[index]
#     t_old = torch.ones(shape[0])
#     if old_tensor is not None:
#         # old tensor times 2
#         old_tensor = old_tensor * 2
#         t_ones = torch.ones_like(old_tensor)
#         old_tensor = torch.where(old_tensor > 1, t_ones, old_tensor)
#     else:
#         old_tensor = t_old
#     t_new = torch.empty(shape[1]).fill_(fill)
#     old_tensor = torch.reshape(old_tensor, (-1,))
#     t_new[:shape[0]] = old_tensor
#     if dim == 4:
#         t_new = torch.reshape(t_new, (1, -1, 1, 1))
#         t_new = t_new.requires_grad_(False)
#     else:
#         t_new = torch.reshape(t_new, (1, -1))
#         t_new = t_new.requires_grad_(False)
#
#     t_new = t_new.to(device)
#     max_gate[index] = t_new
#
#     def hook1(self, input, output):
#         output.mul_(max_gate[0])
#         # print(output)
#         # hook1.t = hook1.t + 0.001
#         # x = torch.ones_like(hook1.t)
#         # hook1.t = torch.where(hook1.t > 1, x, hook1.t)
#
#     def hook2(self, input, output):
#         # if 0 in output:
#         #     print('yes')
#         output.mul_(max_gate[1])
#         # print(output)
#         # hook2.t = hook2.t + 0.001
#         # x = torch.ones_like(hook2.t)
#         # hook2.t = torch.where(hook2.t > 1, x, hook2.t)
#
#     def hook3(self, input, output):
#         # if 0 in output:
#         #     print('yes')
#         output.mul_(max_gate[2])
#         # print(output)
#         #     hook3.t = t_new
#         # output.mul_(hook3.t)
#         # hook3.t = hook3.t + 0.001
#         # x = torch.ones_like(hook3.t)
#         # hook3.t = torch.where(hook3.t > 1, x, hook3.t)
#
#     if index == 0:
#         return hook1
#     if index == 1:
#         return hook2
#     if index == 2:
#         return hook3
#

# def hook_maker(shape, index, dim=4, fill=0):
#     t_old = torch.ones(shape[0])
#     t_new = torch.empty(shape[1]).fill_(fill)
#     t_new[:shape[0]] = t_old
#     if dim == 4:
#         t_new = torch.reshape(t_new, (1, -1, 1, 1))
#         t_new = t_new.requires_grad_(False)
#     else:
#         t_new = torch.reshape(t_new, (1, -1))
#         t_new = t_new.requires_grad_(False)
#
#     def hook1(self, input, output):
#         if not hasattr(hook1, 't'):
#             hook1.t = t_new.clone()
#         output.mul_(hook1.t)
#         # hook1.t = hook1.t + 0.001
#         # x = torch.ones_like(hook1.t)
#         # hook1.t = torch.where(hook1.t > 1, x, hook1.t)
#
#     def hook2(self, input, output):
#         if not hasattr(hook2, 't'):
#             hook2.t = t_new
#         output.mul_(hook2.t)
#         # hook2.t = hook2.t + 0.001
#         # x = torch.ones_like(hook2.t)
#         # hook2.t = torch.where(hook2.t > 1, x, hook2.t)
#
#     def hook3(self, input, output):
#         if not hasattr(hook3, 't'):
#             hook3.t = t_new
#         output.mul_(hook3.t)
#         # hook3.t = hook3.t + 0.001
#         # x = torch.ones_like(hook3.t)
#         # hook3.t = torch.where(hook3.t > 1, x, hook3.t)
#
#     if index == 0:
#         return hook1
#     if index == 1:
#         return hook2
#     if index == 2:
#         return hook3


def shape_hook_tensor(existing_index, s_score, is_rank=False, is_first=False):
    # global hook_tensor
    global max_gate
    dim = [4, 4, 2]
    for layer, ex_index in enumerate(existing_index):
        if max_gate[layer] is None:
            break
        # if layer > 1:
        #     break
        e_score = s_score[layer][:ex_index]
        new_score = s_score[layer][ex_index:]
        if is_rank:
            score = np.max(e_score)
        else:
            score = np.mean(e_score)

        # add a small seed if score is zero
        score = 0.0005 if score == 0 else score
        gate_score = np.asarray(new_score) / score
        t_old = torch.ones(ex_index)
        t_new = torch.from_numpy(gate_score.astype(np.float32))
        t_ones = torch.ones_like(t_new)
        t_new = torch.where(t_new > 1, t_ones, t_new)
        t_new = torch.cat((t_old, t_new), dim=0)

        if dim[layer] == 4:
            t_new = torch.reshape(t_new, (1, -1, 1, 1))
            t_new = t_new.requires_grad_(False)
        else:
            t_new = torch.reshape(t_new, (1, -1))
            t_new = t_new.requires_grad_(False)
        t_new = t_new.to(device)

        # if is_first: # or layer > 1
        #     ratio = 1 - (len(new_score) / len(s_score[layer]))
        #     t_new[:, ex_index:] = t_new[:, ex_index:] * ratio
        #     # t_ratio = t_new * ratio
        #     # t_new = torch.where(t_new < 1, t_ratio, t_new)
        # else:
        #     t_new = torch.max(t_new, max_gate[layer])

        # t_new = t_new.to(device)
        # hook_tensor[layer] = t_new
        max_gate[layer] = t_new


def shape_hook_tensor_auto_add(existing_index, is_first=False, step=1, total_step=300):
    # global hook_tensor
    global max_gate

    for layer, ex_index in enumerate(existing_index):
        if max_gate[layer] is None:
            break
        if is_first:
            break
        t = max_gate[layer]
        t_ones = torch.ones_like(t)
        t_left = 1 - t
        t_step = total_step - step
        t_decay = t_left / t_step
        t = t + t_decay
        t_new = torch.where(t > 1, t_ones, t)
        t_new = t_new.to(device)
        max_gate[layer] = t_new


def shape_hook_tensor_auto_converge1(existing_index, step=1, total_step=300):
    # global hook_tensor
    global max_gate

    for layer, ex_index in enumerate(existing_index):
        t = max_gate[layer]
        t_left = t - 1
        t_step = total_step - step
        t_decay = t_left / t_step
        # max_gate[layer][:, :ex_index] = max_gate[layer][:, :ex_index] - t_decay
        # max_gate[layer][:, ex_index:] = max_gate[layer][:, ex_index:] + t_decay
        max_gate[layer] = max_gate[layer] - t_decay


def shape_hook_tensor_auto_converge2(existing_index, step=1, total_step=300):
    # global hook_tensor
    global max_gate

    # if step >= total_step // 2:
    #     return 0
    for layer, ex_index in enumerate(existing_index):
        t = max_gate[layer]
        scale_old = torch.flatten(t)[0]
        scale_new = torch.flatten(t)[-1]
        t_left = scale_old - scale_new
        t_step = total_step - step
        t_decay = t_left / t_step
        t_decay_old = t_decay / 2
        t_decay_new = t_decay / 2
        max_gate[layer][:, :ex_index] = max_gate[layer][:, :ex_index] - t_decay_old
        max_gate[layer][:, ex_index:] = max_gate[layer][:, ex_index:] + t_decay_new


def shape_hook_tensor_auto_converge(existing_index, step=1, total_step=300):
    # global hook_tensor
    global max_gate

    # if step >= total_step // 2:
    #     return 0
    for layer, ex_index in enumerate(existing_index):
        t = max_gate[layer]

        # get new and old scale
        length = len(torch.flatten(t))
        ns = ex_index
        os = length - ex_index
        # os = ex_index
        # ns = length - ex_index

        # get old and new decay
        scale_old = torch.flatten(t)[0]
        scale_new = torch.flatten(t)[-1]
        t_left = scale_old - scale_new
        t_step = total_step - step
        t_decay = t_left / t_step
        t_decay_old = t_decay / (ns + os) * os
        t_decay_new = t_decay / (ns + os) * ns

        # old and new decay
        max_gate[layer][:, :ex_index] = max_gate[layer][:, :ex_index] - t_decay_old
        max_gate[layer][:, ex_index:] = max_gate[layer][:, ex_index:] + t_decay_new


def shape_hook_tensor_auto_decay(existing_index, step=150, total_step=600):
    # global hook_tensor
    global max_gate

    for layer, ex_index in enumerate(existing_index):
        t = max_gate[layer]
        scale_old = torch.flatten(t)[0]
        scale_new = torch.flatten(t)[-1]
        t_left_old = scale_old - 1
        t_left_new = scale_new - 1
        t_step = total_step - step
        t_decay_old = t_left_old / t_step
        t_decay_new = t_left_new / t_step
        max_gate[layer][:, :ex_index] = max_gate[layer][:, :ex_index] - t_decay_old
        max_gate[layer][:, ex_index:] = max_gate[layer][:, ex_index:] - t_decay_new

        # if step == total_step:
        #     max_gate[layer].fill_(1)


def shape_hook_tensor_max(existing_index, s_score, is_rank=False, is_first=False):
    # global hook_tensor
    global max_gate
    dim = [4, 4, 2]
    for layer, ex_index in enumerate(existing_index):
        if max_gate[layer] is None:
            break
        # if layer > 1:
        #     break
        e_score = s_score[layer][:ex_index]
        new_score = s_score[layer][ex_index:]
        if is_rank:
            score = np.max(e_score)
        else:
            score = np.mean(e_score)

        # add a small seed if score is zero
        score = 0.0005 if score == 0 else score
        gate_score = np.asarray(new_score) / score
        t_old = torch.ones(ex_index)
        t_new = torch.from_numpy(gate_score.astype(np.float32))
        t_ones = torch.ones_like(t_new)
        t_new = torch.where(t_new > 1, t_ones, t_new)
        t_new = torch.cat((t_old, t_new), dim=0)

        if dim[layer] == 4:
            t_new = torch.reshape(t_new, (1, -1, 1, 1))
            t_new = t_new.requires_grad_(False)
        else:
            t_new = torch.reshape(t_new, (1, -1))
            t_new = t_new.requires_grad_(False)
        t_new = t_new.to(device)
        if is_first:  # or layer > 1
            pass
            # ratio = 1 - (len(new_score) / len(s_score[layer]))
            # t_new[:, ex_index:] = t_new[:, ex_index:] #* ratio
            # t_ratio = t_new * ratio
            # t_new = torch.where(t_new < 1, t_ratio, t_new)
        else:
            t_new = torch.max(t_new, max_gate[layer])
        # t_new = t_new.to(device)
        # hook_tensor[layer] = t_new
        max_gate[layer] = t_new


def shape_hook_tensor_auto(existing_index, s_score, is_rank=False, is_first=False, step=1, total_step=300):
    # global hook_tensor
    global max_gate
    dim = [4, 4, 2]
    for layer, ex_index in enumerate(existing_index):
        if max_gate[layer] is None:
            break
        # if layer > 1:  # fully remain no change
        #     break
        e_score = s_score[layer][:ex_index]
        new_score = s_score[layer][ex_index:]
        if is_rank:
            score = np.max(e_score)
        else:
            score = np.mean(e_score)

        # add a small seed if score is zero
        score = 0.0005 if score == 0 else score
        gate_score = np.asarray(new_score) / score
        t_old = torch.ones(ex_index)
        t_new = torch.from_numpy(gate_score.astype(np.float32))
        t_ones = torch.ones_like(t_new)
        t_new = torch.where(t_new > 1, t_ones, t_new)
        t_new = torch.cat((t_old, t_new), dim=0)

        if dim[layer] == 4:
            t_new = torch.reshape(t_new, (1, -1, 1, 1))
            t_new = t_new.requires_grad_(False)
        else:
            t_new = torch.reshape(t_new, (1, -1))
            t_new = t_new.requires_grad_(False)
        t_new = t_new.to(device)
        if is_first:
            pass
            # t_new[:, ex_index:] = t_new[:, ex_index:]
            # t_ratio = t_new * ratio
            # t_new = torch.where(t_new < 1, t_ratio, t_new)
        else:
            grow_length = 1 - max_gate[layer]
            grow_step = grow_length / (total_step - step)
            t_new = max_gate[layer] + grow_step

        # t_new = t_new.to(device)
        # hook_tensor[layer] = t_new
        max_gate[layer] = t_new


def shape_hook_tensor_auto_exsit(existing_index, s_score, is_first=False, step=1, total_step=300):
    exsit_init = 0.5
    exsit_final = 1.0
    # global hook_tensor
    # exsit 1.5 -> 1.0
    #       1.0 -> 0.5
    global max_gate
    dim = [4, 4, 2]
    for layer, ex_index in enumerate(existing_index):
        if max_gate[layer] is None:
            break

        if is_first:
            e_score = s_score[layer][:ex_index]
            new_score = s_score[layer][ex_index:]

            t_old = torch.empty(len(e_score)).fill_(exsit_init)
            t_new = torch.ones(len(new_score))
            t_new = torch.cat((t_old, t_new), dim=0)

            if dim[layer] == 4:
                t_new = torch.reshape(t_new, (1, -1, 1, 1))
                t_new = t_new.requires_grad_(False)
            else:
                t_new = torch.reshape(t_new, (1, -1))
                t_new = t_new.requires_grad_(False)
        else:
            grow_length = exsit_final - max_gate[layer]
            grow_step = grow_length / (total_step - step)
            t_new = max_gate[layer] + grow_step

        t_new = t_new.to(device)

        max_gate[layer] = t_new


def shape_hook_tensor_autoMax(existing_index, s_score, is_rank=False, is_first=False, step=1, total_step=300):
    # global hook_tensor
    global max_gate
    dim = [4, 4, 2]
    for layer, ex_index in enumerate(existing_index):
        if max_gate[layer] is None:
            break
        # if layer > 1:  # fully remain no change
        #     break
        e_score = s_score[layer][:ex_index]
        new_score = s_score[layer][ex_index:]
        if is_rank:
            score = np.max(e_score)
        else:
            score = np.mean(e_score)

        # add a small seed if score is zero
        score = 0.0005 if score == 0 else score
        gate_score = np.asarray(new_score) / score
        t_old = torch.ones(ex_index)
        t_new = torch.from_numpy(gate_score.astype(np.float32))
        t_ones = torch.ones_like(t_new)
        t_new = torch.where(t_new > 1, t_ones, t_new)
        t_new = torch.cat((t_old, t_new), dim=0)

        if dim[layer] == 4:
            t_new = torch.reshape(t_new, (1, -1, 1, 1))
            t_new = t_new.requires_grad_(False)
        else:
            t_new = torch.reshape(t_new, (1, -1))
            t_new = t_new.requires_grad_(False)
        t_new = t_new.to(device)
        if is_first:
            pass
            # ratio = 1 - (len(new_score) / len(s_score[layer]))
            # t_new[:, ex_index:] = t_new[:, ex_index:] * ratio
            # t_ratio = t_new * ratio
            # t_new = torch.where(t_new < 1, t_ratio, t_new)
        else:
            grow_length = 1 - max_gate[layer]
            grow_step = grow_length / (total_step - step)
            max_gate[layer] = max_gate[layer] + grow_step
            t_new = torch.max(t_new, max_gate[layer])
        # t_new = t_new.to(device)
        # hook_tensor[layer] = t_new
        max_gate[layer] = t_new


def scale_gate(arr, min=0.2, max=0.8):
    interval = (max - min) / (len(arr) - 1)
    # arr = arr * -1
    sort = arr.argsort()
    for i in range(len(arr)):
        arr[sort[i]] = min + i * interval
    return arr


#  get grow ratio
def shape_hook_tensor_rank(existing_index, s_score, is_rank=False):
    global hook_tensor
    dim = [4, 4, 2]
    for layer, ex_index in enumerate(existing_index):
        e_score = s_score[layer][:ex_index]
        new_score = s_score[layer][ex_index:]
        # if is_rank:
        #     score = np.max(e_score)
        # else:
        #     score = np.mean(e_score)
        # score = 0.0005 if score == 0 else score
        # gate_score = np.asarray(new_score) / score

        # use rank to generate gate
        gate_score = scale_gate(np.asarray(new_score))

        t_old = torch.ones(ex_index)
        t_new = torch.from_numpy(gate_score.astype(np.float32))
        t_new = torch.cat((t_old, t_new), dim=0)

        if dim[layer] == 4:
            t_new = torch.reshape(t_new, (1, -1, 1, 1))
            t_new = t_new.requires_grad_(False)
        else:
            t_new = torch.reshape(t_new, (1, -1))
            t_new = t_new.requires_grad_(False)
        t_new = t_new.to(device)
        hook_tensor[layer] = t_new


def register_hook_old(model, names, old_size, new_size):
    dic_size = {}

    for key, o, n in zip(names, old_size, new_size):
        dic_size[key] = (o, n)
    hook_dic = {}
    i = 0
    for name, module in model.named_modules():
        if name in names:
            if name.startswith('fea'):
                d = 4
            else:
                d = 2
            hook_dic[name] = module.register_forward_hook(hook_maker(dic_size[name], i, dim=d))
            i = i + 1
    return hook_dic


def register_hook(model, names, old_size, new_size, scale_list):
    dic_size = {}

    for key, o, n, s in zip(names, old_size, new_size, scale_list):
        dic_size[key] = (o, n, s)
    hook_dic = {}
    i = 0
    for name, module in model.named_modules():
        if name in names:
            if name.startswith('fea'):
                d = 4
            else:
                d = 2
            hook_dic[name] = module.register_forward_hook(hook_maker(dic_size[name], i, dim=d, fill=dic_size[name][2]))
            i = i + 1
    return hook_dic


def register_hook_delete(model, old_size=[], old_fill=0, new_size=[], new_fill=1):
    names = ['features.0', 'features.3', 'classifier.0']
    dic_size = {}
    # old_size = [2, 5, 10]
    # new_size = [0, 0, 10]
    for key, o, n in zip(names, old_size, new_size):
        dic_size[key] = (o, n)
    hook_dic = {}
    i = 0
    for name, module in model.named_modules():
        if name in names:
            if name.startswith('fea'):
                d = 4
            else:
                d = 2
            hook_dic[name] = module.register_forward_hook(
                hook_maker_delete(dic_size[name], i, dim=d, old_fill=old_fill, new_fill=new_fill))
            i = i + 1
    return hook_dic


def acv_hook(self, input, output):
    pass


def hook_maker_delete(shape, index, dim=4, old_fill=0, new_fill=1):
    # global hook_tensor
    global delete_gate
    # old_tensor = max_gate[index]
    old_tensor = torch.empty(shape[0]).fill_(old_fill)
    # if old_tensor is not None:
    #     # old tensor times 2
    #     old_tensor = old_tensor * 2
    #     t_ones = torch.ones_like(old_tensor)
    #     old_tensor = torch.where(old_tensor > 1, t_ones, old_tensor)
    # else:
    #     old_tensor = t_old
    t_new = torch.empty((shape[1] + shape[0])).fill_(new_fill)
    old_tensor = torch.reshape(old_tensor, (-1,))
    t_new[:shape[0]] = old_tensor

    if dim == 4:
        t_new = torch.reshape(t_new, (1, -1, 1, 1))
        t_new = t_new.requires_grad_(False)
    else:
        t_new = torch.reshape(t_new, (1, -1))
        t_new = t_new.requires_grad_(False)

    t_new = t_new.to(device)
    delete_gate[index] = t_new

    def hook1(self, input, output):
        output.mul_(delete_gate[0])
        # print(output)
        # hook1.t = hook1.t + 0.001
        # x = torch.ones_like(hook1.t)
        # hook1.t = torch.where(hook1.t > 1, x, hook1.t)

    def hook2(self, input, output):
        # if 0 in output:
        #     print('yes')
        output.mul_(delete_gate[1])
        # print(output)
        # hook2.t = hook2.t + 0.001
        # x = torch.ones_like(hook2.t)
        # hook2.t = torch.where(hook2.t > 1, x, hook2.t)

    def hook3(self, input, output):
        # if 0 in output:
        #     print('yes')
        output.mul_(delete_gate[2])
        # print(output)
        #     hook3.t = t_new
        # output.mul_(hook3.t)
        # hook3.t = hook3.t + 0.001
        # x = torch.ones_like(hook3.t)
        # hook3.t = torch.where(hook3.t > 1, x, hook3.t)

    if index == 0:
        return hook1
    if index == 1:
        return hook2
    if index == 2:
        return hook3


def remove_hook(hook_dic):
    for key in hook_dic.keys():
        hook_dic[key].remove()


def get_max_gate():
    global max_gate
    return max_gate


def record_gate(epoch, batch):
    global max_gate
    global dic_gate
    check_dic_key(dic_gate, epoch)
    check_dic_key(dic_gate[epoch], batch)
    for layer, gate in enumerate(max_gate):
        check_dic_key(dic_gate[epoch][batch], layer)
        flatten_tensor = torch.flatten(gate)
        list_from_tensor = flatten_tensor.tolist()
        dic_gate[epoch][batch][layer] = str(list_from_tensor)


def save_gate_dic_to_json(path):
    global dic_gate
    with open(path['path_to_gate'], 'w') as f:
        json.dump(dic_gate, f, indent=4)


def remove_gate_max():
    global max_gate
    max_gate = [None, None, None]

# import torch
# import numpy as np
#
# hook_tensor = [None, None, None]


# def hook_maker(shape, index, dim=4, fill=1):
#     global hook_tensor
#     t_old = torch.ones(shape[0])
#     t_new = torch.empty(shape[1]).fill_(fill)
#     t_new[:shape[0]] = t_old
#     if dim == 4:
#         t_new = torch.reshape(t_new, (1, -1, 1, 1))
#         t_new = t_new.requires_grad_(False)
#     else:
#         t_new = torch.reshape(t_new, (1, -1))
#         t_new = t_new.requires_grad_(False)
#     hook_tensor[index] = t_new
#
#     def hook1(self, input, output):
#         output.mul_(hook_tensor[0])
#         # hook1.t = hook1.t + 0.001
#         # x = torch.ones_like(hook1.t)
#         # hook1.t = torch.where(hook1.t > 1, x, hook1.t)
#
#     def hook2(self, input, output):
#         # if 0 in output:
#         #     print('yes')
#         output.mul_(hook_tensor[1])
#         # hook2.t = hook2.t + 0.001
#         # x = torch.ones_like(hook2.t)
#         # hook2.t = torch.where(hook2.t > 1, x, hook2.t)
#
#     def hook3(self, input, output):
#         # if 0 in output:
#         #     print('yes')
#         output.mul_(hook_tensor[2])
#         #     hook3.t = t_new
#         # output.mul_(hook3.t)
#         # hook3.t = hook3.t + 0.001
#         # x = torch.ones_like(hook3.t)
#         # hook3.t = torch.where(hook3.t > 1, x, hook3.t)
#
#     if index == 0:
#         return hook1
#     if index == 1:
#         return hook2
#     if index == 2:
#         return hook3


# def hook_maker(shape, index, dim=4, fill=0):
#     t_old = torch.ones(shape[0])
#     t_new = torch.empty(shape[1]).fill_(fill)
#     t_new[:shape[0]] = t_old
#     if dim == 4:
#         t_new = torch.reshape(t_new, (1, -1, 1, 1))
#         t_new = t_new.requires_grad_(False)
#     else:
#         t_new = torch.reshape(t_new, (1, -1))
#         t_new = t_new.requires_grad_(False)
#
#     def hook1(self, input, output):
#         if not hasattr(hook1, 't'):
#             hook1.t = t_new.clone()
#         output.mul_(hook1.t)
#         # hook1.t = hook1.t + 0.001
#         # x = torch.ones_like(hook1.t)
#         # hook1.t = torch.where(hook1.t > 1, x, hook1.t)
#
#     def hook2(self, input, output):
#         if not hasattr(hook2, 't'):
#             hook2.t = t_new
#         output.mul_(hook2.t)
#         # hook2.t = hook2.t + 0.001
#         # x = torch.ones_like(hook2.t)
#         # hook2.t = torch.where(hook2.t > 1, x, hook2.t)
#
#     def hook3(self, input, output):
#         if not hasattr(hook3, 't'):
#             hook3.t = t_new
#         output.mul_(hook3.t)
#         # hook3.t = hook3.t + 0.001
#         # x = torch.ones_like(hook3.t)
#         # hook3.t = torch.where(hook3.t > 1, x, hook3.t)
#
#     if index == 0:
#         return hook1
#     if index == 1:
#         return hook2
#     if index == 2:
#         return hook3

#
# def shape_hook_tensor(existing_index, s_score, is_rank=False):
#     global hook_tensor
#     dim = [4, 4, 2]
#     for layer, ex_index in enumerate(existing_index):
#         e_score = s_score[layer][:ex_index]
#         new_score = s_score[layer][ex_index:]
#         if is_rank:
#             score = np.max(e_score)
#         else:
#             score = np.mean(e_score)
#         score = 0.0005 if score == 0 else score
#         gate_score = np.asarray(new_score) / score
#         t_old = torch.ones(ex_index)
#         t_new = torch.from_numpy(gate_score.astype(np.float32))
#         t_new = torch.cat((t_old, t_new), dim=0)
#
#         if dim[layer] == 4:
#             t_new = torch.reshape(t_new, (1, -1, 1, 1))
#             t_new = t_new.requires_grad_(False)
#         else:
#             t_new = torch.reshape(t_new, (1, -1))
#             t_new = t_new.requires_grad_(False)
#         hook_tensor[layer] = t_new
#
#
# def register_hook(model, names, old_size, new_size):
#     dic_size = {}
#     for key, o, n in zip(names, old_size, new_size):
#         dic_size[key] = (o, n)
#     hook_dic = {}
#     i = 0
#     for name, module in model.named_modules():
#         if name in names:
#             if name.startswith('fea'):
#                 d = 4
#             else:
#                 d = 2
#             hook_dic[name] = module.register_forward_hook(hook_maker(dic_size[name], i, dim=d))
#             i = i + 1
#     return hook_dic
#
#
# def remove_hook(hook_dic):
#     for key in hook_dic.keys():
#         hook_dic[key].remove()
