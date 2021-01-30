import torch
import numpy as np
from utils import check_dic_key
import json

pre_dic = {}
dic_lr = {}
exsit_dic = {}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def gen_diff_lr_channel_dic(old_shape_list, model, score_out, score_in, step=0, total_step=600):
    global pre_dic
    dic = {}

    cur_dic = init_channel_lr(old_shape_list, model, score_out, score_in)

    for key in cur_dic.keys():
        old_lr = pre_dic[key]
        new_lr = cur_dic[key]
        ratio = 1 / new_lr - 1 / old_lr
        ratio_like = torch.empty_like(ratio).fill_(0.0005)
        ratio = torch.where(ratio < 0, ratio_like, ratio)
        lr = old_lr - old_lr * ratio
        lr_ones = torch.ones_like(lr)
        lr = torch.where(lr < 1, lr_ones, lr)

        # ratio_one = torch.ones_like(ratio_lr) - 0.002
        # lambda_lr = torch.where(ratio_lr > 1, ratio_one, ratio_lr)
        # lr = old_lr * lambda_lr

        left_lr = old_lr - 1
        left_step = total_step - step

        left_decay = left_lr / left_step
        lr_decay = old_lr - left_decay

        min_lr = torch.min(lr, lr_decay)

        one_lr = torch.ones_like(min_lr)

        f_lr = torch.where(min_lr < 1, one_lr, min_lr)

        dic[key] = f_lr.to(device)
    pre_dic = dic
    return dic


def gen_diff_lr_channel_dic_out(old_shape_list, model, score_out, score_in, step=0, total_step=600):
    global pre_dic
    dic = {}

    # cur_dic = init_channel_lr(old_shape_list, model, score_out, score_in)
    cur_dic = init_channel_lr_out(old_shape_list, model, score_out, score_in)

    for key in cur_dic.keys():
        old_lr = pre_dic[key]
        new_lr = cur_dic[key]
        ratio = 1 / new_lr - 1 / old_lr
        ratio_like = torch.empty_like(ratio).fill_(0.0005)
        ratio = torch.where(ratio < 0, ratio_like, ratio)
        lr = old_lr - old_lr * ratio
        lr_ones = torch.ones_like(lr)
        lr = torch.where(lr < 1, lr_ones, lr)

        # ratio_one = torch.ones_like(ratio_lr) - 0.002
        # lambda_lr = torch.where(ratio_lr > 1, ratio_one, ratio_lr)
        # lr = old_lr * lambda_lr

        left_lr = old_lr - 1
        left_step = total_step - step

        left_decay = left_lr / left_step
        lr_decay = old_lr - left_decay

        min_lr = torch.min(lr, lr_decay)

        one_lr = torch.ones_like(min_lr)

        f_lr = torch.where(min_lr < 1, one_lr, min_lr)

        dic[key] = f_lr.to(device)
    pre_dic = dic
    return dic


def reshape_tensor(dim, t_list, mode='out'):
    d = {2: {'out': (-1, 1), 'in': (1, -1)}, 4: {'out': (-1, 1, 1, 1), 'in': (1, -1, 1, 1)}}
    if dim > 1:
        return t_list.reshape(d[dim][mode])
    else:
        return t_list


def scale_lr(arr, min=2, max=4):
    interval = (max - min) / (len(arr) - 1)
    arr = arr * -1
    sort = arr.argsort()
    for i in range(len(arr)):
        arr[sort[i]] = min + i * interval
    return arr


def scale_lr_closet(arr, new_len, exist_len, min_value=2, max_value=4):
    grad_lr = []
    # ratio = 1 - (new_len / (new_len + exist_len))
    # max_value = 2 / ratio
    # min_value = 1
    interval = (max_value - min_value) / (len(arr) - 1)
    for i in range(len(arr)):
        grad_lr.append(min_value + i * interval)

    closet_lr = find_closet_value(arr, grad_lr)
    return np.asarray(closet_lr)
    #
    # arr = arr * -1
    # sort = arr.argsort()
    # for i in range(len(arr)):
    #     arr[sort[i]] = min + i * interval
    # return arr


def find_closet_value(calculated_lr, grad_lr):
    closet_list = []
    for c_lr in calculated_lr:
        closet_lr = grad_lr[0]
        closet_len = c_lr - grad_lr[0]
        for g_lr in grad_lr:
            if abs(g_lr - c_lr) < closet_len:
                closet_lr = g_lr
                closet_len = abs(g_lr - c_lr)
        closet_list.append(closet_lr)
    return closet_list


# if is_bn need to change the parameters
def init_channel_lr_old(old_shape_list, new_model, score_out, score_in, is_rank=False, is_first=False):
    dic = {}
    global pre_dic
    for i, z in enumerate(zip(old_shape_list, list(new_model.parameters()))):
        out_lr = None
        in_lr = None
        old_shape = z[0]
        new_p = z[1]
        new_t = torch.ones(new_p.shape)

        # out
        if new_t.shape[0] - old_shape[0] > 0:
            new_out = list(new_p.shape)
            new_out[0] = new_out_size = new_p.shape[0] - old_shape[0]
            new_out = tuple(new_out)
            out_t = torch.ones(new_out)
            layer_index = i // 2
            e_out_score = score_out[layer_index][:-new_out_size]
            new_out_score = score_out[layer_index][-new_out_size:]
            if is_rank:
                score = np.max(e_out_score)
            else:
                score = np.mean(e_out_score)
            # # add small numerator to avoid divide by 0
            new_out_score = np.asarray(new_out_score)

            new_out_score = np.where(new_out_score == 0, new_out_score + 0.0005, new_out_score)
            out_lr = score / new_out_score
            if is_first:
                out_lr = scale_lr_closet(out_lr, len(new_out_score), len(e_out_score))

            # if 0 in new_out_score:
            #     print('yes')
            # out_lr = scale_lr(new_out_score)

            out_lr = torch.from_numpy(out_lr.astype(np.float32))
            out_lr = out_t * reshape_tensor(out_t.ndim, out_lr)

        # in
        if len(old_shape) > 1 and new_t.shape[1] - old_shape[1] > 0:
            new_in = list(new_p.shape)
            new_in[1] = new_in_size = new_p.shape[1] - old_shape[1]
            new_in = tuple(new_in)
            in_t = torch.ones(new_in)
            layer_index = i // 2
            e_in_score = score_in[layer_index][:-new_in_size]
            new_in_score = score_in[layer_index][-new_in_size:]
            if is_rank:
                score = np.max(e_in_score)
            else:
                score = np.mean(e_in_score)

            new_in_score = np.asarray(new_in_score)
            # if 0 in new_in_score:
            #     print('yes')
            # in_lr = scale_lr(new_in_score)

            # add small numerator to avoid divide by 0
            new_in_score = np.where(new_in_score == 0, new_in_score + 0.0005, new_in_score)
            in_lr = score / new_in_score
            if is_first:
                in_lr = scale_lr_closet(in_lr, len(new_in_score), len(e_in_score))

            in_lr = torch.from_numpy(in_lr.astype(np.float32))
            in_lr = in_t * reshape_tensor(in_t.ndim, in_lr, mode='in')

        # split into 3 parts
        if out_lr is not None and in_lr is not None:
            in_lr_shape = in_lr.shape
            out_lr_shape = out_lr.shape
            out_part = out_lr[:, :-in_lr_shape[1]]
            avg_out_part = out_lr[:, -in_lr_shape[1]:]
            in_part = in_lr[:-out_lr_shape[0], :]
            avg_in_part = in_lr[-out_lr_shape[0]:, :]
            avg_part = (avg_out_part + avg_in_part) / 2
            new_t[-out_lr_shape[0]:, :-in_lr_shape[1]] = out_part
            new_t[:-out_lr_shape[0], -in_lr_shape[1]:] = in_part
            new_t[-out_lr_shape[0]:, -in_lr_shape[1]:] = avg_part
        elif out_lr is None and in_lr is not None:
            new_t[:, -new_in_size:] = in_lr
        elif in_lr is None and out_lr is not None:
            new_t[-new_out_size:] = out_lr
        else:
            pass

        dic[new_p] = new_t.to(device)
    if is_first:
        pre_dic = dic
    return dic


# if is_bn need to change the parameters
def init_channel_lr_out(old_shape_list, new_model, score_out, score_in, is_rank=False, is_first=False):
    dic = {}
    global pre_dic
    for i, z in enumerate(zip(old_shape_list, list(new_model.parameters()))):
        out_lr = None
        in_lr = None
        old_shape = z[0]
        new_p = z[1]
        new_t = torch.ones(new_p.shape)

        # out
        if new_t.shape[0] - old_shape[0] > 0:
            new_out = list(new_p.shape)
            new_out[0] = new_out_size = new_p.shape[0] - old_shape[0]
            new_out = tuple(new_out)
            out_t = torch.ones(new_out)
            layer_index = i // 2
            e_out_score = score_out[layer_index][:-new_out_size]
            new_out_score = score_out[layer_index][-new_out_size:]
            if is_rank:
                score = np.max(e_out_score)
            else:
                score = np.mean(e_out_score)
            # # add small numerator to avoid divide by 0
            new_out_score = np.asarray(new_out_score)

            new_out_score = np.where(new_out_score == 0, new_out_score + 0.0005, new_out_score)
            out_lr = score / new_out_score
            if is_first:
                out_lr = scale_lr_closet(out_lr, len(new_out_score), len(e_out_score))

            # if 0 in new_out_score:
            #     print('yes')
            # out_lr = scale_lr(new_out_score)

            out_lr = torch.from_numpy(out_lr.astype(np.float32))
            out_lr = out_t * reshape_tensor(out_t.ndim, out_lr)

        # # in
        # if len(old_shape) > 1 and new_t.shape[1] - old_shape[1] > 0:
        #     new_in = list(new_p.shape)
        #     new_in[1] = new_in_size = new_p.shape[1] - old_shape[1]
        #     new_in = tuple(new_in)
        #     in_t = torch.ones(new_in)
        #     layer_index = i // 2
        #     e_in_score = score_in[layer_index][:-new_in_size]
        #     new_in_score = score_in[layer_index][-new_in_size:]
        #     if is_rank:
        #         score = np.max(e_in_score)
        #     else:
        #         score = np.mean(e_in_score)
        #
        #     new_in_score = np.asarray(new_in_score)
        #     # if 0 in new_in_score:
        #     #     print('yes')
        #     # in_lr = scale_lr(new_in_score)
        #
        #     # add small numerator to avoid divide by 0
        #     new_in_score = np.where(new_in_score == 0, new_in_score + 0.0005, new_in_score)
        #     in_lr = score / new_in_score
        #     if is_first:
        #         in_lr = scale_lr_closet(in_lr, len(new_in_score), len(e_in_score))
        #
        #     in_lr = torch.from_numpy(in_lr.astype(np.float32))
        #     in_lr = in_t * reshape_tensor(in_t.ndim, in_lr, mode='in')

        # split into 3 parts
        if out_lr is not None and in_lr is not None:
            in_lr_shape = in_lr.shape
            out_lr_shape = out_lr.shape
            out_part = out_lr[:, :-in_lr_shape[1]]
            avg_out_part = out_lr[:, -in_lr_shape[1]:]
            in_part = in_lr[:-out_lr_shape[0], :]
            avg_in_part = in_lr[-out_lr_shape[0]:, :]
            avg_part = (avg_out_part + avg_in_part) / 2
            new_t[-out_lr_shape[0]:, :-in_lr_shape[1]] = out_part
            new_t[:-out_lr_shape[0], -in_lr_shape[1]:] = in_part
            new_t[-out_lr_shape[0]:, -in_lr_shape[1]:] = avg_part
        elif out_lr is None and in_lr is not None:
            new_t[:, -new_in_size:] = in_lr
        elif in_lr is None and out_lr is not None:
            new_t[-new_out_size:] = out_lr
        else:
            pass
        dic[new_p] = new_t.to(device)
    if is_first:
        pre_dic = dic
    return dic


# auto decay exist, only out channel
# if is_bn need to change the parameters
def init_channel_lr_exist(old_shape_list, new_model, fill=2.0):
    dic = {}
    global pre_dic
    for i, z in enumerate(zip(old_shape_list, list(new_model.parameters()))):
        old_shape = z[0]
        new_p = z[1]
        new_t = torch.ones(new_p.shape)

        # out
        if new_t.shape[0] - old_shape[0] > 0:
            old_t = torch.empty(old_shape[0]).fill_(fill)
            new = torch.ones(new_t.shape[0])
            new[:old_shape[0]] = old_t
            new = reshape_tensor(new_t.dim(), new)
            new_t = new_t * new

        dic[new_p] = new_t.to(device)
        pre_dic = dic
    return dic


def init_channel_lr(old_shape_list, new_model, fill=2.0):
    dic = {}
    global pre_dic
    for i, z in enumerate(zip(old_shape_list, list(new_model.parameters()))):
        new_p = z[1]
        new_t = torch.empty(new_p.shape).fill_(fill)
        dic[new_p] = new_t.to(device)
        pre_dic = dic
    return dic


def auto_decay_lr_exist(old_shape_list, step=0, total_step=600):
    global pre_dic
    dic = {}

    for key, old_shape in zip(pre_dic.keys(), old_shape_list):
        old_lr = pre_dic[key]
        decay_step = (old_lr - 1) / (total_step - step)
        new_lr = old_lr - decay_step
        dic[key] = new_lr.to(device)

    pre_dic = dic
    return dic


def gen_diff_lr_channel_dic_out_exist(old_shape_list, model, score_out, score_in, step=0, total_step=600):
    global pre_dic
    dic = {}

    # cur_dic = init_channel_lr(old_shape_list, model, score_out, score_in)
    cur_dic = init_channel_lr_out_exist(old_shape_list, model, score_out, score_in)

    for key, old_shape in zip(cur_dic.keys(), old_shape_list):
        old_lr = pre_dic[key]
        new_lr = cur_dic[key]
        ratio = 1 / new_lr - 1 / old_lr
        ratio_like = torch.empty_like(ratio).fill_(0.0005)
        ratio = torch.where(ratio < 0, ratio_like, ratio)
        lr = old_lr - old_lr * ratio
        lr_ones = torch.ones_like(lr)
        lr = torch.where(lr < 1, lr_ones, lr)

        # ratio_one = torch.ones_like(ratio_lr) - 0.002
        # lambda_lr = torch.where(ratio_lr > 1, ratio_one, ratio_lr)
        # lr = old_lr * lambda_lr

        left_lr = old_lr - 1
        left_step = total_step - step

        left_decay = left_lr / left_step
        lr_decay = old_lr - left_decay

        # split out old
        old_lr_decay = lr_decay[:old_shape[0]]
        # old_lr_decay = torch.empty_like(old_lr_decay).fill_(0.5 + 0.5 / total_step * step)

        # split out new
        new_lr_decay = lr_decay[old_shape[0]:]

        #
        new_lr = lr[old_shape[0]:]
        min_lr = torch.min(new_lr, new_lr_decay)
        one_lr = torch.ones_like(min_lr)

        new_f_lr = torch.where(min_lr < 1, one_lr, min_lr)

        f_lr = torch.cat((old_lr_decay, new_f_lr))

        dic[key] = f_lr.to(device)
        # pre_dic[key] = f_lr.to(device)

    pre_dic = dic
    return dic


# out_exist
def init_channel_lr_out_exist(old_shape_list, new_model, score_out, score_in, is_rank=False, is_first=False,
                              exist_rate=0.5):
    dic = {}
    global pre_dic
    for i, z in enumerate(zip(old_shape_list, list(new_model.parameters()))):
        out_lr = None
        in_lr = None
        old_shape = z[0]
        new_p = z[1]
        if new_p.shape[0] - old_shape[0] > 0:
            new_t = torch.empty_like(new_p).fill_(exist_rate)
        else:
            new_t = torch.empty_like(new_p).fill_(1)
        # new_t = torch.ones(new_p.shape)

        # out
        if new_t.shape[0] - old_shape[0] > 0:
            new_out = list(new_p.shape)
            new_out[0] = new_out_size = new_p.shape[0] - old_shape[0]
            new_out = tuple(new_out)
            out_t = torch.ones(new_out)
            layer_index = i // 2
            e_out_score = score_out[layer_index][:-new_out_size]
            new_out_score = score_out[layer_index][-new_out_size:]
            if is_rank:
                score = np.max(e_out_score)
            else:
                score = np.mean(e_out_score)
            # # add small numerator to avoid divide by 0
            new_out_score = np.asarray(new_out_score)

            new_out_score = np.where(new_out_score == 0, new_out_score + 0.0005, new_out_score)
            out_lr = score / new_out_score
            if is_first:
                out_lr = scale_lr_closet(out_lr, len(new_out_score), len(e_out_score))

            # if 0 in new_out_score:
            #     print('yes')
            # out_lr = scale_lr(new_out_score)

            out_lr = torch.from_numpy(out_lr.astype(np.float32))
            out_lr = out_t * reshape_tensor(out_t.ndim, out_lr)

        # # in
        # if len(old_shape) > 1 and new_t.shape[1] - old_shape[1] > 0:
        #     new_in = list(new_p.shape)
        #     new_in[1] = new_in_size = new_p.shape[1] - old_shape[1]
        #     new_in = tuple(new_in)
        #     in_t = torch.ones(new_in)
        #     layer_index = i // 2
        #     e_in_score = score_in[layer_index][:-new_in_size]
        #     new_in_score = score_in[layer_index][-new_in_size:]
        #     if is_rank:
        #         score = np.max(e_in_score)
        #     else:
        #         score = np.mean(e_in_score)
        #
        #     new_in_score = np.asarray(new_in_score)
        #     # if 0 in new_in_score:
        #     #     print('yes')
        #     # in_lr = scale_lr(new_in_score)
        #
        #     # add small numerator to avoid divide by 0
        #     new_in_score = np.where(new_in_score == 0, new_in_score + 0.0005, new_in_score)
        #     in_lr = score / new_in_score
        #     if is_first:
        #         in_lr = scale_lr_closet(in_lr, len(new_in_score), len(e_in_score))
        #
        #     in_lr = torch.from_numpy(in_lr.astype(np.float32))
        #     in_lr = in_t * reshape_tensor(in_t.ndim, in_lr, mode='in')

        # split into 3 parts
        if out_lr is not None and in_lr is not None:
            in_lr_shape = in_lr.shape
            out_lr_shape = out_lr.shape
            out_part = out_lr[:, :-in_lr_shape[1]]
            avg_out_part = out_lr[:, -in_lr_shape[1]:]
            in_part = in_lr[:-out_lr_shape[0], :]
            avg_in_part = in_lr[-out_lr_shape[0]:, :]
            avg_part = (avg_out_part + avg_in_part) / 2
            new_t[-out_lr_shape[0]:, :-in_lr_shape[1]] = out_part
            new_t[:-out_lr_shape[0], -in_lr_shape[1]:] = in_part
            new_t[-out_lr_shape[0]:, -in_lr_shape[1]:] = avg_part
        elif out_lr is None and in_lr is not None:
            new_t[:, -new_in_size:] = in_lr
        elif in_lr is None and out_lr is not None:
            new_t[-new_out_size:] = out_lr
        else:
            pass

        dic[new_p] = new_t.to(device)
    if is_first:
        pre_dic = dic
    return dic


def record_lr(epoch, batch):
    global pre_dic
    global dic_lr
    check_dic_key(dic_lr, epoch)
    check_dic_key(dic_lr[epoch], batch)
    for layer, key in enumerate(pre_dic.keys()):
        if layer in [1, 3, 5, 7]:
            t_lr = pre_dic[key]
            if t_lr.ndim == 4:
                lr = t_lr[:, 0, 0, 0]
            elif t_lr.ndim == 2:
                lr = t_lr[:, 0]
            else:
                lr = t_lr
            lr = torch.flatten(lr)
            list_from_tensor = lr.tolist()
            dic_lr[epoch][batch][layer] = str(list_from_tensor)


def save_lr_dic_to_json(path):
    global dic_lr
    with open(path['path_to_lr'], 'w') as f:
        json.dump(dic_lr, f, indent=4)

#
# # if is_bn need to change the parameters
# def init_channel_lr_out(old_shape_list, new_model, score_out, score_in, is_rank=False):
#     dic = {}
#     global pre_dic
#     for i, z in enumerate(zip(old_shape_list, list(new_model.parameters()))):
#         out_lr = None
#         in_lr = None
#         old_shape = z[0]
#         new_p = z[1]
#         new_t = torch.ones(new_p.shape)
#         # out
#         if i < 4:
#             new_out = list(new_p.shape)
#             new_out[0] = new_out_size = new_p.shape[0] - old_shape[0]
#             new_out = tuple(new_out)
#             out_t = torch.ones(new_out)
#             layer_index = i // 2
#             e_out_score = score_out[layer_index][:-new_out_size]
#             new_out_score = score_out[layer_index][-new_out_size:]
#             # if is_rank:
#             #     score = np.max(e_out_score)
#             # else:
#             #     score = np.mean(e_out_score)
#             # add small numerator to avoid divide by 0
#             new_out_score = np.asarray(new_out_score)
#             # if 0 in new_out_score:
#             #     print('yes')
#             # new_out_score = np.where(new_out_score == 0, np.min(new_out_score), new_out_score)
#             # out_lr = score / new_out_score
#             out_lr = scale_lr(new_out_score)
#             out_lr = torch.from_numpy(out_lr.astype(np.float32))
#             out_lr = out_t * reshape_tensor(out_t.ndim, out_lr)
#
#         # in
#         if new_p.ndim > 1 and 1 < i < 4:
#             new_in = list(new_p.shape)
#             new_in[1] = new_in_size = new_p.shape[1] - old_shape[1]
#             new_in = tuple(new_in)
#             in_t = torch.ones(new_in)
#             layer_index = i // 2
#             e_in_score = score_in[layer_index][:-new_in_size]
#             new_in_score = score_in[layer_index][-new_in_size:]
#             # if is_rank:
#             #     score = np.max(e_in_score)
#             # else:
#             #     score = np.mean(e_in_score)
#             # add small numerator to avoid divide by 0
#             new_in_score = np.asarray(new_in_score)
#             # if 0 in new_in_score:
#             #     print('yes')
#             # new_in_score = np.where(new_in_score == 0.0, np.min(new_in_score), new_in_score)
#             # in_lr = score / new_in_score
#             in_lr = scale_lr(new_in_score)
#             in_lr = torch.from_numpy(in_lr.astype(np.float32))
#             in_lr = in_t * reshape_tensor(in_t.ndim, in_lr, mode='in')
#
#         # split into 3 parts
#         if out_lr is not None and in_lr is not None:
#             in_lr_shape = in_lr.shape
#             out_lr_shape = out_lr.shape
#             out_part = out_lr[:, :-in_lr_shape[1]]
#             avg_out_part = out_lr[:, -in_lr_shape[1]:]
#             in_part = in_lr[:-out_lr_shape[0], :]
#             avg_in_part = in_lr[-out_lr_shape[0]:, :]
#             avg_part = (avg_out_part + avg_in_part) / 2
#             new_t[-out_lr_shape[0]:, :-in_lr_shape[1]] = out_part
#             new_t[:-out_lr_shape[0], -in_lr_shape[1]:] = in_part
#             new_t[-out_lr_shape[0]:, -in_lr_shape[1]:] = avg_part
#         elif out_lr is None and in_lr is not None:
#             new_t[:, -new_in_size:] = in_lr
#         elif in_lr is None and out_lr is not None:
#             new_t[-new_out_size:] = out_lr
#         else:
#             pass
#             # print('wow')
#
#         dic[new_p] = new_t.to(device)
#
#     pre_dic = dic
#     return dic
