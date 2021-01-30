import json
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# ours = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rankgroup_og2_new2_good/grow_rankgroup_og2_new2__json_log.json"
# b_good = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_bridging_og2_new2_good/grow_bridging_og2_new2__json_log.json"
# copy = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_randomMap_og2_new2_good/grow_randomMap_og2_new2__json_log.json"
# rank = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rankconnect_og2_new2_good/grow_rankconnect_og2_new2__json_log.json"
# b_b = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_bridging_og2_new2_20200327-210217/grow_bridging_og2_new2__json_log.json"
# # standard = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_og2_new2_20200327-215019/grow_rankconnect_og2_new2__json_log.json"
#
# good_bn = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rankconnect_ogn_newn_BN_20200402-234427/grow_rankconnect_ogn_newn_BN__json_in_out.json"
# bad_nn = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rankconnect_ogn_newn_20200402-225756/grow_rankconnect_ogn_newn__json_in_out.json"
# bad_00 = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rankconnect_ognon-scale_newnon-scale_20200402-222240/grow_rankconnect_ognon-scale_newnon-scale__json_in_out.json"
#
# ####################
# bad1 = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rankconnect_ogn_newn_bad/grow_rankconnect_ogn_newn__json_in_out.json"
# bad2 = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rankconnect_ogn_newn_4down/grow_rankconnect_ogn_newn__json_in_out.json"
# good2 = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rankconnect_ogn_newn_2/grow_rankconnect_ogn_newn__json_in_out.json"
# good1 = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rankconnect_ogn_newn_2_1/grow_rankconnect_ogn_newn__json_in_out.json"
# good3 = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rankconnect_ogn_newn_3/grow_rankconnect_ogn_newn__json_in_out.json"
# data = None
# nono_bad2 = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rankconnect_ognon-scale_newnon-scale_bad2/grow_rankconnect_ognon-scale_newnon-scale__json_in_out.json"
# nono_bad1 = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rankconnect_ognon-scale_newnon-scale_bad1/grow_rankconnect_ognon-scale_newnon-scale__json_in_out.json"
# standard = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_standard/grow_standard_ognon-scale_newnon-scale__json_in_out.json"
# standard2 = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_standard_2/grow_standard_ognon-scale_newnon-scale__json_in_out.json"
#
# bad_n1 = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rankconnect_ogn_newn_20200402-225756/grow_rankconnect_ogn_newn__json_in_out.json"
#

rank_baseline = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_baseline_6/grow_rank_baseline__json_in_out.json"
copy_n = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_copy_n_6/grow_copy_n__json_in_out.json"
copy_one = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_copy_one_6/grow_copy_one__json_in_out.json"
rank_ours = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_ours_11/grow_rank_ours__json_in_out.json"
bridging = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_bridging_6/grow_bridging__json_in_out.json"
rank_cumulative_cosine = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_cosine_1/grow_rank_cumulative_cosine__json_in_out.json"
random = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_random_1/grow_random__json_in_out.json"
ranklow_n = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_ranklow_n_20200408-232641/grow_ranklow_n__json_in_out.json"
activation_low = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_activation_low_20200409-165613/grow_activation_low__json_in_out.json"
rank_cumulative_cosine_low = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_activation_low_BN_20200409-170241/grow_activation_low_BN__json_in_out.json"
rank_cumulative_low = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_low_1/grow_rank_cumulative__json_in_out.json"

dic_lr = dict(
    bridging_lr="/Users/zber/ProgramDev/exp_pyTorch/results/grow_bridging_LR_20200409-211253/grow_bridging_LR__json_in_out.json",
    ranklow_one_lr="/Users/zber/ProgramDev/exp_pyTorch/results/grow_ranklow_one_LR_20200409-215528/grow_ranklow_one_LR__json_in_out.json",
    rank_cumulative_cosine_low_lr="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_cosine_LR_20200409-212445/grow_rank_cumulative_cosine_LR__json_in_out.json"
)

dic = dict(
    # rank_baseline="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_baseline_6/layer1_log.txt",
    # rank_baseline_gate="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_baseline_gate/layer1_log.txt",
    # rank_baseline_LR3="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_baseline_LR3_20200412-233035/layer2_log.txt",
    # rank_baseline_LR4="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_baseline_LR4_20200413-001813/layer2_log.txt",
    # copy_n="/Users/zber/ProgramDev/exp_pyTorch/results/grow_copy_n_6/layer1_log.txt",
    # copy_one="/Users/zber/ProgramDev/exp_pyTorch/results/grow_copy_one_6/layer1_log.txt",
    # rank_ours="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_ours_11/layer1_log.txt",
    # bridging="/Users/zber/ProgramDev/exp_pyTorch/results/grow_bridging_6/layer1_log.txt",
    # rank_cumulative_cosine="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_cosine_1/layer1_log.txt",
    # rank_cumulative_cosine_gate="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_cosine_gate/layer1_log.txt",
    # rank_cumulative_cosine_LR="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_cosine_LR_20200412-214542/layer1_log.txt",
    # random="/Users/zber/ProgramDev/exp_pyTorch/results/grow_random_1/layer1_log.txt",
    # random_gate="/Users/zber/ProgramDev/exp_pyTorch/results/grow_random_gate/layer1_log.txt",
    # random_LR="/Users/zber/ProgramDev/exp_pyTorch/results/grow_random_LR_20200412-214020/layer1_log.txt",
    # ranklow_n="/Users/zber/ProgramDev/exp_pyTorch/results/grow_ranklow_n_20200408-232641/layer1_log.txt",
    # activation_low="/Users/zber/ProgramDev/exp_pyTorch/results/grow_activation_low_20200409-165613/layer1_log.txt",
    # rank_cumulative_cosine_low="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_cosine_low_20200409-203904/layer1_log.txt",
    # rank_cumulative="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_1/layer1_log.txt",
    # rank_cumulative_low="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_low_1/layer1_log.txt",
    # standard="/Users/zber/ProgramDev/exp_pyTorch/results/grow_standard_3/grow_standard__json_in_out.json",
    # rank_cumulative_gate = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_gate/layer1_log.txt",
    # rank_cumulative_gate_max_test="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_gate_max_test/layer2_log.txt",
    # standard_s="/Users/zber/ProgramDev/exp_pyTorch/results/grow_standard_s/grow_standard__json_dic.json",
    # rank_cumulative_cL_fH="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_convL_fcH/layer1_log.txt",
    # rank_cumulative_cL_fH2="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_convL_fcH2/layer1_log.txt",
    # rank_cumulative_H_normal="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_rank_20200415-161606/layer2_log.txt",
    # rank_cumulative_low_normal="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_low_20200415-161006/layer1_log.txt",
    # rank_cumulative_cosine_bn="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_cosine_BN_1/grow_rank_cumulative_cosine_BN__json_in_out.json",

)


# base_path = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_ours__newn_{i}/grow_rank_ours__newn__json_in_out.json"
#
# rank_baseline_bn = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_baseline_BN_1/grow_rank_baseline_BN__json_in_out.json"
# copy_n_bn = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_copy_n_BN_1/grow_copy_n_BN__json_in_out.json"
# copy_one_bn = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_copy_one_BN_1/grow_copy_one_BN__json_in_out.json"
# rank_ours_bn = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_ours_BN_1/grow_rank_ours_BN__json_in_out.json"
# bridging_bn = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_bridging_BN_1/grow_bridging_BN__json_in_out.json"
# rank_cumulative_cosine_bn = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_cosine_BN_1/grow_rank_cumulative_cosine_BN__json_in_out.json"
# rank_random_bn = ""
#
# path_list = []
# for i in range(1, 5):
#     p = base_path.format(i=i)
#     path_list.append(p)


def str_to_float(str):
    num_list = []
    str = str[1:-1]
    str_arry = str.split(', ')
    for num in str_arry:
        num_list.append(float(num))
    return np.asarray(num_list)


# # with open(j3, 'r') as f:
# #     data1 = json.load(f)
#
# # grow 0
#
# num_growth = 5
#
#
# def grow_x_y_exist(dic):
#     y = []
#     for e in range(1, dic["epoch_to"]):
#         for n in range(1, (600 // dic["num_batch"]) + 1):
#             batches = []
#             for b in range(1, dic["num_batch"] + 1):
#                 s = data[str(e)][str(b * n - 1)][dic["layer"]]["S"]["out"]
#                 f = str_to_float(s)
#                 start = 0
#                 end = dic["grow_base_size"][dic["layer"]]
#                 f_need = f[start:end]
#                 f_mean = np.mean(f_need)
#                 batches.append(f_mean)
#             y.append(np.mean(batches))
#     length = len(y)
#     x = np.arange(0, length)
#     return x, y
#
#
# def grow_x_y(dic, gi=1):
#     y = []
#     for e in range(dic["epoch_grow"][gi], dic["epoch_to"]):
#         for n in range(1, (600 // dic["num_batch"]) + 1):
#             batches = []
#             for b in range(1, dic["num_batch"] + 1):
#                 s = data[str(e)][str(b * n - 1)][dic["layer"]]["S"]["out"]
#                 f = str_to_float(s)
#                 start = dic["grow_base_size"][dic["layer"]] + (gi - 1) * dic["grow_size"][dic["layer"]]
#                 end = start + dic["grow_size"][dic["layer"]]
#                 f_need = f[start:end]
#                 f_mean = np.mean(f_need)
#                 batches.append(f_mean)
#             y.append(np.mean(batches))
#     length = len(y)
#     start_x = (dic["epoch_grow"][gi] - 1) * (600 // dic["num_batch"])
#     x = np.arange(start_x, start_x + length)
#     return x, y
#
#
# def grow_x_y_exist_in(dic):
#     y = []
#     for e in range(1, dic["epoch_to"]):
#         for n in range(1, (600 // dic["num_batch"]) + 1):
#             batches = []
#             for b in range(1, dic["num_batch"] + 1):
#                 s = data[str(e)][str(b * n - 1)][dic["layer"]]["S"]["in"]
#                 f = str_to_float(s)
#                 start = 0
#                 end = dic["grow_base_size"][dic["layer"]]
#                 f_need = f[start:end]
#                 f_mean = np.mean(f_need)
#                 batches.append(f_mean)
#             y.append(np.mean(batches))
#     length = len(y)
#     x = np.arange(0, length)
#     return x, y
#
#
# def grow_x_y_in(dic, gi=1):
#     y = []
#     for e in range(dic["epoch_grow"][gi], dic["epoch_to"]):
#         for n in range(1, (600 // dic["num_batch"]) + 1):
#             batches = []
#             for b in range(1, dic["num_batch"] + 1):
#                 s = data[str(e)][str(b * n - 1)][dic["layer"]]["S"]["in"]
#                 f = str_to_float(s)
#                 start = dic["grow_base_size"][dic["layer"]] + (gi - 1) * dic["grow_size"][dic["layer"]]
#                 end = start + dic["grow_size"][dic["layer"]]
#                 f_need = f[start:end]
#                 f_mean = np.mean(f_need)
#                 batches.append(f_mean)
#             y.append(np.mean(batches))
#     length = len(y)
#     start_x = (dic["epoch_grow"][gi] - 1) * (600 // dic["num_batch"])
#     x = np.arange(start_x, start_x + length)
#     return x, y


def plot_save(y, dir, is_out=False, m='L1'):
    label_list_out = ['layer0_out', 'layer1_out', 'layer2_out']
    label_list_in = ['layer1_in', 'layer2_in', 'layer3_in']
    colour_list = ['r--', 'b', 'g']
    if is_out:
        label_list = label_list_out
    else:
        label_list = label_list_in
    path = os.path.join(dir, 'zeros_{}_{}.png'.format(('out' if is_out else 'in'), m))
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 8))
    for layer, colour, label, in zip(y.keys(), colour_list, label_list):
        x = range(len(y[layer]))
        ax.plot(x, y[layer], colour, label=label)
    ax.legend()
    fig.savefig(path)
    plt.close(fig)


# def gen_x_y(is_out=True):
#     x_y = []
#     if is_out:
#         for gi in range(num_growth):
#             if gi == 0:
#                 x_y.append(grow_x_y_exist(dic_layer_out))
#             else:
#                 x_y.append(grow_x_y(dic_layer_out, gi=gi))
#     else:
#         for gi in range(num_growth):
#             if gi == 0:
#                 x_y.append(grow_x_y_exist_in(dic_layer_in))
#             else:
#                 x_y.append(grow_x_y_in(dic_layer_in, gi=gi))
#     return x_y


def func_cnt(arr):
    zero_els = np.count_nonzero(arr == 0)
    return zero_els


def gen_per_zero_W_T(is_out=True, m='L1'):
    if is_out:
        dic = dic_layer_out
        y = {"0": [], "1": [], "2": [], }
        o_in = 'out'
    else:
        dic = dic_layer_in
        y = {"1": [], "2": [], "3": [], }
        o_in = 'in'
    for layer in y.keys():
        for e in range(dic['epoch_from'], dic["epoch_to"]):
            for b in range(0, dic['batches']):
                # if m == 'W':
                #     s = data[str(e)][str(b)][layer][m][o_in]['mean']
                # else:
                #     s = data[str(e)][str(b)][layer][m][o_in]
                s1 = data[str(e)][str(b)][layer]["W"][o_in]['mean']
                s1 = str_to_float(s1)
                s2 = data[str(e)][str(b)][layer]["L1"][o_in]
                s2 = str_to_float(s2)
                f = np.asarray(s1) * np.asarray(s2)
                # f = str_to_float(s)
                zero_els = func_cnt(np.asarray(f))
                per_zero = zero_els / len(f) * 100
                # if per_zero > 99.9:
                #     print(s)
                y[layer].append(per_zero)
    return y


def gen_per_zero(is_out=True, m='L1'):
    if is_out:
        dic = dic_layer_out
        y = {"0": [], "1": [], "2": [], }
        o_in = 'out'
    else:
        dic = dic_layer_in
        y = {"1": [], "2": [], "3": [], }
        o_in = 'in'
    for layer in y.keys():
        for e in range(dic['epoch_from'], dic["epoch_to"]):
            for b in range(0, dic['batches']):
                if m == 'W':
                    s = data[str(e)][str(b)][layer][m][o_in]['mean']
                else:
                    s = data[str(e)][str(b)][layer][m][o_in]
                # s1 = data[str(e)][str(b)][layer]["W"][o_in]['mean']
                # s1 = str_to_float(s1)
                # s2 = data[str(e)][str(b)][layer]["L1"][o_in]
                # s2 = str_to_float(s2)
                # f = np.asarray(s1) * np.asarray(s2)
                f = str_to_float(s)
                zero_els = func_cnt(np.asarray(f))
                per_zero = zero_els / len(f) * 100
                # if per_zero > 99.9:
                #     print(s)
                y[layer].append(per_zero)
    return y


#
# def gen_per_zero_L1(is_out=True):
#     if is_out:
#         dic = dic_layer_out
#         y = {"0": [], "1": [], "2": [], }
#         o_in = 'out'
#     else:
#         dic = dic_layer_in
#         y = {"1": [], "2": [], "3": [], }
#         o_in = 'in'
#     for layer in y.keys():
#         for e in range(dic['epoch_from'], dic["epoch_to"]):
#             for b in range(0, dic['batches']):
#                 s = data[str(e)][str(b)][layer]["L1"][o_in]
#                 f = str_to_float(s)
#                 zero_els = func_cnt(np.asarray(f))
#                 per_zero = zero_els / len(f) * 100
#                 # if per_zero > 99.9:
#                 #     print(s)
#                 y[layer].append(per_zero)
#     return y


if __name__ == '__main__':
    dic_layer_out = {
        "epoch_from": 1,
        "layer": "1",
        "epoch_to": 11,
        "epoch_grow": [1, 2, 4, 7, 10],  # [1, 1, 1, 1, 1],
        "num_batch": 50,
        "batches": 600,
        "grow_base_size": {"0": 2, "1": 5, "2": 10},
        "grow_size": {"0": 2, "1": 3, "2": 10},
        "exsit_index": {"0": (0, 2), "1": (0, 5), "2": (0, 10)},
    }

    dic_layer_in = {
        "epoch_from": 1,
        "layer": "1",
        "epoch_to": 11,
        "epoch_grow": [1, 2, 4, 7, 10],  # [1, 2, 4, 7, 10],
        "num_batch": 50,
        "batches": 600,
        "grow_base_size": {"1": 2, "2": 80, "3": 10},
        "grow_size": {"1": 2, "2": 48, "3": 10},
        "exsit_index": {"1": (0, 2), "2": (0, 80), "3": (0, 10)},
    }

    # for file in [rank_cumulative_low]:#[ranklow_n,rank_cumulative_cosine_low]:
    # for key in dic_lr.keys():
    #     file = dic_lr[key]
    # [rank_cumulative_cosine,
    # rank_baseline, copy_n, copy_one,bridging,rank_cumulative_cosine,rank_cumulative_cosine,random,
    #          rank_cumulative_cosine_bn]:  # [rank_baseline_bn, copy_n_bn, copy_one_bn, rank_ours_bn, bridging_bn]:#[rank_baseline, copy_n, copy_one, rank_ours]:  # [good1,good2,bad1,bad2,nono_bad1, nono_bad2]: #, good1, good2
    # layer 1 ,2 ,3 ,4  S  the number of zeros

    m = 'S'

    for key in dic.keys():
        print(key, ':')

        # if key.startswith('standard'):
        #     dic_layer_in["epoch_grow"] = [1, 1, 1, 1, 1]
        #     dic_layer_out["epoch_grow"] = [1, 1, 1, 1, 1]
        #     if key == 'standard_s':
        #         dic_layer_out["grow_size"] = {"0": 4, "1": 10, "2": 100}
        #         dic_layer_out["grow_base_size"] = {"0": 4, "1": 10, "2": 100}
        #         dic_layer_in["grow_size"] = {"1": 4, "2": 160, "3": 100}
        #         dic_layer_in["grow_base_size"] = {"1": 4, "2": 160, "3": 100}
        # else:
        #     dic_layer_in["epoch_grow"] = [1, 2, 4, 7, 10]
        #     dic_layer_out["epoch_grow"] = [1, 2, 4, 7, 10]
        #
        path_to_file = dic[key]

        path = os.path.dirname(path_to_file)

        files = [f for f in glob.glob(path + "/*__json_in_out.json")]

        file = files[0]

        with open(file, 'r') as f:
            data = json.load(f)

        for out in [True, False]:
            is_out = out
            y = gen_per_zero(is_out, m=m)
            path = os.path.dirname(file)
            plot_save(y, path, out, m=m)
            #
            # if is_out:
            #     for layer in ["0", "1", "2"]:  #
            #         dic_layer_out["layer"] = layer
            #         x_y = gen_x_y(is_out=is_out)
            #         plot_save(x_y, path, layer=layer, is_out=is_out)
            # else:
            #     for layer in ["1", "2", "3"]:  #
            #         dic_layer_in["layer"] = layer
            #         x_y = gen_x_y(is_out=is_out)
            #         plot_save(x_y, path, layer=layer, is_out=is_out)

    # if is_out:
    #     # layer out
    #     x_0, y_0 = grow_x_y_exist(dic_layer_out)
    #     x_1, y_1 = grow_x_y(dic_layer_out, gi=1)
    #     x_2, y_2 = grow_x_y(dic_layer_out, gi=2)
    #     x_3, y_3 = grow_x_y(dic_layer_out, gi=3)
    #     x_4, y_4 = grow_x_y(dic_layer_out, gi=4)
    #     print('success')
    # else:
    #     # layer in
    #     x_0, y_0 = grow_x_y_exist_in(dic_layer_in)
    #     x_1, y_1 = grow_x_y_in(dic_layer_in, gi=1)
    #     x_2, y_2 = grow_x_y_in(dic_layer_in, gi=2)
    #     x_3, y_3 = grow_x_y_in(dic_layer_in, gi=3)
    #     x_4, y_4 = grow_x_y_in(dic_layer_in, gi=4)
    #     print('success')
    #
    # ax = plt.subplot(111)
    # # ax.plot(x, mean, label='C8_mean')
    # # ax.plot(x, mean1, label='C12_mean')
    # # ax.plot(x, mean2, label='C16_mean')
    # # ax.plot(x, mean3, label='C20_mean')
    # # ax.plot(x, mean4, label='C24_mean')
    # # ax.legend()
    # ax.plot(x_0, y_0, 'r--', label='Exsiting')
    # ax.plot(x_1, y_1, 'b', label='Grow_1')
    # ax.plot(x_2, y_2, 'g', label='Grow_2')
    # ax.plot(x_3, y_3, 'yellow', label='Grow_3')
    # ax.plot(x_4, y_4, 'black', label='Grow_4')
    # ax.legend()
    # plt.show()

    # def grow_x_y(dic, eg=0):
    #     for e in range(eg, 11):
    #         for n in range(600 // dic["num_batch"]):
    #             batches = []
    #             for b in range(dic["batch_from"], dic["num_batch"]):
    #                 if e > eg:
    #
    #                 s_ = data[str(e)][str(b * n)][dic["layer"]]["S"]["exist_out"]
    #
    #
    # epoch = "10"
    # layers = ["0", "1", "2"]
    # before_after = [("B0", "A0"), ("B3", "A3"), ("B5", "A5")]
    #
    # for i, b_a in enumerate(before_after):
    #     for layer in layers[i:]:
    #         for mode in ["W"]:
    #             for ele in ["M_mean", "M_std"]:
    #                 str_title = 'L{},{},{} = '.format(layer, mode, ele)
    #                 b = data1[epoch][b_a[0]][layer][mode][ele]
    #                 a = data1[epoch][b_a[1]][layer][mode][ele]
    #                 c = float(a) - float(b)
    #                 print(str_title, c)
    #         for mode in ["S", "L1", "L2"]:
    #             for ele in ["layer"]:
    #                 str_title = 'L{},{},{} = '.format(layer, mode, ele)
    #                 b = data1[epoch][b_a[0]][layer][mode][ele]
    #                 a = data1[epoch][b_a[1]][layer][mode][ele]
    #                 c = float(a) - float(b)
    #                 print(str_title, c)
    #         print()
    #     print()
    #     print()
    #
    # for i, b_a in enumerate(before_after):
    #     for layer in layers[i:]:
    #         for mode in ["W"]:
    #             for ele in ["mean", "std"]:
    #                 str_title = 'L{},{},{} = '.format(layer, mode, ele)
    #                 b = data1[epoch][b_a[0]][layer][mode][ele]
    #                 a = data1[epoch][b_a[1]][layer][mode][ele]
    #                 # c = str_to_float(a) - str_to_float(b)
    #                 print(str_title, '-Before', b)
    #                 print(str_title, '-After', a)
    #         for mode in ["S", "L1", "L2"]:
    #             for ele in ["channel"]:
    #                 str_title = 'L{},{},{} = '.format(layer, mode, ele)
    #                 b = data1[epoch][b_a[0]][layer][mode][ele]
    #                 a = data1[epoch][b_a[1]][layer][mode][ele]
    #                 # c = str_to_float(a) - str_to_float(b)
    #                 print(str_title, '-Before', b)
    #                 print(str_title, '-After', a)
    #         print()
    #     print()
    #     print()
    #
    # for i, b_a in enumerate(before_after):
    #     for layer in layers[(i + 1):]:
    #         for mode in ["W"]:
    #             for ele in ["mean", "std"]:
    #                 str_title = 'L{},{},{} = '.format(layer, mode, ele)
    #                 b = data1[epoch][b_a[0]][layer][mode][ele]
    #                 a = data1[epoch][b_a[1]][layer][mode][ele]
    #                 c = str_to_float(a) - str_to_float(b)
    #                 print(str_title, c.tolist())
    #         for mode in ["S", "L1", "L2"]:
    #             for ele in ["channel"]:
    #                 str_title = 'L{},{},{} = '.format(layer, mode, ele)
    #                 b = data1[epoch][b_a[0]][layer][mode][ele]
    #                 a = data1[epoch][b_a[1]][layer][mode][ele]
    #                 c = str_to_float(a) - str_to_float(b)
    #                 print(str_title, c.tolist())
    #         print()
    #     print()
    #     print()
    # def get_smooth(x):
    #     if not hasattr(get_smooth, "t"):
    #         get_smooth.t = [x, x, x]
    #
    #     get_smooth.t[2] = get_smooth.t[1]
    #     get_smooth.t[1] = get_smooth.t[0]
    #     get_smooth.t[0] = x
    #
    #     return (get_smooth.t[0] + get_smooth.t[1] + get_smooth.t[2]) / 3
    #
    # print(get_smooth(12))
    # print(get_smooth.t)
    # print(get_smooth(80))
