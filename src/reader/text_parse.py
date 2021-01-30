import glob
import os
import numpy as np
import json
import copy

bridging_bad = '/Users/zber/ProgramDev/exp_pyTorch/results/grow_bridging_og2_new2_bad/layer0_log.txt'
bridging_good = '/Users/zber/ProgramDev/exp_pyTorch/results/grow_bridging_og2_new2_good/layer0_log.txt'
randomMap_bad = '/Users/zber/ProgramDev/exp_pyTorch/results/grow_randomMap_og2_new2_bad/layer0_log.txt'
randomMap_good = '/Users/zber/ProgramDev/exp_pyTorch/results/grow_randomMap_og2_new2_good/layer0_log.txt'
rank_bad = '/Users/zber/ProgramDev/exp_pyTorch/results/grow_rankconnect_og2_new2_bad/layer0_log.txt'
rank_good = '/Users/zber/ProgramDev/exp_pyTorch/results/grow_rankconnect_og2_new2_good/layer0_log.txt'
ours_bad = '/Users/zber/ProgramDev/exp_pyTorch/results/grow_rankgroup_og2_new2_bad/layer0_log.txt'
ours_good = '/Users/zber/ProgramDev/exp_pyTorch/results/grow_rankgroup_og2_new2_good/layer0_log.txt'
bb = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_bridging_og2_new2_20200327-210217/layer0_log.txt"

bridging_bad = '/Users/zber/ProgramDev/exp_pyTorch/results/grow_bridging_og2_new2_bad/grow_bridging_og2_new2__log.txt'
bridging_good = '/Users/zber/ProgramDev/exp_pyTorch/results/grow_bridging_og2_new2_good/grow_bridging_og2_new2__log.txt'
randomMap_bad = '/Users/zber/ProgramDev/exp_pyTorch/results/grow_randomMap_og2_new2_bad/grow_randomMap_og2_new2__log.txt'
randomMap_good = '/Users/zber/ProgramDev/exp_pyTorch/results/grow_randomMap_og2_new2_good/grow_randomMap_og2_new2__log.txt'
rank_bad = '/Users/zber/ProgramDev/exp_pyTorch/results/grow_rankconnect_og2_new2_bad/grow_rankconnect_og2_new2__log.txt'
rank_good = '/Users/zber/ProgramDev/exp_pyTorch/results/grow_rankconnect_og2_new2_good/grow_rankconnect_og2_new2__log.txt'
ours_bad = '/Users/zber/ProgramDev/exp_pyTorch/results/grow_rankgroup_og2_new2_bad/grow_rankgroup_og2_new2__log.txt'
ours_good = '/Users/zber/ProgramDev/exp_pyTorch/results/grow_rankgroup_og2_new2_good/grow_rankgroup_og2_new2__log.txt'
bb = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_bridging_og2_new2_20200327-210217/layer0_log.txt"
"""
/Users/zber/ProgramDev/exp_pyTorch/results/grow_bridging_og2_new2_bad 
/Users/zber/ProgramDev/exp_pyTorch/results/grow_bridging_og2_new2_good 
/Users/zber/ProgramDev/exp_pyTorch/results/grow_randomMap_og2_new2_bad 
/Users/zber/ProgramDev/exp_pyTorch/results/grow_randomMap_og2_new2_good 
/Users/zber/ProgramDev/exp_pyTorch/results/grow_rankconnect_og2_new2_bad 
/Users/zber/ProgramDev/exp_pyTorch/results/grow_rankconnect_og2_new2_good 
/Users/zber/ProgramDev/exp_pyTorch/results/grow_rankgroup_og2_new2_bad 
/Users/zber/ProgramDev/exp_pyTorch/results/grow_rankgroup_og2_new2_good
"""

good1 = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rankconnect_ogn_newn_2_1/layer0_log.txt"
good2 = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rankconnect_ogn_newn_2/layer0_log.txt"
good3 = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rankconnect_ogn_newn_3/layer0_log.txt"

bad2 = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rankconnect_ogn_newn_4down/layer0_log.txt"
bad1 = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rankconnect_ogn_newn_bad/layer0_log.txt"

nono_bad2 = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rankconnect_ognon-scale_newnon-scale_bad2/layer0_log.txt"
nono_bad1 = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rankconnect_ognon-scale_newnon-scale_bad1/layer0_log.txt"

dic_base = dict(
    rank_baseline="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_baseline_6/layer1_log.txt",
    copy_n="/Users/zber/ProgramDev/exp_pyTorch/results/grow_copy_n_6/layer1_log.txt",
    copy_one="/Users/zber/ProgramDev/exp_pyTorch/results/grow_copy_one_6/layer1_log.txt",
    rank_ours="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_ours_11/layer1_log.txt",
    bridging="/Users/zber/ProgramDev/exp_pyTorch/results/grow_bridging_6/layer1_log.txt",
    rank_cumulative_cosine="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_cosine_1/layer1_log.txt",
    random="/Users/zber/ProgramDev/exp_pyTorch/results/grow_random_1/layer1_log.txt",
    ranklow_n="/Users/zber/ProgramDev/exp_pyTorch/results/grow_ranklow_n_20200408-232641/layer1_log.txt",
    activation_low="/Users/zber/ProgramDev/exp_pyTorch/results/grow_activation_low_20200409-165613/layer1_log.txt",
    rank_cumulative_cosine_low="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_cosine_low_20200409-203904/layer1_log.txt",
    rank_cumulative="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_1/layer1_log.txt",
    rank_cumulative_low="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_low_1/layer1_log.txt",
    standard="/Users/zber/ProgramDev/exp_pyTorch/results/grow_standard_3/grow_standard__json_in_out.json",
    rank_cumulative_cL_fH="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_convL_fcH/layer1_log.txt",
    rank_cumulative_cL_fH2="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_convL_fcH2/layer1_log.txt",
    random_gate="/Users/zber/ProgramDev/exp_pyTorch/results/grow_random_gate/layer1_log.txt",
    random_cumulative_cosine_rank="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_cosine_gate/layer1_log.txt"
)

dic_10 = dict(
    rank_baseline="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_baseline_20200411-201831/layer1_log.txt",
    copy_n="/Users/zber/ProgramDev/exp_pyTorch/results/grow_copy_n_20200411-202529/layer1_log.txt",
    copy_one="/Users/zber/ProgramDev/exp_pyTorch/results/grow_copy_one_20200411-203227/layer1_log.txt",
    random="/Users/zber/ProgramDev/exp_pyTorch/results/grow_random_20200411-203926/layer1_log.txt",
    bridging="/Users/zber/ProgramDev/exp_pyTorch/results/grow_bridging_20200411-204627/layer1_log.txt",
    ranklow_n="/Users/zber/ProgramDev/exp_pyTorch/results/grow_ranklow_n_20200411-230501/layer1_log.txt",
    ranklow_one="/Users/zber/ProgramDev/exp_pyTorch/results/grow_ranklow_one_20200411-231200/layer1_log.txt",
    activation_low="/Users/zber/ProgramDev/exp_pyTorch/results/grow_activation_low_20200411-225803/layer1_log.txt",
    rank_cumulative_cosine="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_cosine_20200411-231903/layer1_log.txt",
    rank_cumulative="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_20200411-232608/layer1_log.txt",
    rank_cumulative_cosine_low="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_cosine_low_20200411-233529/layer1_log.txt",
    rank_cumulative_low="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_low_20200411-234224/layer1_log.txt",
    rank_cumulative_cL_fH="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_cL_fH_20200412-000301/layer1_log.txt",
    rank_cumulative_cL_fH2="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_cL_fH2/layer2_log.txt",
    standard="/Users/zber/ProgramDev/exp_pyTorch/results/grow_standard_20200412-182155/grow_standard__json_dic.json",
)

dic_lr = dict(
    bridging_lr="/Users/zber/ProgramDev/exp_pyTorch/results/grow_bridging_LR_20200409-211253/layer1_log.txt",
    ranklow_one_lr="/Users/zber/ProgramDev/exp_pyTorch/results/grow_ranklow_one_LR_20200409-215528/layer1_log.txt",
    rank_cumulative_cosine_low_lr="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_cosine_LR_20200409-212445/layer1_log.txt",
    rank_cumulative_LR="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_LR_20200410-001857/layer1_log.txt",
)

rank_baseline_bn = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_baseline_BN_1/layer1_log.txt"
copy_n_bn = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_copy_n_BN_1/layer1_log.txt"
copy_one_bn = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_copy_one_BN_1/layer1_log.txt"
rank_ours_bn = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_ours_BN_1/layer1_log.txt"
bridging_bn = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_bridging_BN_1/layer1_log.txt"
rank_cumulative_cosine_bn = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_cosine_BN_1/layer1_log.txt"
random_bn = '/Users/zber/ProgramDev/exp_pyTorch/results/grow_random_bn_1/layer1_log.txt'

# path_list = []
# for i in range(1, 6):
#     p = base_path.format(i=i)
#     path_list.append(p)
num_growth = 10
dic = dic_base


def str_to_float(str):
    num_list = []
    str = str[1:-1]
    str_arry = str.split(', ')
    for num in str_arry:
        num_list.append(float(num))
    return np.asarray(num_list)


def print_acc(file):
    acc = []
    acc_str = ''
    with open(file, "r") as f:
        for line in f:
            if line.startswith('Test'):
                acc.append(line[-8:-3])

    path = os.path.dirname(file)

    files = [f for f in glob.glob(path + "/*__log.txt")]

    with open(files[0], "r") as f:
        for line in f:
            if line.startswith('Accuracy:'):
                acc_str = line[9:-1]

    line = 0
    delta_acc = []
    while line < len(acc):
        pre_acc = 0
        for i in [0, 1, 3, 5]:  #:
            if i == 5:
                print('{}'.format(acc[line + i]), end='\n')
            else:
                print('{} -> '.format(acc[line + i]), end='')

            if pre_acc == 0:
                pre_acc = acc[line + i]
            else:
                delta = float(pre_acc) - float(acc[line + i])
                pre_acc = acc[line + i]
                delta_acc.append(delta)
        line += 6

    print('acc: ' + acc_str)
    print('acc drop : ' + str(np.sum(delta_acc) / (num_growth - 1)))


def print_all_acc(file):
    acc = []
    with open(file, "r") as f:
        for line in f:
            if line.startswith('Accuracy:'):
                print(line)
                break
    print()
    print()


def grow_x_y_exist(dic):
    y = []
    for e in range(1, dic["epoch_to"]):
        for n in range(1, (600 // dic["num_batch"]) + 1):
            batches = []
            for b in range(1, dic["num_batch"] + 1):
                s = data[str(e)][str(b * n - 1)][dic["layer"]]["S"]["out"]
                f = str_to_float(s)
                start = 0
                end = dic["grow_base_size"][dic["layer"]]
                f_need = f[start:end]
                f_mean = np.mean(f_need)
                batches.append(f_mean)
            y.append(np.mean(batches))
    length = len(y)
    x = np.arange(0, length)
    return x, y


def grow_x_y(dic, gi=1):
    y = []
    for e in range(dic["epoch_grow"][gi], dic["epoch_to"]):
        for n in range(1, (600 // dic["num_batch"]) + 1):
            batches = []
            for b in range(1, dic["num_batch"] + 1):
                s = data[str(e)][str(b * n - 1)][dic["layer"]]["S"]["out"]
                f = str_to_float(s)
                start = dic["grow_base_size"][dic["layer"]] + (gi - 1) * dic["grow_size"][dic["layer"]]
                end = start + dic["grow_size"][dic["layer"]]
                f_need = f[start:end]
                f_mean = np.mean(f_need)
                batches.append(f_mean)
            y.append(np.mean(batches))
    length = len(y)
    start_x = (dic["epoch_grow"][gi] - 1) * (600 // dic["num_batch"])
    x = np.arange(start_x, start_x + length)
    return x, y


def grow_x_y_exist_in(dic):
    y = []
    for e in range(1, dic["epoch_to"]):
        for n in range(1, (600 // dic["num_batch"]) + 1):
            batches = []
            for b in range(1, dic["num_batch"] + 1):
                s = data[str(e)][str(b * n - 1)][dic["layer"]]["S"]["in"]
                f = str_to_float(s)
                start = 0
                end = dic["grow_base_size"][dic["layer"]]
                f_need = f[start:end]
                f_mean = np.mean(f_need)
                batches.append(f_mean)
            y.append(np.mean(batches))
    length = len(y)
    x = np.arange(0, length)
    return x, y


def grow_x_y_in(dic, gi=1):
    y = []
    for e in range(dic["epoch_grow"][gi], dic["epoch_to"]):
        for n in range(1, (600 // dic["num_batch"]) + 1):
            batches = []
            for b in range(1, dic["num_batch"] + 1):
                s = data[str(e)][str(b * n - 1)][dic["layer"]]["S"]["in"]
                f = str_to_float(s)
                start = dic["grow_base_size"][dic["layer"]] + (gi - 1) * dic["grow_size"][dic["layer"]]
                end = start + dic["grow_size"][dic["layer"]]
                f_need = f[start:end]
                f_mean = np.mean(f_need)
                batches.append(f_mean)
            y.append(np.mean(batches))
    length = len(y)
    start_x = (dic["epoch_grow"][gi] - 1) * (600 // dic["num_batch"])
    x = np.arange(start_x, start_x + length)
    return x, y


# for f in [bb]:
#     print_acc(f)


def gen_x_y(is_out=True):
    x_y = []
    if is_out:
        for gi in range(num_growth):
            if gi == 0:
                x_y.append(grow_x_y_exist(dic_layer_out))
            else:
                x_y.append(grow_x_y(dic_layer_out, gi=gi))
    else:
        for gi in range(num_growth):
            if gi == 0:
                x_y.append(grow_x_y_exist_in(dic_layer_in))
            else:
                x_y.append(grow_x_y_in(dic_layer_in, gi=gi))
    return x_y


if __name__ == '__main__':
    dic_layer_out = {
        "epoch_from": 1,
        "layer": "1",
        "epoch_to": 11,
        "epoch_grow": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # [1, 2, 4, 7, 10],  # [1, 1, 1, 1, 1] #
        "num_batch": 50,
        "grow_base_size": {"0": 2, "1": 5, "2": 10},
        "grow_size": {"0": 2, "1": 3, "2": 10},
        "exsit_index": {"0": (0, 2), "1": (0, 5), "2": (0, 10)},
    }

    dic_layer_in = {
        "epoch_from": 1,
        "layer": "1",
        "epoch_to": 11,
        "epoch_grow": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # [1, 2, 4, 7, 10],  # [1, 1, 1, 1, 1],
        "num_batch": 50,
        "grow_base_size": {"1": 2, "2": 80, "3": 10},
        "grow_size": {"1": 2, "2": 48, "3": 10},
        "exsit_index": {"1": (0, 2), "2": (0, 80), "3": (0, 10)},
    }

    dic_element = {'Grow_1': [], 'Grow_2': [], 'Grow_3': [], 'Grow_4': [], 'Grow_5': [], 'Grow_6': [], 'Grow_7': [],
                   'Grow_8': [], 'Grow_9': []}
    dic_content = {'out': {'0': [],
                           '1': [],
                           '2': [], },
                   'in': {'1': [],
                          '2': [],
                          '3': []}}

    for key in dic.keys():
        if key == 'standard':
            dic_layer_in["epoch_grow"] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            dic_layer_out["epoch_grow"] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        else:
            dic_layer_in["epoch_grow"] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            dic_layer_out["epoch_grow"] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        print(key, ':')
        print_acc(dic[key])

        path_to_file = dic[key]

        path = os.path.dirname(path_to_file)

        files = [f for f in glob.glob(path + "**/*__json_in_out.json")]

        filename = files[0]

        with open(filename, 'r') as f:
            data = json.load(f)
        y = []
        for out in [True, False]:
            is_out = out
            if is_out:
                for layer in ["0", "1", "2"]:  #
                    dic_layer_out["layer"] = layer
                    x_y = gen_x_y(is_out=is_out)
                    existing = np.asarray(x_y[0][1])
                    sum_y = []
                    for i, label in enumerate(dic_element.keys()):
                        y = np.asarray(x_y[i + 1][1])
                        length = len(y)
                        delta_y = np.abs(existing[-length:] - y).tolist()
                        sum_y = sum_y + delta_y

                        # print('Out,Layer{},{}:{:.5f}'.format(layer, label, delta_y))
                    mean_y = np.mean(sum_y)
                    dic_content['out'][layer].append((key, mean_y))
                print()
            else:
                for layer in ["1", "2", "3"]:  #
                    dic_layer_in["layer"] = layer
                    x_y = gen_x_y(is_out=is_out)
                    existing = np.asarray(x_y[0][1])
                    sum_y = []
                    for i, label in enumerate(dic_element.keys()):
                        y = np.asarray(x_y[i + 1][1])
                        length = len(y)
                        delta_y = np.abs(existing[-length:] - y).tolist()
                        sum_y = sum_y + delta_y

                        # print('Out,Layer{},{}:{:.5f}'.format(layer, label, delta_y))
                    mean_y = np.mean(sum_y)
                    dic_content['in'][layer].append((key, mean_y))
        print()

    for i_o in ["out", "in"]:
        for l in dic_content[i_o].keys():
            print('{}, L{}:'.format(i_o, l))
            list_c = dic_content[i_o][l]
            list_c.sort(key=lambda tup: tup[1])
            for mode, value in list_c:
                print('{}: {:.8f}'.format(mode, value))
            print()
            print()
