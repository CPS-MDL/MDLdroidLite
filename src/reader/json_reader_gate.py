import json
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from numpy import inf

dic = dict(
    # cosine_RI_600="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_20200519-002735",
    # cosine_RI="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_20200519-020328",
    # cosine_RI_T10="/Users/zber/ProgramDev/exp_pyTorch/results/cosine_RI_T-10_20200519-192937",
    # cosine_RI_T="/Users/zber/ProgramDev/exp_pyTorch/results/cosine_RI_T_20200519-204602"
    # cosine_RI="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_20200519-212733",
    # cosine_RI_back_T10="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_lambda_back_T10_20200519-214351",
    # cosine_RI_T10="/Users/zber/ProgramDev/exp_pyTorch/results/cosine_RI_back_T10_20200519-215807",
    # cosine_lambda_back_T3="/Users/zber/ProgramDev/exp_pyTorch/results/cosine_lambda_back_T3_20200519-224154",
    # cosine_RI_std="/Users/zber/ProgramDev/exp_pyTorch/results/cosine_RI_Sstd_20200520-000613",
    # cosine_RI_std="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_20200520-174858"
    # cosine_ri="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_20200520-201724",
    # cosine_ri="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_20200522-004826",
    # cosine_RI_gradGate_lambdaR="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_20200522-174535",
    # consine_RI_Sstd_weightNew_power="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_20200522-233322",
    # cosine_momentum="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_20200530-010944",
    # one_lambda="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_20200530-160804",
    # cosine_L_1_index = "/Users/zber/ProgramDev/exp_pyTorch/results/cosine_L_1_index_20200530-192920",
    # cosine_L_L_index = "/Users/zber/ProgramDev/exp_pyTorch/results/cosine_L_L_index_20200530-194103",
    # cosine_L_1L_index = "/Users/zber/ProgramDev/exp_pyTorch/results/cosine_L_1L_index_20200530-195356"
    # cosine_L1="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine20200605-200734",
    mb_har="/Users/zber/ProgramDev/exp_pyTorch/results/G_Har_MobileNet20200620-150322",
)


def str_to_float(str):
    num_list = []
    str = str[1:-1]
    str_arry = str.split(', ')
    for num in str_arry:
        num_list.append(float(num))
    return np.asarray(num_list)


def gen_data(dic, layer='0', approach='t-d'):  # lambda
    if approach == 't-d':
        dic_mode = {
            'min_distance': [],
            'distance': [],
        }
    elif approach == 'gate' or approach == 'grad_gate':
        dic_mode = {
            'old': [],
            'new': [],
        }
    elif approach == 'case':
        dic_mode = {
            # 'lr_case': [],
            'window_case': [],
        }
    elif approach == 'strength':
        dic_mode = {
            # 'lr_case': [],
            'strength': [],
        }
    elif approach == 'weight':
        dic_mode = {
            'old_weight': [],
            'new_weight': [],
        }

    for mode in dic_mode.keys():
        # for e in range(dic["epoch_from"], dic["epoch_to"]):
        for e in [2, 4, 7, 10]:
            for n in range(1, (dic["num_batches"] // dic["num_batch"]) + 1):
                batches = []
                for b in range(1, dic["num_batch"] + 1):
                    if (b * n - 1) == 0:
                        continue
                    if mode == 'old':
                        s = data[str(e)][str(b * n - 1)][layer][approach]
                        s = str_to_float(s)
                        f = s[0]
                    elif mode == 'new':
                        s = data[str(e)][str(b * n - 1)][layer][approach]
                        s = str_to_float(s)
                        f = s[-1]
                    else:
                        s = data[str(e)][str(b * n - 1)][layer][mode]
                        f = float(s)
                    batches.append(f)
                dic_mode[mode].append(np.nanmean(batches))

    if approach == 't-d':
        length = len(dic_mode['min_distance'])
    elif approach == 'gate' or approach == 'grad_gate':
        length = len(dic_mode['old'])
    elif approach == 'case':
        length = len(dic_mode['window_case'])
    elif approach == 'strength':
        length = len(dic_mode['strength'])
    elif approach == 'weight':
        length = len(dic_mode['old_weight'])

    start_x = (dic["epoch_from"] - 1) * (dic["num_batches"] // dic["num_batch"])
    x = np.arange(start_x, start_x + length)
    return x, dic_mode


def plot_save(x, dic, dir, layer='0', is_out=False):
    global mode
    global dic_layer
    # grow_epochs = [1, 3, 6, 9]
    # grow_epochs = [1, 2, 3, 4, 5]
    # grow_epochs = [1, 2, 3]
    grow_epochs = [1]

    colour_list = ['r--', 'b', 'g', 'yellow', 'black']
    path = os.path.join(dir, '{}_L{}.png'.format(mode, layer))
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 8))
    for key, colour in zip(dic.keys(), colour_list):
        if mode == 'case':
            ax.scatter(x, dic[key], label=key)
        else:
            ax.plot(x, dic[key], colour, label=key)
    for i in grow_epochs:
        point_x = i * (dic_layer["num_batches"] / dic_layer['num_batch']) - 1
        plt.axvline(x=point_x, color='grey', linestyle='--', alpha=0.5)
    ax.legend()
    fig.savefig(path)
    plt.close(fig)


if __name__ == '__main__':
    mode = "MobileNet"
    # mode = "LeNet"

    if mode == "MobileNet":
        dic_layer = {
            "epoch_from": 2,
            "epoch_to": 11,
            "epoch_grow": [1, 2, 4, 7, 10],  # [1, 1, 1, 1, 1],
            "num_batch": 1,
            "grow_base_size": {"0": 2, "1": 5, "2": 10},
            "grow_size": {"0": 1, "1": 2, "2": 4, "3": 8},
            "exsit_index": {"0": (0, 3), "1": (0, 6), "2": (0, 12), "3": (0, 25)},
            "num_batches": 230
        }
    else:
        dic_layer = {
            "epoch_from": 2,
            "epoch_to": 11,
            "epoch_grow": [1, 2, 4, 7, 10],  # [1, 1, 1, 1, 1],
            "num_batch": 23,
            "grow_base_size": {"0": 2, "1": 5, "2": 10},
            "grow_size": {"0": 2, "1": 3, "2": 10},
            "exsit_index": {"0": (0, 2), "1": (0, 5), "2": (0, 10)},
            "num_batches": 600
        }

    # mode = 't-d'
    # mode = 'case'
    # mode = 'grad_gate'
    # mode = 'gate'

    num_growth = 4

    for key in dic.keys():
        print(key, ':')

        path_to_file = dic[key]

        files = [f for f in glob.glob(path_to_file + "/*ate.json")]

        file = files[0]

        with open(file, 'r') as f:
            data = json.load(f)
            for mode in ['t-d', 'case', 'grad_gate', 'gate', 'weight', 'strength']:  #
                # for mode in ['grad_gate']:
                if mode == 'gate' or mode == 'grad_gate':
                    dic_layer['num_batch'] = 1
                for layer in ["0", "1", "2"]:  #
                    x, dic = gen_data(dic_layer, layer, approach=mode)
                    plot_save(x, dic, path_to_file, layer=layer)
