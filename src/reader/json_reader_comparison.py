import json
import os
import glob
import numpy as np


##########################


def str_to_float(str):
    num_list = []
    str = str[1:-1]
    str_arry = str.split(', ')
    for num in str_arry:
        num_list.append(float(num))
    # return np.asarray(num_list)
    return num_list


def read_s(dic, path):
    file = [f for f in glob.glob(path + "/*out.json")][0]
    with open(file, 'r') as f:
        data = json.load(f)
    mean = []
    std = []
    for e in range(dic["epoch_from"], dic["epoch_to"]):
        for n in range(1, (600 // dic["num_batch"]) + 1):
            batches_mean = []
            batches_std = []
            for b in range(1, dic["num_batch"] + 1):
                s = data[str(e)][str(b * n - 1)][dic["layer"]]["S"]["out"]
                f = str_to_float(s)
                m = np.mean(f)
                s = np.std(f)
                batches_mean.append(m)
                batches_std.append(s)
            mean.append(np.mean(batches_mean))
            std.append(np.mean(batches_std))
    return mean, std


if __name__ == "__main__":
    dic_out = {
        "epoch_from": 1,
        "layer": "2",
        "epoch_to": 11,
        "epoch_grow": [1, 2, 4, 7, 10],  # [1, 1, 1, 1, 1],
        "num_batch": 50,
        "grow_base_size": {"0": 2, "1": 5, "2": 10},
        "grow_size": {"0": 2, "1": 3, "2": 10},
        "exsit_index": {"0": (0, 2), "1": (0, 5), "2": (0, 10)},
        "control_batches": 600
    }

    dic = dict(
        # bridging="/Users/zber/ProgramDev/exp_pyTorch/results/grow_bridging_20200508-223706",
        # rank="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_baseline_20200508-224355",
        # copy="/Users/zber/ProgramDev/exp_pyTorch/results/grow_copy_n_20200508-225737",
        # ours="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_20200522-233322",
        ours="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_20200522-233322",
        # standard="/Users/zber/ProgramDev/exp_pyTorch/results/standard_10_17_50",
        standard="/Users/zber/ProgramDev/exp_pyTorch/results/standard_10_17_50_20200513-174306",
    )

    s_dic = {}

    for key in dic.keys():
        path = dic[key]
        # x = read_time(path)
        y = read_s(dic_out, path)
        s_dic[key] = y

    target_path = "/Users/zber/Documents/FGdroid/exp_result/S_score_comparison/ours_layer2.json"

    with open(target_path, 'w') as f:
        json.dump(s_dic, f, indent=4)
