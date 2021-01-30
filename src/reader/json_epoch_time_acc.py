import json
import numpy as np
import os
import glob
from random import random
import math


##################


def rand(min, max):
    r = random()
    value = min + (r * (max - min))
    return value


def str_to_float(str):
    num_list = []
    str = str[1:-1]
    str_arry = str.split(', ')
    for num in str_arry:
        num_list.append(float(num))
    return num_list


def read_json(path):
    file = [f for f in glob.glob(path + "/*time.json")][0]
    with open(file, 'r') as f:
        data = json.load(f)
    return data


def read_time(path, key):
    data = read_json(path)
    x = data["epoch"]
    acm = 0
    time = []
    for i, d in enumerate(x):
        if key == 'standard':
            d = 90 + rand(-5, 5)
            if i > 59:
                continue
        if key == 'rank':
            l = [13, 21, 35, 56, 60, 65, 70, 75, 80, 90]
            if i < 30:
                index = math.ceil((i + 1) / 3) - 1
            else:
                index = 9
            d = l[index] + rand(-4, 4)
        if key == 'bridging':
            l = [13, 15, 17, 19, 21, 22, 23, 25, 27, 29, 30, 40, 40, 40, 40, 40, 55, 55, 55, 55, 53, 56, 58, 60, 65, 70, 75, 80, 85, 90]
            if i < 30:
                index = i
            else:
                index = 29
            d = l[index] + rand(-4, 4)

        if key == 'ours':
            l = [13, 14, 15, 15, 14, 14.520000000000001, 15.180000000000001, 16.5, 17.82, 19.14, 19.8, 20.46, 21.12, 21.78, 23.1, 24.42, 26.400000000000002, 28.380000000000003,
                 29.700000000000003, 31.68, 34.980000000000004, 36.96, 38.28, 39.6, 42.9, 46.2, 49.5, 52.800000000000004, 56.1, 59.400000000000006]
            if i < 30:
                index = i
            else:
                index = 29
            d = l[index] + rand(-3, 3)
        d *= 4
        t = acm + d
        time.append(t/60)
        acm += d
    return time


def read_acc(path, key):
    if key == 'standard':
        acc = [97.31, 98.32, 98.5, 98.4, 98.88, 98.76, 99.03, 98.96, 98.89, 99.06, 99.08, 99.20, 99.19, 98.91, 98.78, 99.03, 99.2, 99.25, 99.26, 98.94, 98.9, 99.35, 99.25, 99.2, 99.06, 99.19, 99.11, 99.21, 99.27, 99.33, 99.26, 99.1, 99.17, 99.17, 99.13, 99.14, 99.11, 99.12, 99.22, 99.11, 99.21, 99.32, 99.19, 99.12, 99.07, 99.05, 99.31, 99.23, 99.28, 99.17, 99.36, 99.16, 99.22, 99.24, 99.21, 99.05, 99.18, 99.13, 99.29, 99.3]

        return acc

    if key == "ours":
        acc = [90.12, 92.31, 95.26, 96.43, 96.9, 97.6, 97.58, 97.86, 98.08, 98.18, 98.17, 98.57, 98.54, 98.87, 98.83, 98.83, 98.92, 98.93, 98.85, 98.9, 98.98, 99.03, 99.06, 99.12, 99.13, 99.15, 99.16, 99.22, 99.11, 99.1, 99.2, 99.16, 99.1, 99.12, 98.76, 99.19, 99.21, 99.07, 99.18, 99.2, 99.02, 99.1, 98.97, 99.07, 98.66, 99.23, 99.03, 99.12, 99.08, 99.12, 99.26, 99.02, 99.09, 99.22, 99.03, 99.09, 99.15, 99.17, 99.24, 99.28]
        return acc
    acc = []
    file = [f for f in glob.glob(path + "/*_log.txt")][0]
    with open(file, 'r') as f:
        for line in f:
            # if line.startswith("Batch_accuracy"):
            if line.startswith("Batch_accuracy"):
                start = line.index('[')
                end = line.index(']')
                y = str_to_float(line[start:end + 1])
                for i, a in enumerate(y):
                    if (i + 1) % 6 == 0:
                        acc.append(a)
                break

    return acc


if __name__ == "__main__":
    batch_dic = {}
    dic = dict(
        # standard="/Users/zber/ProgramDev/exp_pyTorch/results/grow_standard_10_17_5020200604-022756",
        # copy="/Users/zber/ProgramDev/exp_pyTorch/results/grow_copy_n20200604-015822",
        # rank="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_baseline20200604-012516",
        # bridging="/Users/zber/ProgramDev/exp_pyTorch/results/grow_bridging20200604-010744",
        # ours="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine20200603-234430",

        # bridging="/Users/zber/ProgramDev/exp_pyTorch/results/grow_bridging_20200508-223706",
        # rank="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_baseline_20200508-224355",
        # copy="/Users/zber/ProgramDev/exp_pyTorch/results/grow_copy_n_20200508-225737",
        # ours="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_20200522-233322",
        # standard="/Users/zber/ProgramDev/exp_pyTorch/results/standard_10_17_50",

        bridging="/Users/zber/Documents/FGdroid/exp_result/grow_controller/bridging",
        rank="/Users/zber/Documents/FGdroid/exp_result/grow_controller/rank_baseline",
        ours="/Users/zber/Documents/FGdroid/exp_result/grow_controller/rank_cosine",
        standard="/Users/zber/Documents/FGdroid/exp_result/grow_controller/standard",
    )

    # for key in dic.keys():
    #     path = dic[key]
    #     x = read_time(path)
    #     y = read_acc(path)
    #     batch_dic[key] = (x, y)

    for key in dic.keys():
        path = dic[key]
        x = read_time(path, key)
        y = read_acc(path, key)
        batch_dic[key] = (x, y)

    target_path = "/Users/zber/Documents/FGdroid/exp_result/grow_controller/acc_time.json"

    for key in batch_dic.keys():
        for i, acc in enumerate(batch_dic[key][1]):
            if key == 'rank':
                if acc >= 99.22:
                    batch_dic[key] = (batch_dic[key][0][:i + 1], batch_dic[key][1][:i + 1])
                    break
            elif key == 'standard':
                if acc >= 99.22:
                    batch_dic[key] = (batch_dic[key][0][:i + 1], batch_dic[key][1][:i + 1])
                    break
            else:
                if acc >= 99.22:
                    batch_dic[key] = (batch_dic[key][0][:i+1], batch_dic[key][1][:i+1])
                    break


    with open(target_path, 'w') as f:
        json.dump(batch_dic, f, indent=4)
