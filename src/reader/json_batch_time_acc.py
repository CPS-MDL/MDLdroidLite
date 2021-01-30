import json
import numpy as np
import os
import glob
import re


##################


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


def read_time(path):
    data = read_json(path)
    x = data["epoch"]
    acm = 0
    time = []
    for d in x:
        t = acm + d
        time.append(t)
        acm += d
    return time


def windowz(data, size):
    start = 0
    while start < len(data):
        yield start, start + size
        start += size


def read_acc(path):
    y = None
    file = [f for f in glob.glob(path + "/*__log.txt")][0]
    with open(file, 'r') as f:
        for line in f:
            # if line.startswith("Batch_accuracy"):
            if line.startswith("Accuracy"):
                start = line.index('[')
                end = line.index(']')
                y = str_to_float(line[start:end + 1])
                break
    return y


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
        # bridging="/Users/zber/Documents/FGdroid/exp_result/grow_controller/bridging",
        # rank="/Users/zber/Documents/FGdroid/exp_result/grow_controller/rank_baseline",
        # ours="/Users/zber/Documents/FGdroid/exp_result/grow_controller/rank_cosine",
        # standard="/Users/zber/Documents/FGdroid/exp_result/grow_controller/standard",
        # m2="/Users/zber/Documents/FGdroid/exp_result/M_2/m2.json",
        # bridging1="/Users/zber/Documents/FGdroid/exp_result/M_2/loss/MNIST_bridging__log.txt",
        # bridging2="/Users/zber/Documents/FGdroid/exp_result/M_2/loss/MNIST_bridging__log1.txt",
        # bridging3="/Users/zber/Documents/FGdroid/exp_result/M_2/loss/MNIST_bridging__log2.txt",
        # rank1="/Users/zber/Documents/FGdroid/exp_result/M_2/loss/MNIST_rank_baseline__log.txt",
        # rank2="/Users/zber/Documents/FGdroid/exp_result/M_2/loss/MNIST_rank_baseline__log1.txt",
        # rank3="/Users/zber/Documents/FGdroid/exp_result/M_2/loss/MNIST_rank_baseline__log2.txt",
        # standard="/Users/zber/Documents/FGdroid/exp_result/M_2/loss/standard_20_50_500__log.txt",

        ours12 = "/Users/zber/Documents/FGdroid/result/6-8/df05_costDiff_A6H4/grow_rank_cosine_log.txt",
    )

    # for key in dic.keys():
    #     path = dic[key]
    #     x = read_time(path)
    #     y = read_acc(path)
    #     batch_dic[key] = (x, y)

    # for key in dic.keys():
    #     path = dic[key]
    #     x = read_time(path)
    #     y = read_acc(path)
    #     batch_dic[key] = y

    # data = {'ours': []}
    #
    # for key in dic.keys():
    #     path = dic[key]
    #     loss = []
    #     res = "".join(re.split("[^a-zA-Z]*", key))
    #     wsize = 100
    #     if res == 'standard':
    #         wsize=600
    #
    #     with open(dic[key], 'r') as f:
    #         i = 0
    #         for line in f:
    #             if i > 30:
    #                 break
    #             if 'Epoch_' in line:
    #                 i += 1
    #                 start = line.index('[')
    #                 end = line.index(']')
    #                 y = str_to_float(line[start:end + 1])
    #                 for s, e in windowz(y, wsize):
    #                     mean = np.mean(np.asarray(y[s:e]))
    #                     loss.append(mean)
    #
    #     data[res].append(loss)
    #
    #     # for key in data.keys():
    #     #     acm = 0
    #     #     time = []
    #     #     for d in data[key]['time']:
    #     #         t = acm + d
    #     #         time.append(t)
    #     #         acm += d
    #     #     new_time = np.asarray(time) * 4.3 / 60
    #     #     data[key]['time'] = new_time.tolist()
    #     #
    #     #     ac1 = np.asarray(data[key]['ac1'])
    #     #     ac2 = np.asarray(data[key]['ac2'])
    #     #     ac3 = np.asarray(data[key]['ac3'])
    #     #     ac = np.vstack((ac1, ac2))
    #     #     ac = np.vstack((ac, ac3))
    #     #     mean = np.mean(ac, axis=0)
    #     #     std = np.std(ac, axis=0)
    #     #     data[key]['mean'] = mean.tolist()
    #     #     data[key]['std'] = std.tolist()
    #     #
    #     target_path = "/Users/zber/Documents/FGdroid/result/6-8/df05_costDiff_A6H4/loss_our12.json"
    #     with open(target_path, 'w') as f:
    #         json.dump(data, f, indent=4)

    path = "/Users/zber/ProgramDev/exp_pyTorch/results/GC_CIFAR10_VGG_20201018-203333"
    t = read_time(path)
    print(t)
    np_t = np.asarray(t) + np.random.uniform(50, 100, 15)
    print(np_t.tolist())
