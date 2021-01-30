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

    x1 = [0, 0, 0] + data["control"]
    if len(x1) == len(x):
        x = np.asarray(x) + np.asarray(x1)
        x = x.tolist()

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
    data_dic = {}
    save_dic = {}

    dic = dict(
        # mnist
        # vnew1=("/Users/zber/ProgramDev/exp_pyTorch/results/v2_Lenet/GC_MNIST_LeNet_3", 19),
        # vnew2=("/Users/zber/ProgramDev/exp_pyTorch/results/v2_Lenet/GC_MNIST_LeNet_best", 19),
        # vnewnonstop1=("/Users/zber/ProgramDev/exp_pyTorch/results/v2_Lenet/GC_MNIST_LeNet_best", 20),
        # vnew_opt=("/Users/zber/ProgramDev/exp_pyTorch/results/v2_M2_result/Standard_MNIST_LeNet_20210127-212705", 20),

        # har
        # standard1 = "",
        # standard2 = "",
        # vold1=("/Users/zber/ProgramDev/exp_pyTorch/results/Har_MB/GC_Har_MobileNet_H3_best1", 20),
        # vold2=("/Users/zber/ProgramDev/exp_pyTorch/results/Har_MB/GC_Har_MobileNet_H3_best3", 20),
        # vnew1=("/Users/zber/ProgramDev/exp_pyTorch/results/v2_MBnet/GC_Har_MobileNet_best5", 18),
        # vnew2=("/Users/zber/ProgramDev/exp_pyTorch/results/v2_MBnet/GC_Har_MobileNet_bestbest", 18),
        # vnewnonstep1=("/Users/zber/ProgramDev/exp_pyTorch/results/v2_MBnet/GC_Har_MobileNet_best5", 20),
        # vnewnonstep2=("/Users/zber/ProgramDev/exp_pyTorch/results/v2_MBnet/GC_Har_MobileNet_bestbest", 20),
        # standard2=("/Users/zber/ProgramDev/exp_pyTorch/results/Har_MB/Standard_Har_MobileNet_20200711-010005", 15),
        vnew_opt=("/Users/zber/ProgramDev/exp_pyTorch/results/v2_M2_result/Standard_Har_MobileNet_20210127-212353", 21)
    )

    for key in dic.keys():
        path, length = dic[key][0], dic[key][1]
        res = "".join(re.split("[^a-zA-Z]*", key))

        x = read_time(path)
        y = read_acc(path)
        y = np.asarray(y).reshape(-1, 1)[:length]
        if res in data_dic:
            new_y = np.asarray(data_dic[res][1])
            y = np.hstack((y, new_y)).tolist()
            data_dic[res] = (x, y, length)
        else:
            data_dic[res] = (x, y, length)

    for key in data_dic.keys():
        l = data_dic[key][2]
        x = data_dic[key][0][:l]
        y = data_dic[key][1]
        y_mean = np.mean(y, axis=1).tolist()[:l]
        y_std = np.std(y, axis=1).tolist()[:l]

        # mnist time

        # x = np.asarray(x) * 3.5 / 60
        # har time
        x = np.asarray(x) * 19.02
        # x = np.asarray(x)
        x = x.tolist()

        save_dic[key] = (x, y_mean, y_std)
        # get mean with length

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

        # for key in data.keys():
        #     acm = 0
        #     time = []
        #     for d in data[key]['time']:
        #         t = acm + d
        #         time.append(t)
        #         acm += d
        #     new_time = np.asarray(time) * 4.3 / 60
        #     data[key]['time'] = new_time.tolist()
        #
        #     ac1 = np.asarray(data[key]['ac1'])
        #     ac2 = np.asarray(data[key]['ac2'])
        #     ac3 = np.asarray(data[key]['ac3'])
        #     ac = np.vstack((ac1, ac2))
        #     ac = np.vstack((ac, ac3))
        #     mean = np.mean(ac, axis=0)
        #     std = np.std(ac, axis=0)
        #     data[key]['mean'] = mean.tolist()
        #     data[key]['std'] = std.tolist()
        #
    # target_path = "/Users/zber/Documents/FGdroid/result/6-8/df05_costDiff_A6H4/loss_our12.json"
    # with open(target_path, 'w') as f:
    #     json.dump(data, f, indent=4)

    # target_path = "/Users/zber/Desktop/v2_result/4_Time_ACC/lenet_mnist.json"
    target_path = "/Users/zber/Documents/FGdroid/exp_result/v2_result/4_Time_ACC/mbnet_har_v2_revision_opt_1.json"
    with open(target_path, 'w') as f:
        json.dump(save_dic, f, indent=4)
