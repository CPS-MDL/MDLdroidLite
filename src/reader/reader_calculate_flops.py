import json
import torch
import glob
import os
import numpy as np

from ri_control.ri_controller import grab_input_weight_shape
from model.CNN import LeNet5_GROW_P, LeNet5
from model.VGG import VGG, VGG_BN
from model.summary import summary
from model.mobileNet import NetS
from ptflops import get_model_complexity_info

"""
mnist : bridging 36 epoch, ours 17 epoch, search 23 epoch , rank 30 epoch, full size 10 epoch
har: rank 29 epoch, bridging 25 epoch, ours 19 epoch, standard 12 epoch, search 10 epoch

"""


def str_to_int(str):
    num_list = []
    str = str[1:-1]
    str_arry = str.split(', ')
    for num in str_arry:
        num_list.append(int(num))
    return num_list


def str_to_float(str):
    num_list = []
    str = str[1:-1]
    str_arry = str.split(', ')
    for num in str_arry:
        num_list.append(float(num))
    return num_list


def read_size(file):
    # size_list = [[3, 6, 12, 25]]
    # size_list = [[2, 5, 10]]
    size_list = []

    with open(file, 'r') as f:
        for line in f:
            if "size" in line:
                start = line.index('[')
                end = line.index(']')
                y = str_to_int(line[start:end + 1])
                size_list.append(y)
    return size_list


def read_acc(file):
    y = None
    with open(file, 'r') as f:
        i = 0
        for line in f:
            if "Accuracy:[" in line:
                start = line.index('[')
                end = line.index(']')
                y = str_to_float(line[start:end + 1])
                break
    return y


def cal_flops_lenet(s_list):
    input_shape = (1, 28, 28)
    M = []
    P = []
    para_model = {'out1': 20, 'out2': 50, 'fc1': 500}
    for size in s_list:
        for s, key in zip(size, para_model.keys()):
            para_model[key] = s

        net = LeNet5_GROW_P(**para_model)
        macs, params = get_model_complexity_info(net, input_shape, as_strings=False,
                                                 print_per_layer_stat=False, verbose=False)
        M.append(macs / 1000000)
        P.append(params / 1000000)

    return M, P


def cal_flops_mobile_har(s_list):
    input_shape = (9, 1, 128)
    M = []
    P = []
    kwargs = {
        'in_channel': 9,
        'out1_channel': 32,
        'out2_channel': 64,
        'out3_channel': 128,
        'out4_channel': 256,
        'out_classes': 6,
        'kernel_size': 14,
        'avg_factor': 2
    }
    for size in s_list:
        if size == 0:
            pass
        else:
            for s, key in zip(size, ['out1_channel', 'out2_channel', 'out3_channel', 'out4_channel']):
                kwargs[key] = s

        net = NetS(**kwargs)
        macs, params = get_model_complexity_info(net, input_shape, as_strings=False,
                                                 print_per_layer_stat=False, verbose=False)
        M.append(macs / 1000000)
        P.append(params / 1000)

    return M, P


def cal_flops_mobile_myhealth(s_list):
    input_shape = (23, 1, 100)
    M = []
    P = []
    kwargs = {
        'in_channel': 23,
        'out1_channel': 32,
        'out2_channel': 64,
        'out3_channel': 128,
        'out4_channel': 256,
        'out_classes': 11,
        'kernel_size': 12,
        'avg_factor': 1
    }
    for size in s_list:
        if size == 0:
            pass
        else:
            for s, key in zip(size, ['out1_channel', 'out2_channel', 'out3_channel', 'out4_channel']):
                kwargs[key] = s

        net = NetS(**kwargs)
        macs, params = get_model_complexity_info(net, input_shape, as_strings=False,
                                                 print_per_layer_stat=False, verbose=False)
        M.append(macs / 1000000)
        P.append(params / 1000)

    return M, P


def cal_flops_mobile_fin(s_list):
    input_shape = (6, 1, 150)
    M = []
    P = []
    kwargs = {
        'in_channel': 6,
        'out1_channel': 32,
        'out2_channel': 64,
        'out3_channel': 128,
        'out4_channel': 256,
        'out_classes': 6,
        'kernel_size': 14,
        'avg_factor': 4
    }

    for size in s_list:
        for s, key in zip(size, ['out1_channel', 'out2_channel', 'out3_channel', 'out4_channel']):
            kwargs[key] = s

        net = NetS(**kwargs)
        macs, params = get_model_complexity_info(net, input_shape, as_strings=False,
                                                 print_per_layer_stat=False, verbose=False)
        M.append(macs / 1000000)
        P.append(params / 1000000)

    return M, P


def cal_flops_mobile_emg(s_list):
    input_shape = (8, 1, 100)
    M = []
    P = []
    kwargs = {
        'in_channel': 8,
        'out1_channel': 32,
        'out2_channel': 64,
        'out3_channel': 128,
        'out4_channel': 256,
        'out_classes': 6,
        'kernel_size': 12,
        'avg_factor': 1
    }

    for size in s_list:
        if size == 0:
            pass
        else:
            for s, key in zip(size, ['out1_channel', 'out2_channel', 'out3_channel', 'out4_channel']):
                kwargs[key] = s

        net = NetS(**kwargs)
        macs, params = get_model_complexity_info(net, input_shape, as_strings=False,
                                                 print_per_layer_stat=False, verbose=False)
        M.append(macs / 1000000)
        P.append(params / 1000)

    return M, P


def cal_flops_har(s_list):
    input_shape = (9, 1, 128)
    M = []
    P = []
    kwargs = {
        'in_channel': 9,
        'out1_channel': 2,
        'out2_channel': 5,
        'fc': 10,
        'out_classes': 6,
        'kernel_size': 14,
        'flatten_factor': 22
    }
    for size in s_list:
        for s, key in zip(size, ['out1_channel', 'out2_channel', 'fc']):
            kwargs[key] = s

        net = LeNet5(**kwargs)
        macs, params = get_model_complexity_info(net, input_shape, as_strings=False,
                                                 print_per_layer_stat=False, verbose=False)
        M.append(macs / 1000000)
        P.append(params / 1000000)

    return M, P


def cal_flops_lenet_emg(s_list):
    input_shape = (8, 1, 100)
    M = []
    P = []

    kwargs = {
        'in_channel': 8,
        'out1_channel': 2,
        'out2_channel': 5,
        'fc': 10,
        'out_classes': 6,
        'kernel_size': 14,
        'flatten_factor': 15
    }
    for size in s_list:
        for s, key in zip(size, ['out1_channel', 'out2_channel', 'fc']):
            kwargs[key] = s

        net = LeNet5(**kwargs)
        macs, params = get_model_complexity_info(net, input_shape, as_strings=False,
                                                 print_per_layer_stat=False, verbose=False)
        M.append(macs / 1000000)
        P.append(params / 1000)

    return M, P


def cal_flops_lenet_myHealth(s_list):
    input_shape = (23, 1, 100)
    M = []
    P = []

    kwargs = {
        'in_channel': 23,
        'out1_channel': 2,
        'out2_channel': 5,
        'fc': 10,
        'out_classes': 11,
        'kernel_size': 14,
        'flatten_factor': 15,
    }

    for size in s_list:
        for s, key in zip(size, ['out1_channel', 'out2_channel', 'fc']):
            kwargs[key] = s

        net = LeNet5(**kwargs)
        macs, params = get_model_complexity_info(net, input_shape, as_strings=False,
                                                 print_per_layer_stat=False, verbose=False)
        M.append(macs / 1000000)
        P.append(params / 1000)

    return M, P


def cal_flops_fin(s_list):
    input_shape = (6, 1, 150)
    M = []
    P = []
    kwargs = {
        'in_channel': 6,
        'out1_channel': 5,
        'out2_channel': 18,
        'fc': 32,
        'out_classes': 6,
        'kernel_size': 14,
        'flatten_factor': 27
    }
    for size in s_list:
        for s, key in zip(size, ['out1_channel', 'out2_channel', 'fc']):
            kwargs[key] = s

        net = LeNet5(**kwargs)
        macs, params = get_model_complexity_info(net, input_shape, as_strings=False,
                                                 print_per_layer_stat=False, verbose=False)
        M.append(macs / 1000000)
        P.append(params / 1000)

    return M, P


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


def read_json(path):
    file = [f for f in glob.glob(path + "/*time.json")][0]
    with open(file, 'r') as f:
        data = json.load(f)
    return data


if __name__ == "__main__":
    # findroid on lenet
    # size_list = [[6, 4, 55], [9, 32, 49], [9, 35, 54], [20, 50, 500]]
    # M, P = cal_flops_fin(size_list)
    # print(M)
    # print(P)

    # findroid on MB
    # size_list =[[15, 18, 16, 31],[8, 10, 35, 40],[15, 18, 16, 31],[32,64,128,256]]
    # size_list = [[10, 29, 35, 44], [10, 20, 43, 37], [19, 15, 24, 39], [25, 33, 39, 58]]
    # size_list = [[8, 13, 30, 38], [8, 15, 35, 43], [9, 18, 31, 45], [6, 13, 36, 45]]
    #
    # M, P = cal_flops_mobile(size_list)
    # print(M)
    # print(P)

    # mnist on Lenet
    # size_list = [[6, 8, 28], [18, 25, 42], [16, 23, 44], [20, 50, 500]]
    # size_list= [[20, 27, 46], [18, 38, 50], [20, 30, 31], [10, 23, 42]]
    # size_list = [[5, 25, 40], [7, 27, 41], [8, 27, 40], [6, 31, 42]]
    # M, P = cal_flops_lenet(size_list)
    # print(M)
    # print(P)

    # har size list
    # file = "/Users/zber/ProgramDev/exp_pyTorch/results/GC_Har_LeNet_v1_best/size_structure.txt"
    # file = "/Users/zber/Documents/FGdroid/exp_result/G_2_resource/Har/ours.txt"
    # size_list= read_size(file)
    # f_list = cal_flops_har(size_list)
    # print(f_list)

    m_list = [('ours', 17), ('standard', 10), ('rank', 30), ('bridging', 36), ('search', 23)]
    h_list = [('ours', 19), ('standard', 12), ('rank', 29), ('bridging', 25), ('search', 10)]

    m_time_path = "/Users/zber/Documents/FGdroid/exp_result/G_1_acc_to_time/acc_time_minist_all.json"
    h_time_path = "/Users/zber/Documents/FGdroid/exp_result/G_1_acc_to_time/acc_time_har.json"

    with open(m_time_path, 'r') as f:
        mdata = json.load(f)

    with open(h_time_path, 'r') as f:
        hdata = json.load(f)

    dic = dict(
        # har
        # ours="/Users/zber/Documents/FGdroid/exp_result/G_2_resource/Har/ours.txt",
        # standard="/Users/zber/Documents/FGdroid/exp_result/G_2_resource/Har/standard.txt",
        # rank="/Users/zber/Documents/FGdroid/exp_result/G_2_resource/Har/rank.txt",
        # bridging="/Users/zber/Documents/FGdroid/exp_result/G_2_resource/Har/bridging.txt",
        # search="/Users/zber/Documents/FGdroid/exp_result/G_2_resource/Har/search.txt",

        # mnist
        # ours="/Users/zber/Documents/FGdroid/exp_result/G_2_resource/MNIST/ours.txt",
        # standard="/Users/zber/Documents/FGdroid/exp_result/G_2_resource/MNIST/standard.txt",
        # rank="/Users/zber/Documents/FGdroid/exp_result/G_2_resource/MNIST/rank.txt",
        # bridging="/Users/zber/Documents/FGdroid/exp_result/G_2_resource/MNIST/bridging.txt",
        # search="/Users/zber/Documents/FGdroid/exp_result/G_2_resource/MNIST/search.txt",

        # mobile net
        # H1="/Users/zber/Documents/FGdroid/exp_result/G_4_Horizon/MobileNet_Horizon/GC_Har_MobileNet_H1/GC_Har_MobileNet__log.txt",
        # H2="/Users/zber/Documents/FGdroid/exp_result/G_4_Horizon/MobileNet_Horizon/GC_Har_MobileNet_H2_1/GC_Har_MobileNet__log.txt",
        # H3="/Users/zber/Documents/FGdroid/exp_result/G_4_Horizon/MobileNet_Horizon/GC_Har_MobileNet_H3_best1/GC_Har_MobileNet__log.txt",
        # H4="/Users/zber/Documents/FGdroid/exp_result/G_4_Horizon/MobileNet_Horizon/GC_Har_MobileNet_H4_2/GC_Har_MobileNet__log.txt",
        # H5="/Users/zber/Documents/FGdroid/exp_result/G_4_Horizon/MobileNet_Horizon/GC_Har_MobileNet_h5_best/GC_Har_MobileNet__log.txt",

        # lenet
        # H1="/Users/zber/Documents/FGdroid/exp_result/G_4_Horizon/LeNet_Horizon/H1/GC_MNIST_LeNet__log.txt",
        # H2="/Users/zber/Documents/FGdroid/exp_result/G_4_Horizon/LeNet_Horizon/H2_1/GC_MNIST_LeNet__log.txt",
        # H3="/Users/zber/Documents/FGdroid/exp_result/G_4_Horizon/LeNet_Horizon/H3/GC_MNIST_LeNet__log.txt",
        # H4="/Users/zber/Documents/FGdroid/exp_result/G_4_Horizon/LeNet_Horizon/H4_Best/GC_MNIST_LeNet__log.txt",
        # H5="/Users/zber/Documents/FGdroid/exp_result/G_4_Horizon/LeNet_Horizon/H5/GC_MNIST_LeNet__log.txt",

        # v2_lenet_mnist
        # H1="/Users/zber/ProgramDev/exp_pyTorch/results/v2_Lenet/GC_MNIST_LeNet_TH1_1/GC_MNIST_LeNet__log.txt",
        # H2="/Users/zber/ProgramDev/exp_pyTorch/results/v2_Lenet/GC_MNIST_LeNet_TH2/GC_MNIST_LeNet__log.txt",
        # H3="/Users/zber/ProgramDev/exp_pyTorch/results/v2_Lenet/GC_MNIST_LeNet_TH3/GC_MNIST_LeNet__log.txt",
        # H4="/Users/zber/ProgramDev/exp_pyTorch/results/v2_Lenet/GC_MNIST_LeNet_TH4/GC_MNIST_LeNet__log.txt",
        # H4="/Users/zber/ProgramDev/exp_pyTorch/results/v2_Lenet/GC_MNIST_LeNet_best4/GC_MNIST_LeNet__log.txt",
        # H5="/Users/zber/ProgramDev/exp_pyTorch/results/v2_Lenet/GC_MNIST_LeNet_TH5/GC_MNIST_LeNet__log.txt",

        # v2_mobileNet_Har
        # H1="/Users/zber/ProgramDev/exp_pyTorch/results/v2_MBnet/GC_Har_MobileNet_TH1/GC_Har_MobileNet__log.txt",
        # H2="/Users/zber/ProgramDev/exp_pyTorch/results/v2_MBnet/GC_Har_MobileNet_TH2/GC_Har_MobileNet__log.txt",
        # H3="/Users/zber/ProgramDev/exp_pyTorch/results/v2_MBnet/GC_Har_MobileNet_best5/GC_Har_MobileNet__log.txt",
        # H4="/Users/zber/ProgramDev/exp_pyTorch/results/v2_MBnet/GC_Har_MobileNet_TH3/GC_Har_MobileNet__log.txt",
        # H4="/Users/zber/ProgramDev/exp_pyTorch/results/v2_MBnet/GC_Har_MobileNet_TH4/GC_Har_MobileNet__log.txt",
        # H5="/Users/zber/ProgramDev/exp_pyTorch/results/v2_MBnet/GC_Har_MobileNet_TH5/GC_Har_MobileNet__log.txt",

        # v2_Lenet_Har
        # v2=("/Users/zber/ProgramDev/exp_pyTorch/results/v2_Lenet/GC_Har_LeNet_best1", 15, 2.87),
        # v2nonstop=("/Users/zber/ProgramDev/exp_pyTorch/results/v2_Lenet/GC_Har_LeNet_best1", 20, 2.87),
        # v2opt=("/Users/zber/ProgramDev/exp_pyTorch/results/v2_M2_result/Standard_Har_LeNet_20210127-214915", 16, 2.87),

        # v2_lenet_mnist
        # v2=("/Users/zber/ProgramDev/exp_pyTorch/results/v2_Lenet/GC_MNIST_LeNet_best",20, 3.5)
        # v2nonstop=("/Users/zber/ProgramDev/exp_pyTorch/results/v2_Lenet/GC_MNIST_LeNet_best", 20, 3.6)
        # v2opt=("/Users/zber/ProgramDev/exp_pyTorch/results/v2_M2_result/Standard_MNIST_LeNet_20210127-212705", 20, 3.6),

        # mbnet_har
        # v1=("/Users/zber/ProgramDev/exp_pyTorch/results/Har_MB/GC_Har_MobileNet_H3_best1",  20, 19.02),
        # v2=("/Users/zber/ProgramDev/exp_pyTorch/results/v2_MBnet/GC_Har_MobileNet_best5", 12, 19.02),
        # v2=("/Users/zber/ProgramDev/exp_pyTorch/results/v2_MBnet/GC_Har_MobileNet_bestbest", 19, 19.02),
        # v2nonstop=("/Users/zber/ProgramDev/exp_pyTorch/results/v2_MBnet/GC_Har_MobileNet_bestbest", 19, 19.02),
        # v2opt=("/Users/zber/ProgramDev/exp_pyTorch/results/v2_M2_result/Standard_Har_MobileNet_20210127-212353", 21, 19.02),
        # standard=("/Users/zber/ProgramDev/exp_pyTorch/results/Har_MB/Standard_Har_MobileNet_20200711-010005", 15, 19.02),

        # myhealth
        # v1=("/Users/zber/ProgramDev/exp_pyTorch/results/myHealth_MB/GC_myHealth_MobileNet_best", 36, 10.83),
        # v2=("/Users/zber/ProgramDev/exp_pyTorch/results/v2_MBnet/GC_myHealth_MobileNet_best", 25, 10.83),
        # v2nonstop=("/Users/zber/ProgramDev/exp_pyTorch/results/v2_MBnet/GC_myHealth_MobileNet_best", 25, 10.83),
        v2opt=("/Users/zber/ProgramDev/exp_pyTorch/results/v2_M2_result/Standard_myHealth_MobileNet_20210128-191958", 25, 10.83),
        # standard=("/Users/zber/ProgramDev/exp_pyTorch/results/myHealth_MB/Standard_myHealth_MobileNet_20200711-021420", 20, 10.83),

        # v2 emg
        # v1=("/Users/zber/ProgramDev/exp_pyTorch/results/EMG_MB/GC_EMG_MobileNet_best", 29, 14.17),
        # v2=("/Users/zber/ProgramDev/exp_pyTorch/results/v2_MBnet/GC_EMG_MobileNet_Time_Flops_best", 29, 14.17),
        # v2=("/Users/zber/ProgramDev/exp_pyTorch/results/v2_MBnet/GC_EMG_MobileNet_bestbest", 24, 14.17),
        # v2opt=("/Users/zber/ProgramDev/exp_pyTorch/results/v2_M2_result/Standard_EMG_MobileNet_20210127-215143", 25, 14.17),
        # standard=("/Users/zber/ProgramDev/exp_pyTorch/results/EMG_MB/Standard_EMG_MobileNet_20200711-012939", 22, 14.17),

        # emg_lenet_v1=("/Users/zber/ProgramDev/exp_pyTorch/results/v2_Lenet/GC_EMG_LeNet_v1_best", 28, 3.9504),
        # emg_lenet_v2=("/Users/zber/ProgramDev/exp_pyTorch/results/v2_Lenet/GC_EMG_LeNet_best", 18, 3.9504),

        # myHealth_lenet_v1 = ("/Users/zber/ProgramDev/exp_pyTorch/results/v2_Lenet/GC_myHealth_LeNet_best", 21, 8.13),
        # myHealth_lenet_v1 = (),

        # fin_lenet_v1 = ("/Users/zber/ProgramDev/exp_pyTorch/results/v2_Lenet/GC_FinDroid_LeNet_v1_best1", 11,3.04),
        # fin_lenet_v2= ("/Users/zber/ProgramDev/exp_pyTorch/results/v2_Lenet/GC_FinDroid_LeNet_best", 7, 3.04),

    )

    dic_save = {}

    # for key, length in m_list:
    #     time = mdata[key][0][:length]
    #     s_list = read_size(dic[key])[:length]
    #     macs, params = cal_flops_lenet(s_list)
    #     dic_save[key] = (time, macs, params)

    # for key, length in m_list:
    #     time = mdata[key][0][:length]
    #     s_list = read_size(dic[key])[:length]
    #     macs, params = cal_flops_lenet(s_list)
    #     dic_save[key] = (time, macs, params)

    # Horizon
    # for key in dic.keys():
    #     dir_path = os.path.dirname(dic[key])
    #     size_file = [f for f in glob.glob(dir_path + "/*structure.txt")][0]
    #     acc = read_acc(dic[key])
    #     s_list = read_size(size_file)
    #     # macs, params = cal_flops_mobile(s_list)
    #     macs, params = cal_flops_lenet(s_list)
    #     dic_save[key] = (macs, acc)

    # Horizon V2
    # for key in dic.keys():
    #     dir_path = os.path.dirname(dic[key])
    #     size_file = [f for f in glob.glob(dir_path + "/*structure.txt")][0]
    #     acc = read_acc(dic[key])
    #     s_list = read_size(size_file)
    #     # macs, params = cal_flops_mobile(s_list)
    #     macs, params = cal_flops_lenet(s_list)
    #     dic_save[key] = (macs, acc)

    # Flops to time V2
    # for key in dic.keys():
    #     # har :15
    #     # length, time_factor = 15, 2.87
    #     # mnist :
    #     # length, time_factor = 19, 3.6
    #     # mb_har
    #     # length, time_factor = 20, 19.02
    #     # dir_path = dic[key]
    #     dir_path, length, time_factor = dic[key]
    #     log_path = file = [f for f in glob.glob(dir_path + "/*log.txt")][0]
    #     # size_file = [f for f in glob.glob(dir_path + "/*structure.txt")][0]
    #     s_list = read_size(log_path)
    #     time = read_time(dir_path)
    #     time = np.asarray(time) * time_factor / 60
    #     time = time.tolist()[:length]
    #
    #     # macs, params = cal_flops_mobile(s_list)
    #     macs, params = cal_flops_lenet(s_list)
    #     # macs, params = cal_flops_har(s_list)
    #
    #     dic_save[key] = (time, macs[:length])

    # Flops to time MobileNet v2
    for key in dic.keys():

        dir_path, length, time_factor = dic[key]

        log_path = file = [f for f in glob.glob(dir_path + "/*log.txt")][0]
        size_file = [f for f in glob.glob(dir_path + "/*structure.txt")][0]
        if key != 'standard':
            s_list = read_size(size_file)
        else:
            s_list = [0] * length
        # print(s_list[length - 1])

        time = read_time(dir_path)
        time = np.asarray(time) * time_factor  #/ 60
        time = time.tolist()[:length]
        print(time[length - 1])

        # accuracy
        ac_list = read_acc(log_path)
        print(ac_list[length - 1])

        # macs, params = cal_flops_mobile_emg(s_list)
        macs, params = cal_flops_mobile_myhealth(s_list)
        # macs, params = cal_flops_mobile_har(s_list)
        # macs, params = cal_flops_lenet(s_list)
        # macs, params = cal_flops_har(s_list)
        # macs, params = cal_flops_lenet_myHealth(s_list)
        # macs, params = cal_flops_fin(s_list)

        macs = macs * length
        params = params * length
        print("macs is {}".format(macs[length - 1]))
        print("params is {}".format(params[length - 1]))

        dic_save[key] = (time, macs[:length])
        # dic_save[key] = (time, macs)

    # read acc
    # for key in dic.keys():
    #     # dir_path = dic[key]
    #     dir_path, length, time_factor = dic[key]
    #
    #     log_path = file = [f for f in glob.glob(dir_path + "/*log.txt")][0]
    #     ac_list = read_acc(log_path)
    #     print(ac_list)

    # target_path = "/Users/zber/Documents/FGdroid/exp_result/G_2_resource/mnist_flops.json"
    # target_path = "/Users/zber/Documents/FGdroid/exp_result/G_2_resource/MNIST/mnist_flops.json"
    # target_path = "/Users/zber/Documents/FGdroid/exp_result/G_4_Horizon/lenet_flops_acc.json"
    # target_path = "/Users/zber/Desktop/v2_result/2_TH_FLOPS_AC/v2_mnist_flops.json"
    # target_path = "/Users/zber/Desktop/v2_result/2_TH_FLOPS_AC/v2_har_flops.json"
    # target_path = "/Users/zber/Desktop/v2_result/3_Time_FLOPs/v2_lenet_har.json"
    # target_path = "/Users/zber/Desktop/v2_result/3_Time_FLOPs/v2_lenet_mnist.json"
    # target_path = "/Users/zber/Desktop/v2_result/3_Time_FLOPs/v2_mbnet_har.json"
    # target_path = "/Users/zber/Desktop/v2_result/3_Time_FLOPs/v2_mbnet_myhealth.json"
    # target_path = "/Users/zber/Desktop/v2_result/3_Time_FLOPs/v2_mbnet_emg_new.json"

    # target_path = "/Users/zber/Documents/FGdroid/exp_result/v2_result/3_Time_FLOPs/v2_lenet_har_revision.json"
    # target_path = "/Users/zber/Documents/FGdroid/exp_result/v2_result/3_Time_FLOPs/v2_lenet_mnist_revision.json"
    # target_path = "/Users/zber/Documents/FGdroid/exp_result/v2_result/3_Time_FLOPs/v2_mbnet_har_revision.json"
    # target_path = "/Users/zber/Documents/FGdroid/exp_result/v2_result/3_Time_FLOPs/v2_mbnet_myhealth_revision.json"
    # target_path = "/Users/zber/Documents/FGdroid/exp_result/v2_result/3_Time_FLOPs/v2_mbnet_emg_new_revision.json"

    # TMC M2 revision
    # target_path = "/Users/zber/Documents/FGdroid/exp_result/v2_result/3_Time_FLOPs/v2_lenet_har_revision_m2.json"
    # target_path = "/Users/zber/Documents/FGdroid/exp_result/v2_result/3_Time_FLOPs/v2_lenet_mnist_revision_m2.json"
    # target_path = "/Users/zber/Documents/FGdroid/exp_result/v2_result/3_Time_FLOPs/v2_mbnet_har_revision_m2.json"
    target_path = "/Users/zber/Documents/FGdroid/exp_result/v2_result/3_Time_FLOPs/v2_mbnet_myhealth_revision_m2.json"
    # target_path = "/Users/zber/Documents/FGdroid/exp_result/v2_result/3_Time_FLOPs/v2_mbnet_emg_new_revision_m2.json"

    with open(target_path, 'w') as f:
        json.dump(dic_save, f, indent=4)
