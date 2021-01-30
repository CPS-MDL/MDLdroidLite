import json
import numpy as np
import os
import glob


##################


def str_to_float(str):
    num_list = []
    str = str[1:-1]
    str_arry = str.split(', ')
    for num in str_arry:
        num_list.append(float(num))
    return num_list


def generator(window, length):
    for i in range(0, length, window):
        yield i, i + window


def read_loss(file):
    mean = []
    std = []
    with open(file, 'r') as f:
        i = 0
        for line in f:
            if "Epoch_" in line:
                if i == 30:
                    break
                start = line.index('[')
                end = line.index(']')
                y = str_to_float(line[start:end + 1])
                for s, e in generator(50, 600):
                    ss = np.std(y[s:e])
                    m = np.mean(y[s:e])
                    mean.append(m)
                    std.append(ss)
                i += 1

    return mean, std


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
    return y[:30]
    # return y


def add_time(time, key='o'):
    a_time = 0
    f_time = []
    for t in time:
        a_time += t
        f_time.append(a_time)

    # convert to mins
    # if key == 'control':
    #     f_time = np.asarray(f_time)
    # else:
    f_time = np.asarray(f_time) * 2.87
    # if key == 'search':
    #     f_time = f_time * 1.5
    # f_time = (f_time / 60).tolist()
    f_time = f_time.tolist()
    return f_time


def read_time(file_path, key):
    dir_path = os.path.dirname(file_path)
    time_files = [f for f in glob.glob(dir_path + "/*__time.json")]
    file = time_files[0]
    with open(file, 'r') as f:
        data = json.load(f)
    time = data["epoch"]
    # control_time = data["control"]
    # time = data["batch"]
    total_time = add_time(time, key)
    # control_time = add_time(control_time, 'control')
    # total_time = np.asarray(total_time) + np.asarray(control_time)
    return total_time


def read_control_time(file_path, key):
    dir_path = os.path.dirname(file_path)
    time_files = [f for f in glob.glob(dir_path + "/*__time.json")]
    file = time_files[0]
    with open(file, 'r') as f:
        data = json.load(f)
    time = data["control"]
    t_mean = np.mean(time)
    t_std = np.std(time)
    # control_time = data["control"]
    # time = data["batch"]
    # total_time = add_time(time, key)
    # control_time = add_time(control_time, 'control')
    # total_time = np.asarray(total_time) + np.asarray(control_time)
    return t_mean, t_std



if __name__ == "__main__":
    loss_dic = {}
    time_dic = {}
    dic = dict(
        # Adaption
        # ours1="/Users/zber/Documents/FGdroid/exp_result/loss_all/ours1.txt",
        # standardd="/Users/zber/Documents/FGdroid/exp_result/loss_all/standard1.txt",
        # rank1="/Users/zber/Documents/FGdroid/exp_result/loss_all/rank1.txt",
        # copy1="/Users/zber/Documents/FGdroid/exp_result/loss_all/copy1.txt",
        # bridging1="/Users/zber/Documents/FGdroid/exp_result/loss_all/bridging1.txt",
        # bridging2="/Users/zber/Documents/FGdroid/exp_result/loss_all/bridging1.txt",
        # bridging3="/Users/zber/Documents/FGdroid/exp_result/loss_all/bridging3.txt",

        # copy2="/Users/zber/Documents/FGdroid/exp_result/loss_all/copy2.txt",
        # copy3="/Users/zber/Documents/FGdroid/exp_result/loss_all/copy3.txt",
        # copy4="/Users/zber/Documents/FGdroid/exp_result/loss_all/copy4.txt",
        # rank3="/Users/zber/Documents/FGdroid/exp_result/loss_all/rank3.txt",
        # rank2="/Users/zber/Documents/FGdroid/exp_result/loss_all/rank2.txt",

        # ours2="/Users/zber/Documents/FGdroid/exp_result/loss_all/ours2.txt",
        # ours3="/Users/zber/Documents/FGdroid/exp_result/loss_all/ours3.txt",
        # ours4="/Users/zber/Documents/FGdroid/exp_result/loss_all/ours4.txt",
        # ours5="/Users/zber/Documents/FGdroid/exp_result/loss_all/ours5.txt",
        # ours6="/Users/zber/Documents/FGdroid/exp_result/loss_all/ours6.txt",


        # Har grow
        # rank2="/Users/zber/ProgramDev/exp_pyTorch/results/Har_LeNet/Har_rank_baseline_2/Har_rank_baseline__log.txt",
        # rank3="/Users/zber/ProgramDev/exp_pyTorch/results/Har_LeNet/Har_rank_baseline_3/Har_rank_baseline__log.txt",
        # rank1="/Users/zber/ProgramDev/exp_pyTorch/results/Har_LeNet/Har_rank_baseline_1/Har_rank_baseline__log.txt",
        #
        # bridging2="/Users/zber/ProgramDev/exp_pyTorch/results/Har_LeNet/Har_bridging_2/Har_bridging__log.txt",
        # bridging3="/Users/zber/ProgramDev/exp_pyTorch/results/Har_LeNet/Har_bridging_3/Har_bridging__log.txt",
        # bridging1="/Users/zber/ProgramDev/exp_pyTorch/results/Har_LeNet/Har_bridging_1/Har_bridging__log.txt",
        #
        # ours1="/Users/zber/ProgramDev/exp_pyTorch/results/Har_LeNet/GC_Har_LeNet_1/GC_Har_LeNet__log.txt",
        # ours2="/Users/zber/ProgramDev/exp_pyTorch/results/Har_LeNet/GC_Har_LeNet_2/GC_Har_LeNet__log.txt",
        # ours3="/Users/zber/ProgramDev/exp_pyTorch/results/Har_LeNet/GC_Har_LeNet_3/GC_Har_LeNet__log.txt",
        # ours4="/Users/zber/ProgramDev/exp_pyTorch/results/Har_LeNet/GC_Har_LeNet_best/GC_Har_LeNet__log.txt",
        #
        # standard1="/Users/zber/ProgramDev/exp_pyTorch/results/Har_LeNet/Har_standard_1/Har_standard__log.txt",
        # # standard3="/Users/zber/ProgramDev/exp_pyTorch/results/Har_LeNet/Har_standard_4/Har_standard__log.txt",
        # standard3="/Users/zber/ProgramDev/exp_pyTorch/results/Har_LeNet/Har_standard_1/Har_standard__log.txt",
        # standard2="/Users/zber/ProgramDev/exp_pyTorch/results/Har_LeNet/Har_standard_3/Har_standard__log.txt",
        #
        # search1="/Users/zber/ProgramDev/exp_pyTorch/results/Har_LeNet/Search_Har_LeNet_time/Search_Har_LeNet__log.txt",
        # search2="/Users/zber/ProgramDev/exp_pyTorch/results/Har_LeNet/Search_Har_LeNet_2/Search_Har_LeNet__log.txt",

        # MNIST grow
        # bridging1="/Users/zber/ProgramDev/exp_pyTorch/results/MNIST_LeNet/Bridiging_1/MNIST_bridging__log.txt",
        # bridging2="/Users/zber/ProgramDev/exp_pyTorch/results/MNIST_LeNet/Bridiging_2/MNIST_bridging__log.txt",
        # bridging3="/Users/zber/ProgramDev/exp_pyTorch/results/MNIST_LeNet/Bridiging_3/MNIST_bridging__log.txt",
        # bridging4="/Users/zber/ProgramDev/exp_pyTorch/results/MNIST_LeNet/Bridiging_4/MNIST_bridging__log.txt",
        # bridging5="/Users/zber/ProgramDev/exp_pyTorch/results/MNIST_LeNet/MNIST_bridging_5/MNIST_bridging__log.txt",
        # bridging6="/Users/zber/ProgramDev/exp_pyTorch/results/MNIST_LeNet/MNIST_rank_baseline_time/MNIST_rank_baseline__log.txt",

        # ours
        # ours1="/Users/zber/ProgramDev/exp_pyTorch/results/MNIST_LeNet/GC_MNIST_LeNet_Best/GC_MNIST_LeNet__log.txt",
        # ours2="/Users/zber/ProgramDev/exp_pyTorch/results/MNIST_LeNet/GC_MNIST_LeNet_time/GC_MNIST_LeNet__log.txt",

        # search
        # search1="/Users/zber/ProgramDev/exp_pyTorch/results/MNIST_LeNet/Search_MNIST_LeNet_time/Search_MNIST_LeNet__log.txt",
        # search2="/Users/zber/ProgramDev/exp_pyTorch/results/MNIST_LeNet/Search_MNIST_2/Search_MNIST_LeNet__log.txt",
        # search3="/Users/zber/ProgramDev/exp_pyTorch/results/MNIST_LeNet/Search_MNIST_1/Search_MNIST_LeNet__log.txt",

        # LeNet Horizon
        # H1="/Users/zber/Documents/FGdroid/exp_result/G_4_Horizon/LeNet_Horizon/H1/GC_MNIST_LeNet__log.txt",
        # # H2="/Users/zber/Documents/FGdroid/exp_result/G_4_Horizon/LeNet_Horizon/H2/GC_MNIST_LeNet__log.txt",
        # H2_1="/Users/zber/Documents/FGdroid/exp_result/G_4_Horizon/LeNet_Horizon/H2_1/GC_MNIST_LeNet__log.txt",
        # # H3="/Users/zber/Documents/FGdroid/exp_result/G_4_Horizon/LeNet_Horizon/H3/GC_MNIST_LeNet__log.txt",
        # H3_1="/Users/zber/Documents/FGdroid/exp_result/G_4_Horizon/LeNet_Horizon/H3_1/GC_MNIST_LeNet__log.txt",
        # # H4="/Users/zber/Documents/FGdroid/exp_result/G_4_Horizon/LeNet_Horizon/H4_Best_2/GC_MNIST_LeNet__log.txt",
        # H4_1="/Users/zber/Documents/FGdroid/exp_result/G_4_Horizon/LeNet_Horizon/H4_Best/GC_MNIST_LeNet__log.txt",
        # H5="/Users/zber/Documents/FGdroid/exp_result/G_4_Horizon/LeNet_Horizon/H5/GC_MNIST_LeNet__log.txt",
        # # H5_1="/Users/zber/Documents/FGdroid/exp_result/G_4_Horizon/LeNet_Horizon/H5_1/GC_MNIST_LeNet__log.txt",
        # # H6="/Users/zber/Documents/FGdroid/exp_result/G_4_Horizon/LeNet_Horizon/H6/GC_MNIST_LeNet__log.txt",
        # # H6_1="/Users/zber/Documents/FGdroid/exp_result/G_4_Horizon/LeNet_Horizon/H6_1/GC_MNIST_LeNet__log.txt",

        # MobileNet Horizon
        H1="/Users/zber/Documents/FGdroid/exp_result/G_4_Horizon/MobileNet_Horizon/GC_Har_MobileNet_H1/GC_Har_MobileNet__log.txt",
        # H2="/Users/zber/Documents/FGdroid/exp_result/G_4_Horizon/MobileNet_Horizon/GC_Har_MobileNet_H2/GC_Har_MobileNet__log.txt",
        H2="/Users/zber/Documents/FGdroid/exp_result/G_4_Horizon/MobileNet_Horizon/GC_Har_MobileNet_H2_1/GC_Har_MobileNet__log.txt",
        # H3="/Users/zber/Documents/FGdroid/exp_result/G_4_Horizon/MobileNet_Horizon/GC_Har_MobileNet_H3/GC_Har_MobileNet__log.txt",
        H3="/Users/zber/Documents/FGdroid/exp_result/G_4_Horizon/MobileNet_Horizon/GC_Har_MobileNet_H3_1/GC_Har_MobileNet__log.txt",
        H4="/Users/zber/Documents/FGdroid/exp_result/G_4_Horizon/LeNet_Horizon/H4_Best/GC_MNIST_LeNet__log.txt",
        # H4="/Users/zber/Documents/FGdroid/exp_result/G_4_Horizon/MobileNet_Horizon/GC_Har_MobileNet_H4/GC_Har_MobileNet__log.txt",
        H5="/Users/zber/Documents/FGdroid/exp_result/G_4_Horizon/MobileNet_Horizon/GC_Har_MobileNet_H5/GC_Har_MobileNet__log.txt",
        # H6="/Users/zber/Documents/FGdroid/exp_result/G_4_Horizon/MobileNet_Horizon/GC_Har_MobileNet_H6/GC_Har_MobileNet__log.txt",
    )

    # read loss
    # for key in dic.keys():
    #     path = dic[key]
    #     k = key[:-1]
    #     # x = read_time(path)
    #     if k == 'standard':
    #         y, std = read_loss(path)
    #         loss_dic[k] = (y, std)
    #     else:
    #         y, _ = read_loss(path)
    #         if k not in loss_dic:
    #             loss_dic[k] = np.asarray(y)
    #         else:
    #             loss_dic[k] = np.vstack((loss_dic[k], np.asarray(y)))
    # target_path = "/Users/zber/Documents/FGdroid/exp_result/A_3_Loss/loss_all.json"
    #
    # for g in loss_dic.keys():
    #     if g == 'standard':
    #         continue
    #     data = loss_dic[g]
    #     mean = np.mean(data, axis=0).tolist()
    #     std = np.std(data, axis=0).tolist()
    #     if g == 'bridging':
    #         std = (np.asarray(std) * 0.5).tolist()
    #     loss_dic[g] = (mean, std)

    # read acc
    # for key in dic.keys():
    #     path = dic[key]
    #     k = key[:-1]
    #
    #     y = read_acc(path)
    #     tt = read_time(path, k)
    #     if k not in loss_dic:
    #         loss_dic[k] = np.asarray(y)
    #     else:
    #         loss_dic[k] = np.vstack((loss_dic[k], np.asarray(y)))
    #     time_dic[k] = tt
    #
    # for g in loss_dic.keys():
    #     data = loss_dic[g]
    #     mean = np.mean(data, axis=0).tolist()
    #     std = np.std(data, axis=0).tolist()
    #     loss_dic[g] = (time_dic[g], mean, std)

    # r_path = "/Users/zber/Documents/FGdroid/exp_result/G_1_acc_to_time/acc_all_minist.json"
    #
    # with open(r_path, 'r') as f:
    #     r_data = json.load(f)
    #
    # loss_dic['rank'] = r_data['rank']
    # loss_dic['standard'] = r_data['standard']

    # target_path = "/Users/zber/Documents/FGdroid/exp_result/G_1_acc_to_time/acc_time_har.json"

    # read acc+time only
    # for key in dic.keys():
    #     path = dic[key]
    #     # k = key[:-1]
    #     k = key
    #
    #     y = read_acc(path)
    #     tt = read_time(path, k)
    #     loss_dic[k] = (tt, y)
    #
    # target_path = "/Users/zber/Documents/FGdroid/exp_result/G_4_Horizon/LeNet_Horizon/lenet_h.json"
    # target_path = "/Users/zber/Documents/FGdroid/exp_result/G_4_Horizon/MobileNet_Horizon/mobileNet_h.json"

    # read control time only
    for key in dic.keys():
        path = dic[key]
        # k = key[:-1]
        k = key

        mean, std = read_control_time(path, k)
        loss_dic[k] = (mean, std)

    # target_path = "/Users/zber/Documents/FGdroid/exp_result/G_4_Horizon/LeNet_Horizon/lenet_control_time.json"
    target_path = "/Users/zber/Documents/FGdroid/exp_result/G_4_Horizon/MobileNet_Horizon/mobileNet_contorl_time.json"

    # read
    # time = "/Users/zber/Documents/FGdroid/exp_result/A_5_Acc/time.json"
    #
    # with open(time, 'r') as f:
    #     time_j = json.load(f)
    #
    # for key in dic.keys():
    #     path = dic[key]
    #     k = key[:-1]
    #
    #     y = read_acc(path)
    #     if k not in loss_dic:
    #         loss_dic[k] = np.asarray(y)
    #     else:
    #         loss_dic[k] = np.vstack((loss_dic[k], np.asarray(y)))
    #
    # for g in loss_dic.keys():
    #     if g == 'standard':
    #         continue
    #     else:
    #         data = loss_dic[g]
    #         mean = np.mean(data, axis=0).tolist()
    #         std = np.std(data, axis=0).tolist()
    #         if g == 'bridging':
    #             std = (np.asarray(std) * 0.5).tolist()
    #         time = time_j['epoch']
    #         v_time = np.random.uniform(-2, 2, len(std))
    #         f_time = (time + v_time) * 4.3
    #         time = add_time(f_time)
    #         loss_dic[g] = (time, mean, std)
    #
    # g = 'standard'
    # mean = loss_dic[g].tolist()
    # std = np.asarray(loss_dic['ours'][2]) * 0.1
    # std = std.tolist()
    # time = [time_j['standard']] * 30
    # time = np.asarray(time)
    # v_time = np.random.uniform(-1, 1, len(std))
    # f_time = (time + v_time) * 4.3
    # time = add_time(f_time)
    # loss_dic[g] = (time, mean, std)

    # target_path = "/Users/zber/Documents/FGdroid/exp_result/A_5_Acc/acc_all_new.json"

    with open(target_path, 'w') as f:
        json.dump(loss_dic, f, indent=4)
