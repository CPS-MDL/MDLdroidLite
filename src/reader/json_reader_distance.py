import json
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from sklearn.metrics.pairwise import cosine_similarity


def str_to_float(str):
    num_list = []
    str = str[1:-1]
    str_arry = str.split(', ')
    for num in str_arry:
        num_list.append(float(num))
    return np.asarray(num_list)


def distance_x_y(file_key, dic, gi=1, is_out=False):
    y = []
    key = "out" if is_out else "in"

    # if file_key.startswith('standard'):
    #     observe_epoch = 10
    # else:
    #     observe_epoch = 1

    observe_epoch = 1

    num_batch = 10

    for e in range(dic["epoch_grow"][gi], dic["epoch_grow"][gi] + observe_epoch):
        for n in range(1, (600 // num_batch) + 1):
            batches = []
            for b in range(1, num_batch + 1):
                s = data[str(e)][str(b * n - 1)][dic["layer"]]["S"][key]
                f = str_to_float(s)
                start = dic["grow_base_size"][dic["layer"]] + (gi - 1) * dic["grow_size"][dic["layer"]]
                f_exist = f[:start]
                f_need = f[start:]
                f_exist_mean = np.mean(f_exist)
                f_new_mean = np.mean(f_need)
                batches.append(f_exist_mean - f_new_mean)
            y.append(np.mean(batches))
    length = len(y)
    x = np.arange(length)
    return x, y


def generator(window=50, length=600):
    for i in range(0, length, window):
        yield i, i + window


def window_mean(array, length=12):
    new_array = []
    for start, end in generator(window=length, length=len(array)):
        new_array.append(np.mean(array[start:end]))
    return new_array


def x_y_sparsity(dic):
    dic_layer = {
        '0': [],
        '1': [],
        '2': [],
        '3': [],
    }

    for e in range(dic["epoch_from"], dic["epoch_to"]):
        for n in range(num_batches):
            for layer in dic_layer.keys():
                s = data[str(e)][str(n)][layer]["Sparsity"]["layer"]
                f = float(s)
                dic_layer[layer].append(f)

    for key in dic_layer.keys():
        dic_layer[key] = window_mean(dic_layer[key])
    return dic_layer


def x_y_cosine(dic):
    dic_layer = {
        '0': [],
        '1': [],
        '2': [],
        '3': [],
    }

    for e in range(dic["epoch_from"], dic["epoch_to"]):
        for n in range(num_batches):
            for layer in dic_layer.keys():
                s = data[str(e)][str(n)][layer]["Cosine"]
                if s:
                    dic_layer[layer].append(np.mean(np.abs(s)))
                else:
                    dic_layer[layer].append(0)

    for key in dic_layer.keys():
        dic_layer[key] = window_mean(dic_layer[key])
    return dic_layer


def x_y_score(dic, key="S", is_window=True):
    if mode == 'MobileNet':
        dic_layer = {
            '0': [],
            '2': [],
            '4': [],
            '6': [],
        }

    else:
        dic_layer = {
            '0': [],
            '1': [],
            '2': [],
            '3': [],
        }

    for e in range(dic["epoch_from"], dic["epoch_to"]):
        for n in range(num_batches):
            for layer in dic_layer.keys():
                if key == 'S' or key == 'VG':
                    s = data[str(e)][str(n)][layer][key]["layer"]
                    if s != "[]":
                        f = float(s)
                else:
                    f = data[str(e)][str(n)][layer][key]
                dic_layer[layer].append(f)
    if is_window:
        for key in dic_layer.keys():
            dic_layer[key] = window_mean(dic_layer[key])
    return dic_layer


def x_y_score_loss(dic, key="S"):
    dic_layer = {
        '0': [],
        '1': [],
        '2': [],
        '3': [],
    }

    for e in range(dic["epoch_from"], dic["epoch_to"]):
        for n in range(num_batches):
            for layer in dic_layer.keys():
                if key == 'S' or key == 'VG':
                    s = data[str(e)][str(n)][layer][key]["layer"]
                    if s != "[]":
                        f = float(s)
                else:
                    f = data[str(e)][str(n)][layer][key]
                dic_layer[layer].append(f)

    for key in dic_layer.keys():
        dic_layer[key] = window_mean(dic_layer[key])
    return dic_layer


def x_y_std(dic, is_out=False):
    global num_growth

    if is_out:
        dic_layer = {
            '0': [],
            '1': [],
            '2': [],
        }
    else:
        dic_layer = {
            '1': [],
            '2': [],
            '3': [],
        }

    out_in = "out" if is_out else "in"

    for e in range(dic["epoch_from"], dic["epoch_to"]):
        for n in range(num_batches):
            for layer in dic_layer.keys():
                s = data[str(e)][str(n)][layer]["S"][out_in]
                f = str_to_float(s)
                std = np.std(f)
                dic_layer[layer].append(std)

    for key in dic_layer.keys():
        dic_layer[key] = window_mean(dic_layer[key])
    return dic_layer


def plot_save(x_y, dir, layer='0', is_out=False):
    label_list = ['Grow_1', 'Grow_2', 'Grow_3', 'Grow_4']
    colour_list = ['b', 'g', 'yellow', 'black']
    path = os.path.join(dir, 'Distance_L{}_{}.png'.format(layer, ('out' if is_out else 'in')))
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 8))
    for xy, colour, label, in zip(x_y, colour_list, label_list):
        ax.plot(xy[0], xy[1], colour, label=label)

    ax.legend()
    fig.savefig(path)
    plt.close(fig)


def plot_std(dic, dir, is_out=False):
    grow_epochs = [1, 3, 6, 9]
    colour_list = ['b', 'g', 'yellow', 'black']
    path = os.path.join(dir, 'STD_{}.png'.format(('out' if is_out else 'in')))
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 8))

    for key, colour in zip(dic.keys(), colour_list):
        x = np.arange(len(dic[key]))
        ax.plot(x, dic[key], colour, label='Layer_{}'.format(key))

    for i in grow_epochs:
        point_x = i * (num_batches /window_size) - 1
        plt.axvline(x=point_x, color='red', linestyle='--')

    ax.legend()
    fig.savefig(path)
    plt.close(fig)


def print_loss_batch(path, is_window=True):
    a_c = []
    files = [f for f in glob.glob(path + "/*_log.txt")]
    file = files[0]

    with open(file, "r") as f:
        for line in f:
            l = None
            mean_list = []
            if line.endswith(';\n'):
                start = line.index('[')
                end = line.index(']')
                l = str_to_float(line[start:end + 1]).tolist()
                if is_window:
                    for start, end in generator(window_size, num_batches ):
                        m = np.mean(l[start: end])
                        mean_list.append(m)
                    a_c = a_c + mean_list
                else:
                    a_c = a_c + l
    return a_c


def plot_sparsity(dic, dir):
    grow_epochs = [1, 3, 6, 9]
    colour_list = ['b', 'g', 'yellow', 'black']
    path = os.path.join(dir, 'Sparsity.png')
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 8))

    for key, colour in zip(dic.keys(), colour_list):
        x = np.arange(len(dic[key]))
        ax.plot(x, dic[key], colour, label='Layer_{}'.format(key))

    for i in grow_epochs:
        point_x = i * (num_batches / window_size) - 1
        plt.axvline(x=point_x, color='red', linestyle='--')

    ax.legend()
    fig.savefig(path)
    plt.close(fig)


def plot_mode(dic, dir, mode='Cosine'):
    grow_epochs = [1, 3, 6, 9]
    colour_list = ['b', 'g', 'yellow', 'black']

    path = os.path.join(dir, '{}.png'.format(mode))

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 8))

    for key, colour in zip(dic.keys(), colour_list):
        x = np.arange(len(dic[key]))
        ax.plot(x, dic[key], colour, label='Layer_{}'.format(key))

    for i in grow_epochs:
        point_x = i * (num_batches / window_size)
        plt.axvline(x=point_x, color='grey', linestyle='--', alpha=0.5)

    ax.legend()
    fig.savefig(path)
    plt.close(fig)


def plot_mode_loss(dic, loss, dir, mode='Cosine'):
    grow_epochs = [1, 3, 6, 9]
    colour_list = ['b', 'g', 'yellow', 'black']

    path = os.path.join(dir, '{}.png'.format(mode))

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 8))

    for key, colour in zip(dic.keys(), colour_list):
        x = np.arange(len(dic[key]))
        y = np.asarray(dic[key])
        ax.plot(x, y, colour, label='Layer_{}'.format(key))

    for i in grow_epochs:
        point_x = i
        plt.axvline(x=point_x, color='grey', linestyle='--', alpha=0.5)

    ax.legend()
    fig.savefig(path)
    plt.close(fig)


def plot_total(dic, dir, title='Grad'):
    grow_epochs = [1, 3, 6, 9]
    colour_list = ['b', 'g', 'yellow', 'black', 'm', 'khaki']
    for layer in dic.keys():

        path = os.path.join(dir, '{}_{}.png'.format(title, layer))

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 8))

        for key, colour in zip(dic[layer].keys(), colour_list):
            x = np.arange(len(dic[layer][key]))
            ax.plot(x, dic[layer][key], colour, label='{}'.format(key))
        if title == 'Grad':
            for i in grow_epochs:
                point_x = i * (num_batches / window_size) - 1
                plt.axvline(x=point_x, color='grey', linestyle='--', alpha=0.5)

        ax.legend()
        plt.title(title)
        fig.savefig(path)
        plt.close(fig)


def gen_x_y(dic, is_out=False, file_key=''):
    global num_growth
    x_y = []
    for gi in range(1, num_growth):
        x_y.append(distance_x_y(file_key, dic, gi=gi, is_out=is_out))
    return x_y


def gen_x_y_different(k, dic, key='s'):
    global dic_total
    dic_layer = {
        '0': [],
        '1': [],
        '2': [],
        '3': [],
    }

    for e in range(dic["epoch_from"], dic["epoch_to"]):
        for n in range(num_batches):
            for layer in dic_layer.keys():
                if key == 'S' or key == 'VG':
                    s = data[str(e)][str(n)][layer][key]["layer"]
                    f = float(s)
                else:
                    f = data[str(e)][str(n)][layer][key]
                dic_layer[layer].append(f)

    for layer in dic_layer.keys():
        dic_total[layer][k] = window_mean(dic_layer[layer])


def window_cosine(dic, loss):
    if mode == 'MobileNet':
        dic_layer = {
            '0': [],
            '2': [],
            '4': [],
            '6': [],
        }

    else:
        dic_layer = {
            '0': [],
            '1': [],
            '2': [],
            '3': [],
        }
    for key in dic.keys():
        for start, end in generator(window=window_size, length=len(dic[key])):
            # s = np.reshape(np.asarray(dic[key][start:end]), (-1, 1))
            s = np.asarray(dic[key][start:end])
            # l = np.reshape(np.asarray(loss[start:end]), (-1, 1))
            l = np.asarray(loss[start:end])
            # cs = cosine_similarity(s, l)
            cs = np.dot(s, l) / (np.linalg.norm(s) * np.linalg.norm(l))
            dic_layer[key].append(cs)
    for key in dic_layer.keys():
        dic_layer[key] = window_mean(dic_layer[key], length=12)
    return dic_layer


def window_loss(dic, loss):
    dic_layer = {
        '0': [],
        '1': [],
        '2': [],
        '3': [],
    }
    for key in dic.keys():
        for start, end in generator(window=window_size, length=len(dic[key])):
            # s = np.reshape(np.asarray(dic[key][start:end]), (-1, 1))
            s = np.asarray(dic[key][start:end])
            # l = np.reshape(np.asarray(loss[start:end]), (-1, 1))
            l = np.asarray(loss[start:end])
            # cs = cosine_similarity(s, l)
            cs = s * l
            dic_layer[key].append(cs)
    for key in dic_layer.keys():
        dic_layer[key] = window_mean(dic_layer[key], length=12)
    return dic_layer


# def window_mean(array):
#     mean_array = []
#     for start, end in generator(window=12, length=120):
#         mean_array.append(np.mean(array[start:end]))
#     return mean_array


if __name__ == '__main__':
    mode = 'MobileNet'
    window_size = 23
    num_batches = 230
    dic = dict(
        # rank_baseline="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_baseline_6/layer1_log.txt",
        # rank_baseline_gate="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_baseline_gate/layer1_log.txt",
        # rank_baseline_LR3="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_baseline_LR3_20200412-233035/layer2_log.txt",
        # rank_baseline_LR4="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_baseline_LR4_20200413-001813/layer2_log.txt",
        # copy_n="/Users/zber/ProgramDev/exp_pyTorch/results/grow_copy_n_6/layer1_log.txt",
        # copy_one="/Users/zber/ProgramDev/exp_pyTorch/results/grow_copy_one_6/layer1_log.txt",
        # rank_ours="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_ours_11/layer1_log.txt",
        # bridging="/Users/zber/ProgramDev/exp_pyTorch/results/grow_bridging_6/layer1_log.txt",
        # bridging2="/Users/zber/ProgramDev/exp_pyTorch/results/grow_bridging_7/layer1_log.txt",
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
        # rank_cumulative_gate="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_gate/layer1_log.txt",
        # rank_cumulative_gate_max_test="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_gate_max_test/layer2_log.txt",
        # standard="/Users/zber/ProgramDev/exp_pyTorch/results/grow_standard_3/grow_standard__json_in_out.json",
        # standard_s="/Users/zber/ProgramDev/exp_pyTorch/results/grow_standard_s/grow_standard__json_dic.json",
        # rank_cumulative_cL_fH="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_convL_fcH/layer1_log.txt",
        # rank_cumulative_cL_fH2="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_convL_fcH2/layer1_log.txt",
        # rank_cumulative_H_normal="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_rank_20200415-161606/layer2_log.txt",
        # rank_cumulative_con1_fc2="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_con1_fc2_f/layer1_log.txt",
        # rank_cumulative_con2_fc1="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_con2_fc1_f/layer1_log.txt",
        # rank_cumulative_conR_fcR="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_conR_fcR/layer1_log.txt",
        # rank_cumulative_con2_fc2="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_con2_fc2/layer1_log.txt",
        # rank_cumulative_con2_fc1="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_con2_fc1/layer1_log.txt",
        # rank_cumulative_con1_fc2="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_con1_fc2/layer1_log.txt",
        # standard_seed="/Users/zber/ProgramDev/exp_pyTorch/results/grow_standard_seed/layer1_log.txt",
        # standard_seed_BN="/Users/zber/ProgramDev/exp_pyTorch/results/grow_standard_seed_BN/layer1_log.txt",
        # rank_cumulative_con2_fc2_in2="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_con2_fc2_in2/layer1_log.txt",
        # rank_cumulative_conR_fcR_in2="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_conR_fcR_in2/layer1_log.txt",
        # rank_cumulative_conN_fcN_in2="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_conN_fcN_in2/layer1_log.txt",
        #
        # rank_cumulative_conR_fcR_in2_out2="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_conR_fcR_in2_out2/layer1_log.txt",
        # rank_cumulative_conN_fcN_in2_out2="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_conN_fcN_in2_out2/layer1_log.txt",
        # rank_cumulative_con2_fc2_in2_out2="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_con2_fc2_in2_out2/layer1_log.txt",

        # rank_cumulative_conR_fcR_inR_ogR="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_conR_fcR_inR_ogR/layer1_log.txt",
        # rank_cumulative_conR_fcR_gateAuto_test="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_conR_fcR_gateAuto/layer1_log.txt",
        # rank_cumulative_con2R_fc2R_gateAuto_test = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_con2R_fc2R_gateAuto/layer1_log.txt",
        # rank_cumulative_con2R_fc2R_gateAuto300_test="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_con2R_fc2R_gateAuto300/layer1_log.txt",
        # rank_cumulative_conR_fcR="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_conR_fcR/layer1_log.txt",

        # rank_cumulative_con2R_fc2R_in2R="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_con2R_fc2R_in2R/layer1_log.txt",
        # rank_cumulative_con2R_fc2R_in2R_gateAutoMax_LR600_test="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_con2R_fc2R_in2R_gateAutoMax_LR600/layer1_log.txt",
        # rank_cumulative_con2R_fc2R_in2R_gateAutoMax_LR300_test="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_con2R_fc2R_in2R_gateAutoMax_LR300/layer1_log.txt",
        # rank_cumulative_con2R_fc2R_in2R_gateAutoMax300_test="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_con2R_fc2R_in2R_gateAutoMax300/layer1_log.txt",
        # rank_cumulative_con2R_fc2R_in2R_LR_600="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_con2R_fc2R_in2R_LR_600/layer1_log.txt",
        # rank_cumulative_con2R_fc2R_gate_LR_300_test="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_con2R_fc2R_gate&LR_300/layer1_log.txt",

        # rank_cumulative_gate_600_test="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative+gate_600/layer1_log.txt",
        # rank_cumulative_gate_300_test="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative+gate_300/layer1_log.txt",
        # rank_cumulative_2R_gate_600_test="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_2R+gate_600/layer1_log.txt",
        # rank_cumulative_2R_gate_300_test="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_2R+gate_300/layer1_log.txt",

        # rank_cumulative_conR_fc1="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_conR_fc1/layer1_log.txt",
        # rank_cumulative_con1_fcR="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_con1_fcR/layer1_log.txt",
        # rank_cumulative_conN_fcN="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_conN_fcN/layer1_log.txt",
        # rank_cumulative_conN_fc1="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_conN_fc1/layer1_log.txt",
        # rank_cumulative_con1_fcN="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_con1_fcN/layer1_log.txt",
        # rank_cumulative_conR_fcR_15="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_conR_fcR_15/layer1_log.txt",
        # rank_cumulative_con2_fc2_15="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_con2_fc2_15/layer1_log.txt",
        # rank_cumulative_og2_in2="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_og2_in2/layer1_log.txt",
        # rank_cumulative_low_normal="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_low_20200415-161006/layer1_log.txt",
        # rank_cumulative_conv_gate_fc_nochange_test="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_conv_gate_fc_nochange_test/layer1_log.txt",
        # rank_cumulative_conv_gateauto_fc_nochange_test="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_conv_gateauto_fc_nochange_test/layer1_log.txt",
        # rank_cumulative_conv_gate_fc_n_test="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_conv_gate_fc_n_test/layer1_log.txt",
        # rank_cumulative_conv_gate_fc_gate_n_test="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_conv_gate_fc_gate_n_test/layer1_log.txt",
        # rank_cumulative_conv_gate_fc_gate_test="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_conv_gate_fc_gate_test/layer1_log.txt",
        # rank_cumulative_conv_gate_fc_gate_ratio_test="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_conv_gate_fc_gate_ratio_test/layer1_log.txt",
        # rank_baseline="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_baseline_20200430-190631/layer1_log.txt",
        # random="/Users/zber/ProgramDev/exp_pyTorch/results/grow_random_20200430-194619/layer1_log.txt",
        # copy_n="/Users/zber/ProgramDev/exp_pyTorch/results/grow_copy_n_20200430-191941/layer1_log.txt",
        # bridging="/Users/zber/ProgramDev/exp_pyTorch/results/grow_bridging_20200430-193258/layer1_log.txt",
        # rank_cumulative="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_20200430-213135/layer1_log.txt",
        # rank_cumulative_cosine="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_cosine_20200430-214432/layer1_log.txt",

        # standard_s_20_500_50="/Users/zber/ProgramDev/exp_pyTorch/results/grow_standard_20_50_500/layer1_log.txt",
        # standard_10_17_50="/Users/zber/ProgramDev/exp_pyTorch/results/standard_10_17_50/layer1_log.txt",
        # standard_8_14_40="/Users/zber/ProgramDev/exp_pyTorch/results/standard_8_14_40/layer1_log.txt",
        # standard_6_11_30="/Users/zber/ProgramDev/exp_pyTorch/results/standard_6_11_30/layer1_log.txt",

        # standard_s_20_500_50="/Users/zber/ProgramDev/exp_pyTorch/results/grow_standard_20_50_500/layer1_log.txt",
        # standard_10_17_50="/Users/zber/ProgramDev/exp_pyTorch/results/standard_10_17_50/layer1_log.txt",
        # standard_8_14_40="/Users/zber/ProgramDev/exp_pyTorch/results/standard_8_14_40/layer1_log.txt",
        # standard_6_11_30="/Users/zber/ProgramDev/exp_pyTorch/results/standard_6_11_30/layer1_log.txt",
        # standard_4_8_20="/Users/zber/ProgramDev/exp_pyTorch/results/standard_4_8_20/layer1_log.txt",
        # # standard_2_5_10="/Users/zber/ProgramDev/exp_pyTorch/results/standard_2_5_10/layer1_log.txt",
        # rank_baseline="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_baseline_20200430-190631/layer1_log.txt",
        # random="/Users/zber/ProgramDev/exp_pyTorch/results/grow_random_20200430-194619/layer1_log.txt",
        # copy_n="/Users/zber/ProgramDev/exp_pyTorch/results/grow_copy_n_20200430-191941/layer1_log.txt",
        # bridging="/Users/zber/ProgramDev/exp_pyTorch/results/grow_bridging_20200430-193258/layer1_log.txt",
        # rank_cumulative="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_20200430-213135/layer1_log.txt",
        # rank_cumulative_cosine="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_cosine_20200430-214432/layer1_log.txt",
        #
        # # 5-1
        # rank_consine="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_20200501-195838/layer1_log.txt",
        # rank_cumulative2R_output_no_2R="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative2R_output_no2R/layer1_log.txt",
        # rank_cumulative_gate_autoadd_test="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_gate_autoadd/layer1_log.txt",
        # rank_2R="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_2R/layer1_log.txt",
        #
        # # 5-3
        # rank_one_all2="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_one_all2/layer1_log.txt",
        # standard_s_random_overP="/Users/zber/ProgramDev/exp_pyTorch/results/grow_random_overP/layer1_log.txt",
        # standard_s_rank_baseline_overP="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_baseline_overP/layer1_log.txt",
        # standard_s_copy_n="/Users/zber/ProgramDev/exp_pyTorch/results/grow_copy_n_20200503-183009/layer1_log.txt",
        # standard_s_bridging="/Users/zber/ProgramDev/exp_pyTorch/results/grow_bridging_20200503-191340/layer1_log.txt",
        # rank_ours_batch20="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_ours_batch20_no2R/layer1_log.txt",
        # rank_cumulative_batch20="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_batch20_no2R/layer1_log.txt",

        # 5-5
        # random0="/Users/zber/ProgramDev/exp_pyTorch/results/grow_random_0.1/layer1_log.txt",
        # random1="/Users/zber/ProgramDev/exp_pyTorch/results/grow_random_-1to1/layer1_log.txt",
        # rank_baseline="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_baseline_20200506-181521/layer1_log.txt",
        # standard_4_10_20="/Users/zber/ProgramDev/exp_pyTorch/results/standard_4_10_20_20200505-231505/layer1_log.txt",
        # init_random="/Users/zber/ProgramDev/exp_pyTorch/results/grow_random_20200505-230457/layer1_log.txt",
        # rank_cosine="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_E3/layer1_log.txt",
        # standard_2_5_10="/Users/zber/ProgramDev/exp_pyTorch/results/standard_2_5_10_20200505-211708/layer1_log.txt",
        # standard_4_8_20="/Users/zber/ProgramDev/exp_pyTorch/results/standard_4_8_20_20200505-212418/layer1_log.txt",
        # standard_6_11_30="/Users/zber/ProgramDev/exp_pyTorch/results/standard_6_11_30_20200505-213355/layer1_log.txt",
        # standard_8_14_40="/Users/zber/ProgramDev/exp_pyTorch/results/standard_8_14_40_20200505-214554/layer1_log.txt",
        # standard_10_17_50="/Users/zber/ProgramDev/exp_pyTorch/results/standard_10_17_50_20200505-220015/layer1_log.txt",
        # standard_2_5_10="/Users/zber/ProgramDev/exp_pyTorch/results/standard_2_5_10_20200510-005727/layer1_log.txt",

        # 5-9
        # random="/Users/zber/ProgramDev/exp_pyTorch/results/grow_random_20200508-225047/layer1_log.txt",
        # copy_n="/Users/zber/ProgramDev/exp_pyTorch/results/grow_copy_n_20200508-225737/layer1_log.txt",
        # rank_baseline="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_baseline_20200508-224355/layer1_log.txt",

        # E20
        # low_cosine_e20="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_20e_20200511-041701/layer1_log.txt",
        # rank_low_e20="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_ours_20e_20200511-043252/layer1_log.txt",
        # random_e20="/Users/zber/ProgramDev/exp_pyTorch/results/grow_random_20e_20200511-040126/layer1_log.txt",
        # copy_n_e20="/Users/zber/ProgramDev/exp_pyTorch/results/grow_copy_n_e20_20200512-004939/layer1_log.txt",
        # rank_baseline_e20="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_baseline_e20_20200512-010604/layer1_log.txt",
        # bridging_e20="/Users/zber/ProgramDev/exp_pyTorch/results/grow_bridging_e20_20200512-003340/layer1_log.txt",

        # 5-10
        # standard_2_5_10="/Users/zber/ProgramDev/exp_pyTorch/results/standard_2_5_10_20200510-010756",
        # standard_4_8_20="/Users/zber/ProgramDev/exp_pyTorch/results/standard_4_8_20_20200510-011213",
        # standard_6_11_30="/Users/zber/ProgramDev/exp_pyTorch/results/standard_6_11_30_20200510-011700",
        # standard_8_14_40="/Users/zber/ProgramDev/exp_pyTorch/results/standard_8_14_40_20200510-012239",
        # standard_10_17_50="/Users/zber/ProgramDev/exp_pyTorch/results/standard_10_17_50_20200510-012903",
        # standard_20_50_500="/Users/zber/ProgramDev/exp_pyTorch/results/standard_20_50_500_20200512-191304"
        # rank_ours_vg_low="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_ours_vg_low",

        # 5-10
        # rank_ours_low="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_ours_low",
        # rank_ours_vg_low="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_ours_vg_low",
        # standard_10_17_50="/Users/zber/ProgramDev/exp_pyTorch/results/standard_10_17_50_20200510-201607",

        # 5-13
        # standard_2_5_10="/Users/zber/ProgramDev/exp_pyTorch/results/standard_2_5_10_20200513-214705",
        # standard_4_8_20="/Users/zber/ProgramDev/exp_pyTorch/results/standard_4_8_20_20200513-215217",
        # standard_6_11_30="/Users/zber/ProgramDev/exp_pyTorch/results/standard_6_11_30_20200513-215804",
        # standard_8_14_40="/Users/zber/ProgramDev/exp_pyTorch/results/standard_8_14_40_20200513-220428",
        # standard_10_17_50="/Users/zber/ProgramDev/exp_pyTorch/results/standard_10_17_50_20200513-221254",

        # cosine_RI_momentum="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_20200522-233322",
        # standard_10_17_50="/Users/zber/ProgramDev/exp_pyTorch/results/standard_10_17_50_20200514-194428",

        # cosine_L_1_index="/Users/zber/ProgramDev/exp_pyTorch/results/cosine_L_1_index_20200530-192920",
        # cosine_L_L_index="/Users/zber/ProgramDev/exp_pyTorch/results/cosine_L_L_index_20200530-194103",

        grow_mb = "/Users/zber/ProgramDev/exp_pyTorch/results/G_Har_MobileNet20200620-174316",
        grow_mb_bn = "/Users/zber/ProgramDev/exp_pyTorch/results/G_Har_MobileNetBN_20200620-192435",
        standard_mb= "/Users/zber/ProgramDev/exp_pyTorch/results/Standard_Har_MobileNet_20200620-174936",
        standard_mb_bn = "/Users/zber/ProgramDev/exp_pyTorch/results/Standard_Har_MobileNetBN_20200620-181236",

    )
    # plot_mode_list = ['distance', 'sparsity', 'std', 'cosine']
    # plot_mode_list = ['Compare_VG' ,'s_score']
    # plot_mode_list = ['cosine']
    # plot_mode_list = ['Compare']
    # plot_mode_list = ['V', 'G', 'M', 'Grad']
    # plot_mode_list = ['std']
    # plot_mode_list = ['Compare_Loss_VG_cosine']
    plot_mode_list = ['Compare_Loss_S_cosine']

    if mode == 'MobileNet':
        dic_total = {
            '0': {},
            '2': {},
            '4': {},
            '6': {},
        }

    else:
        dic_total = {
            '0': {},
            '1': {},
            '2': {},
            '3': {},
        }

    num_growth = 5

    dic_layer_out = {
        "epoch_from": 1,
        "layer": "1",
        "epoch_to": 11,
        "epoch_grow": [1, 2, 4, 7, 10],  # [1, 1, 1, 1, 1],
        "num_batch": 100,
        "grow_base_size": {"0": 2, "1": 5, "2": 10},
        "grow_size": {"0": 2, "1": 3, "2": 10},
        "exsit_index": {"0": (0, 2), "1": (0, 5), "2": (0, 10)},
        "control_batches": 600
    }

    dic_layer_in = {
        "epoch_from": 1,
        "layer": "1",
        "epoch_to": 11,
        "epoch_grow": [1, 2, 4, 7, 10],  # [1, 2, 4, 7, 10],
        "num_batch": 100,
        "grow_base_size": {"1": 2, "2": 80, "3": 10},
        "grow_size": {"1": 2, "2": 48, "3": 10},
        "exsit_index": {"1": (0, 2), "2": (0, 80), "3": (0, 10)},
        "control_batches": 600
    }

    for key in dic.keys():
        print(key, ':')

        num = key[-2:]

        if num.isdigit():
            num_int = int(num)
            num_growth = num_int / 10
            num_growth = int(num_growth)
        else:
            num_growth = 5

        if key.startswith('standard'):
            dic_layer_in["epoch_grow"] = [1, 1, 1, 1, 1]
            dic_layer_out["epoch_grow"] = [1, 1, 1, 1, 1]
            if key == 'standard_s':
                dic_layer_in["epoch_grow"] = [1, 2, 4, 7, 10]
                dic_layer_out["epoch_grow"] = [1, 2, 4, 7, 10]
                dic_layer_out["grow_size"] = {"0": 4, "1": 10, "2": 100}
                dic_layer_out["grow_base_size"] = {"0": 4, "1": 10, "2": 100}
                dic_layer_in["grow_size"] = {"1": 4, "2": 160, "3": 100}
                dic_layer_in["grow_base_size"] = {"1": 4, "2": 160, "3": 100}
        else:
            dic_layer_in["epoch_grow"] = [1, 2, 4, 7, 10]
            dic_layer_out["epoch_grow"] = [1, 2, 4, 7, 10]

        path_to_file = dic[key]

        # path = os.path.dirname(path_to_file)
        path = path_to_file

        files = [f for f in glob.glob(path + "/*copy.json")]

        file = files[0]

        with open(file, 'r') as f:
            data = json.load(f)

        for out in [True, False]:
            is_out = out
            path = os.path.dirname(file)

            if 'distance' in plot_mode_list:
                if is_out:
                    for layer in ["0", "1", "2"]:  #
                        dic_layer_out["layer"] = layer
                        x_y = gen_x_y(dic_layer_out, is_out=is_out, file_key=key)
                        plot_save(x_y, path, layer=layer, is_out=is_out)
                else:
                    for layer in ["1", "2", "3"]:  #
                        dic_layer_in["layer"] = layer
                        x_y = gen_x_y(dic_layer_in, is_out=is_out, file_key=key)
                        plot_save(x_y, path, layer=layer, is_out=is_out)

            if 'std' in plot_mode_list:
                if is_out:
                    dic_std = x_y_std(dic_layer_out, is_out=is_out)
                    plot_std(dic_std, path, is_out=is_out)
                else:
                    dic_std = x_y_std(dic_layer_out, is_out=is_out)
                    plot_std(dic_std, path, is_out=is_out)

            if 'sparsity' in plot_mode_list:
                dic_sparsity = x_y_sparsity(dic_layer_out)
                plot_sparsity(dic_sparsity, path)

            if 'cosine' in plot_mode_list:
                dic_cosine = x_y_cosine(dic_layer_out)
                plot_mode(dic_cosine, path)

        if 's_score' in plot_mode_list:
            dic_score = x_y_score(dic_layer_out)
            plot_mode(dic_score, path, mode='S_score')

        if 'VG' in plot_mode_list:
            dic_score = x_y_score(dic_layer_out, key='VG')
            plot_mode(dic_score, path, mode='VG')

        if 'V' in plot_mode_list:
            dic_score = x_y_score(dic_layer_out, key='V')
            plot_mode(dic_score, path, mode='V')

        if 'M' in plot_mode_list:
            dic_score = x_y_score(dic_layer_out, key='M')
            plot_mode(dic_score, path, mode='M')

        if 'G' in plot_mode_list:
            dic_score = x_y_score(dic_layer_out, key='G')
            plot_mode(dic_score, path, mode='G')

        if 'Grad' in plot_mode_list:
            dic_score = x_y_score(dic_layer_out, key='Grad')
            plot_mode(dic_score, path, mode='Grad')

        if 'LS' in plot_mode_list:
            dic_score = x_y_score(dic_layer_out, key='S')
            loss = print_loss_batch(path)

            plot_mode_loss(dic_score, loss, path, mode='Loss_S')

        if 'LS_cosine' in plot_mode_list:
            dic_score = x_y_score(dic_layer_out, key='S', is_window=False)
            loss = print_loss_batch(path, is_window=False)
            dic_cosine = window_cosine(dic_score, loss)
            plot_mode_loss(dic_cosine, loss, path, mode='Loss_S_cosine')

        if 'LVG' in plot_mode_list:
            dic_score = x_y_score(dic_layer_out, key='VG')
            loss = print_loss_batch(path)
            plot_mode_loss(dic_score, loss, path, mode='Loss_SVG')

        if 'Compare_Loss_S_cosine' in plot_mode_list:
            dic_score = x_y_score(dic_layer_out, key='S', is_window=False)
            loss = print_loss_batch(path, is_window=False)
            dic_cosine = window_cosine(dic_score, loss)
            # gen_x_y_different(k=key, dic=dic_layer_out, key='S')
            for layer in dic_cosine.keys():
                dic_total[layer][key] = dic_cosine[layer]

        if 'Compare_Loss_VG_cosine' in plot_mode_list:
            dic_score = x_y_score(dic_layer_out, key='VG', is_window=False)
            loss = print_loss_batch(path, is_window=False)
            dic_cosine = window_cosine(dic_score, loss)
            # gen_x_y_different(k=key, dic=dic_layer_out, key='S')
            for layer in dic_cosine.keys():
                dic_total[layer][key] = dic_cosine[layer]

        if 'Compare_Loss_S' in plot_mode_list:
            dic_score = x_y_score(dic_layer_out, key='S', is_window=False)
            loss = print_loss_batch(path, is_window=False)
            dic_cosine = window_loss(dic_score, loss)
            # gen_x_y_different(k=key, dic=dic_layer_out, key='S')
            for layer in dic_cosine.keys():
                dic_total[layer][key] = dic_cosine[layer]

        if 'Compare_Loss_VG' in plot_mode_list:
            dic_score = x_y_score(dic_layer_out, key='VG', is_window=False)
            loss = print_loss_batch(path, is_window=False)
            dic_cosine = window_loss(dic_score, loss)
            # gen_x_y_different(k=key, dic=dic_layer_out, key='S')
            for layer in dic_cosine.keys():
                dic_total[layer][key] = dic_cosine[layer]

        if 'Compare_VG' in plot_mode_list:
            gen_x_y_different(k=key, dic=dic_layer_out, key='VG')

        if 'Compare_S' in plot_mode_list:
            gen_x_y_different(k=key, dic=dic_layer_out, key='S_score')

    if 'Compare_VG' in plot_mode_list:
        plot_total(dic=dic_total, dir="/Users/zber/ProgramDev/exp_pyTorch/results/5-10", title='Compare_VG')

    if 'Compare_S' in plot_mode_list:
        plot_total(dic=dic_total, dir="/Users/zber/ProgramDev/exp_pyTorch/results/5-10", title='Compare_S')

    if 'Compare_Loss_S_cosine' in plot_mode_list:
        plot_total(dic=dic_total, dir="/Users/zber/ProgramDev/exp_pyTorch/results/5-10", title='Loss_S_cosine')

    if 'Compare_Loss_VG_cosine' in plot_mode_list:
        plot_total(dic=dic_total, dir="/Users/zber/ProgramDev/exp_pyTorch/results/5-10", title='Loss_VG_cosine')

    if 'Compare_Loss_S' in plot_mode_list:
        plot_total(dic=dic_total, dir="/Users/zber/ProgramDev/exp_pyTorch/results/5-10", title='Compare_Loss_S')

    if 'Compare_Loss_VG' in plot_mode_list:
        plot_total(dic=dic_total, dir="/Users/zber/ProgramDev/exp_pyTorch/results/5-10", title='Compare_Loss_VG')
