import json
import numpy as np
import matplotlib.pyplot as plt
import os

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

base_path = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_baseline_Conv_{i}/grow_rank_baseline_Conv__json_in_out.json"

path_list = []
for i in range(1, 6):
    p = base_path.format(i=i)
    path_list.append(p)


def str_to_float(str):
    num_list = []
    str = str[1:-1]
    str_arry = str.split(', ')
    for num in str_arry:
        num_list.append(float(num))
    return np.asarray(num_list)


# with open(j3, 'r') as f:
#     data1 = json.load(f)

# grow 0

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


def plot_save(x_y, dir, layer='0', is_out=False):
    label_list = ['Exsiting', 'Grow_1', 'Grow_2', 'Grow_3', 'Grow_4']
    colour_list = ['r--', 'b', 'g', 'yellow', 'black']
    # ax.plot(x, mean, label='C8_mean')
    # ax.plot(x, mean1, label='C12_mean')
    # ax.plot(x, mean2, label='C16_mean')
    # ax.plot(x, mean3, label='C20_mean')
    # ax.plot(x, mean4, label='C24_mean')
    # ax.legend()
    path = os.path.join(dir, 'L{}_{}.png'.format(layer, ('out' if is_out else 'in')))
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 8))
    for xy, colour, label, in zip(x_y, colour_list, label_list):
        ax.plot(xy[0], xy[1], colour, label=label)
    ax.legend()
    fig.savefig(path)
    plt.close(fig)


def gen_x_y(is_out=True):
    x_y = []
    if is_out:
        for gi in range(5):
            if gi == 0:
                x_y.append(grow_x_y_exist(dic_layer_out))
            else:
                x_y.append(grow_x_y(dic_layer_out, gi=gi))
    else:
        for gi in range(5):
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
        "epoch_grow": [1, 2, 4, 7, 10],  # [1, 1, 1, 1, 1]
        "num_batch": 50,
        "grow_base_size": {"0": 2, "1": 5, "2": 10},
        "grow_size": {"0": 2, "1": 3, "2": 10},
        "exsit_index": {"0": (0, 2), "1": (0, 5), "2": (0, 10)},
    }

    dic_layer_in = {
        "epoch_from": 1,
        "layer": "1",
        "epoch_to": 11,
        "epoch_grow": [1, 2, 4, 7, 10],  # [1, 1, 1, 1, 1],
        "num_batch": 50,
        "grow_base_size": {"1": 2, "2": 80, "3": 10},
        "grow_size": {"1": 2, "2": 48, "3": 10},
        "exsit_index": {"1": (0, 2), "2": (0, 80), "3": (0, 10)},
    }

    for file in path_list:  # [good1,good2,bad1,bad2,nono_bad1, nono_bad2]: #, good1, good2
        with open(file, 'r') as f:
            data = json.load(f)

        for out in [True, False]:
            is_out = out
            path = os.path.dirname(file)

            if is_out:
                for layer in ["0", "1"]:  # , "2"
                    dic_layer_out["layer"] = layer
                    x_y = gen_x_y(is_out=is_out)
                    plot_save(x_y, path, layer=layer, is_out=is_out)
            else:
                for layer in ["1", "2"]:  # , "3"
                    dic_layer_in["layer"] = layer
                    x_y = gen_x_y(is_out=is_out)
                    plot_save(x_y, path, layer=layer, is_out=is_out)

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
