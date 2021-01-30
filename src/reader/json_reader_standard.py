import json
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# min, max extract

dic = dict(
    standard="/Users/zber/ProgramDev/exp_pyTorch/results/grow_standard_3/grow_standard__json_in_out.json",
    standard_s="/Users/zber/ProgramDev/exp_pyTorch/results/grow_standard_s/grow_standard__json_in_out.json",
)


def str_to_float(str):
    num_list = []
    str = str[1:-1]
    str_arry = str.split(', ')
    for num in str_arry:
        num_list.append(float(num))
    return np.asarray(num_list)


def gen_x_y(dic, is_out=True, mode="S"):
    y_min = []
    y_max = []
    key = "out" if is_out else "in"
    last_epoch = "10"
    last_batch = "598"
    last_s = data[last_epoch][last_batch][dic["layer"]][mode][key]
    last_f = np.asarray(str_to_float(last_s))
    min_s = np.argmin(last_f)
    max_s = np.argmax(last_f)

    for e in range(dic["epoch_from"], dic["epoch_to"]):
        for n in range(1, (600 // dic["num_batch"]) + 1):
            batches_min = []
            batches_max = []
            for b in range(1, dic["num_batch"] + 1):
                s = data[str(e)][str(b * n - 1)][dic["layer"]][mode][key]
                f = str_to_float(s)
                min_score = f[min_s]
                max_score = f[max_s]
                batches_min.append(min_score)
                batches_max.append(max_score)
            y_min.append(np.mean(batches_min))
            y_max.append(np.mean(batches_max))
    length = len(y_min)
    x = np.arange(length)
    return x, y_min, y_max


def plot_save(x_y, dir, layer='0', is_out=False, mode="S"):
    label_list = ['min_channel', 'max_channel']
    colour_list = ['yellow', 'black']
    path = os.path.join(dir, 'Min_Max_L{}_{}_M{}.png'.format(layer, ('out' if is_out else 'in'), mode))
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 8))
    x, y_min, y_max = x_y
    ax.plot(x, y_min, colour_list[0], label=label_list[0])
    ax.plot(x, y_max, colour_list[1], label=label_list[1])
    ax.legend()
    fig.savefig(path)
    plt.close(fig)


if __name__ == '__main__':

    num_growth = 5
    mode = "S"

    dic_layer_out = {
        "epoch_from": 1,
        "layer": "1",
        "epoch_to": 11,
        "epoch_grow": [1, 2, 4, 7, 10],  # [1, 1, 1, 1, 1],
        "num_batch": 20,
        "grow_base_size": {"0": 2, "1": 5, "2": 10},
        "grow_size": {"0": 2, "1": 3, "2": 10},
        "exsit_index": {"0": (0, 2), "1": (0, 5), "2": (0, 10)},
    }

    dic_layer_in = {
        "epoch_from": 1,
        "layer": "1",
        "epoch_to": 11,
        "epoch_grow": [1, 2, 4, 7, 10],  # [1, 2, 4, 7, 10],
        "num_batch": 20,
        "grow_base_size": {"1": 2, "2": 80, "3": 10},
        "grow_size": {"1": 2, "2": 48, "3": 10},
        "exsit_index": {"1": (0, 2), "2": (0, 80), "3": (0, 10)},
    }

    for key in dic.keys():
        print(key, ':')

        if key.startswith('standard'):
            dic_layer_in["epoch_grow"] = [1, 1, 1, 1, 1]
            dic_layer_out["epoch_grow"] = [1, 1, 1, 1, 1]
            if key == 'standard_s':
                dic_layer_out["grow_size"] = {"0": 4, "1": 10, "2": 100}
                dic_layer_out["grow_base_size"] = {"0": 4, "1": 10, "2": 100}
                dic_layer_in["grow_size"] = {"1": 4, "2": 160, "3": 100}
                dic_layer_in["grow_base_size"] = {"1": 4, "2": 160, "3": 100}
        else:
            dic_layer_in["epoch_grow"] = [1, 2, 4, 7, 10]
            dic_layer_out["epoch_grow"] = [1, 2, 4, 7, 10]

        path_to_file = dic[key]

        path = os.path.dirname(path_to_file)

        files = [f for f in glob.glob(path + "/*__json_in_out.json")]

        file = files[0]

        with open(file, 'r') as f:
            data = json.load(f)

        for out in [True, False]:
            is_out = out
            path = os.path.dirname(file)

            if is_out:
                for layer in ["0", "1", "2"]:  #
                    dic_layer_out["layer"] = layer
                    x_y = gen_x_y(dic_layer_out, is_out=is_out, mode=mode)
                    plot_save(x_y, path, layer=layer, is_out=is_out, mode=mode)
            else:
                for layer in ["1", "2", "3"]:  #
                    dic_layer_in["layer"] = layer
                    x_y = gen_x_y(dic_layer_in, is_out=is_out, mode=mode)
                    plot_save(x_y, path, layer=layer, is_out=is_out, mode=mode)

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
