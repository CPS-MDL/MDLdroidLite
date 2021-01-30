import json
import numpy as np
import os


def generator(window, full_data):
    for i in range(0, len(full_data), window):
        yield i, i + window


def read_length(data, layer):
    s_list = data[str(layer)]['bridging1']
    l = []
    for i in range(0, epoch * num_batches, num_batches):
        length = len(s_list[i])
        l.append(length)
    return l


if __name__ == "__main__":
    # path = "/Users/zber/Documents/FGdroid/exp_result/MD_3/bridging_1_s.json"
    # path = "/Users/zber/Documents/FGdroid/exp_result/MD_3/bridging_s3.json"
    # path = "/Users/zber/Documents/FGdroid/exp_result/MD_2/rank_standard_s.json"
    # path = "/Users/zber/Documents/FGdroid/exp_result/MD_2/ours_s.json"
    # path = "/Users/zber/Documents/FGdroid/exp_result/MD_2/rank_standard_new.json"
    # path = "/Users/zber/Documents/FGdroid/exp_result/A_1_variance/all_s.json"
    # path = "/Users/zber/Documents/FGdroid/exp_result/A_1_variance/channel_s.json"
    path = "/Users/zber/Documents/FGdroid/exp_result/A_2_S/mean_std_all.json"

    # e1 = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20]  # 18    1 + [1,4,4,4,3,3]
    # e2 = [5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 50, 50, 50, 50, 50, 50, 50]  # 23 [1,5,5,4,4,5]
    # e3 = [10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400, 420, 440, 460, 480, 500, 500, 500, 500, 500]  # 25 [1,5,5,5,5,4]

    e1 = [0, 1, 5, 9, 13, 16]
    end1 = 20

    e2 = [0, 1, 6, 11, 15, 19]
    end2 = 25

    e3 = [0, 1, 6, 11, 16, 21]
    end3 = 26

    layer1 = [(0, 2), (2, 6), (6, 10), (10, 14), (14, 17), (17, 20)]
    layer2 = [(0, 5), (5, 15), (15, 25), (25, 33), (33, 41), (41, 50)]
    layer3 = [(0, 10), (10, 100), (100, 200), (200, 300), (300, 400), (400, 500)]

    layer_guider = {
        '0': (e1, layer1, end1),
        '1': (e2, layer2, end2),
        '2': (e3, layer3, end3),
    }

    epoch = 30
    num_batches = 600
    num_layers = 3
    window_size = 50

    length_list = []

    growth = {}

    # load json
    with open(path, 'r') as f:
        data = json.load(f)

    # 1. select layer

    # for layer in range(3):
    #     el, ll, end = layer_guider[str(layer)]
    #
    #     growth[layer] = {}
    #
    #     # 2. get list
    #     for key in ['bridging1', 'bridging2', 'bridging3']:
    #         # for key in ['bridging2']:
    #         # for key in ['bridging1']:
    #
    #         growth[layer][key] = {}
    #
    #         s_list = data[str(layer)][key]
    #
    #         # 3. start, end index
    #         g = 0
    #         for e_from, s_e in zip(el, ll):
    #             start, end = s_e
    #             # 4. from epoch to end
    #             y = []
    #             y_window = []
    #             for e_index in range(e_from * num_batches, len(s_list)):
    #                 end1 = len(s_list[e_index])
    #                 if end1 < end:
    #                     e = end1
    #                 else:
    #                     e = end
    #
    #                 m = np.mean(s_list[e_index][start: e])
    #                 y.append(m)
    #
    #             for s_w, e_w in generator(50, y):
    #                 y_window.append(np.mean(y[s_w:e_w]))
    #
    #             growth[layer][key][g] = (e_from, y_window)
    #
    #             # if g not in growth[layer]:
    #             #     growth[layer][g] = (e_from, np.asarray(y_window))
    #             # else:
    #             #     growth[layer][g] = (growth[layer][g][0], np.vstack((growth[layer][g][1], np.asarray(y_window))))
    #
    #             g += 1

    # for layer in growth.keys():
    #     for g in growth[layer].keys():
    #         data = growth[layer][g][1]
    #         label = growth[layer][g][0]
    #         mean = np.mean(data, axis=0).tolist()
    #         std = np.std(data, axis=0).tolist()
    #         growth[layer][g] = (label, mean, std)

    # target_path = "/Users/zber/Documents/FGdroid/exp_result/MD_3/mean_std.json"

    # for layer in range(0, 3):
    #
    #     growth[layer] = {}
    #
    #     # 2. get list
    #     for key in ['rank1', 'rank2', 'rank3', 'copy1', 'copy2', 'copy3', 'bridging1', 'bridging2', 'bridging3', 'ours4', 'ours2', 'ours3']:
    #
    #         s_list = data[str(layer)][key]
    #         y_window = []
    #
    #         for s_w, e_w in generator(50, s_list):
    #             y_window.append(np.mean(s_list[s_w:e_w]))
    #
    #         if key[:-1] not in growth[layer]:
    #             growth[layer][key[:-1]] = np.asarray(y_window)
    #         else:
    #             growth[layer][key[:-1]] = np.vstack((growth[layer][key[:-1]], np.asarray(y_window)))
    #
    #     for key in ['standard']:
    #         # for key in ['bridging2']:
    #         # for key in ['bridging1']:
    #
    #         s_list = data[str(layer)][key]
    #         y_window = []
    #
    #         for s_w, e_w in generator(50, s_list):
    #             y_window.append(np.mean(s_list[s_w:e_w]))
    #
    #         if key not in growth[layer]:
    #             growth[layer][key] = y_window
    #
    #     for key in ['ours4']:
    #         # for key in ['bridging2']:
    #         # for key in ['bridging1']:
    #
    #         s_list = data[str(layer)][key]
    #         y_window = []
    #
    #         for s_w, e_w in generator(50, s_list):
    #             y_window.append(np.mean(s_list[s_w:e_w]))
    #
    #         if key not in growth[layer]:
    #             growth[layer][key] = y_window
    #
    # for layer in growth.keys():
    #     for g in growth[layer].keys():
    #         if g == 'ours4':
    #             continue
    #         data = growth[layer][g]
    #         mean = np.mean(data, axis=0).tolist()
    #         std = np.std(data, axis=0).tolist()
    #         growth[layer][g] = (mean, std)
    #
    # target_path = "/Users/zber/Documents/FGdroid/exp_result/MD_2/ours_s.json"

    # for layer in range(0, 3):
    #
    #     # 2. get list
    #     # for key in ['rank1', 'rank2', 'rank3', 'copy1', 'copy2', 'copy3', 'bridging1', 'bridging2', 'bridging3', 'ours1', 'ours2', 'ours3']:
    #     #
    #     #     s_list = data[str(layer)][key]
    #     #     y_window = []
    #     #
    #     #     for s_w, e_w in generator(50, s_list):
    #     #         y_window.append(np.mean(s_list[s_w:e_w]))
    #     #
    #     #     if key[:-1] not in growth[layer]:
    #     #         growth[layer][key[:-1]] = np.asarray(y_window)
    #     #     else:
    #     #         growth[layer][key[:-1]] = np.vstack((growth[layer][key[:-1]], np.asarray(y_window)))
    #
    #     # for key in ['standard']:
    #     #     # for key in ['bridging2']:
    #     #     # for key in ['bridging1']:
    #     #
    #     #     s_list = data[str(layer)][key]
    #     #     y_window = []
    #     #
    #     #     for s_w, e_w in generator(50, s_list):
    #     #         y_window.append(np.mean(s_list[s_w:e_w]))
    #     #
    #     #     if key not in growth[layer]:
    #     #         growth[layer][key] = y_window
    #     layer = str(layer)
    #

    # New size is:[4, 8, 16]
    # New size is:[7, 13, 26]
    # New size is:[12, 21, 42]
    # New size is:[20, 34, 68]
    # New size is:[20, 50, 109]
    # New size is:[20, 50, 175]
    # New size is:[20, 50, 280]
    # New size is:[20, 50, 448]
    # New size is:[20, 50, 500]
    # New size is:[20, 50, 500]
    # new = {}
    #
    # for layer in range(0, 3):
    #     layer = str(layer)
    #     new[layer] = {}
    #     for key in data[layer].keys():
    #
    #         if layer == "0":
    #             split = 4
    #             epoch = 5
    #         elif layer == "1":
    #             split = 13
    #             epoch = 9
    #         else:
    #             split = 26
    #             epoch = 9
    #
    #         key_data = []
    #
    #         for i in range(epoch * 600, epoch * 600 + 600):
    #             s_list = data[layer][key][i]
    #             o_data = s_list[:split]
    #             n_data = s_list[split:]
    #             o_mean = np.mean(o_data)
    #             n_mean = np.mean(n_data)
    #             # std = np.std([o_mean, n_mean])
    #             std = abs(o_mean - n_mean)
    #             key_data.append(std)
    #
    #         y_window = []
    #         for s_w, e_w in generator(50, key_data):
    #             y_window.append(np.mean(key_data[s_w:e_w]))
    #
    #         # if key not in data[layer]:
    #         new[layer][key] = y_window
    #
    # target_path = "/Users/zber/Documents/FGdroid/exp_result/A_1_variance/channel_variance_new.json"

    for layer in range(0, 3):
        layer = str(layer)

        for key in data[layer].keys():

            s_list = data[str(layer)][key]
            y_window = []
            y_std = []

            for s_w, e_w in generator(30, s_list[0]):
                y_window.append(np.mean(s_list[0][s_w:e_w]))
                y_std.append(np.mean(s_list[1][s_w:e_w]))

            # if key not in data[layer]:
            data[layer][key] = y_std

    target_path = "/Users/zber/Documents/FGdroid/exp_result/A_1_variance/std_all_new.json"

    with open(target_path, 'w') as f:
        json.dump(data, f, indent=4)
