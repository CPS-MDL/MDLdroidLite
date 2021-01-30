import json
import numpy as np

ours = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rankgroup_og2_new2_good/grow_rankgroup_og2_new2__json_log.json"
b_good = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_bridging_og2_new2_good/grow_bridging_og2_new2__json_log.json"
copy = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_randomMap_og2_new2_good/grow_randomMap_og2_new2__json_log.json"
rank = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rankconnect_og2_new2_good/grow_rankconnect_og2_new2__json_log.json"
b_b = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_bridging_og2_new2_20200327-210217/grow_bridging_og2_new2__json_log.json"
standard = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_og2_new2_20200327-215019/grow_rankconnect_og2_new2__json_log.json"

j3 = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_randomMap_og2_new2_20200330-133546/grow_randomMap_og2_new2__json_log.json"
with open(j3, 'r') as f:
    data1 = json.load(f)


def str_to_float(str):
    num_list = []
    str = str[1:-1]
    str_arry = str.split(', ')
    for num in str_arry:
        num_list.append(float(num))
    return np.asarray(num_list)


# with open(j3, 'r') as f:
#     data1 = json.load(f)

epoch = "10"
layers = ["0", "1", "2"]
before_after = [("B0", "A0"), ("B3", "A3"), ("B5", "A5")]

for i, b_a in enumerate(before_after):
    for layer in layers[i:]:
        for mode in ["W"]:
            for ele in ["M_mean", "M_std"]:
                str_title = 'L{},{},{} = '.format(layer, mode, ele)
                b = data1[epoch][b_a[0]][layer][mode][ele]
                a = data1[epoch][b_a[1]][layer][mode][ele]
                c = float(a) - float(b)
                print(str_title, c)
        for mode in ["S", "L1", "L2"]:
            for ele in ["layer"]:
                str_title = 'L{},{},{} = '.format(layer, mode, ele)
                b = data1[epoch][b_a[0]][layer][mode][ele]
                a = data1[epoch][b_a[1]][layer][mode][ele]
                c = float(a) - float(b)
                print(str_title, c)
        print()
    print()
    print()

for i, b_a in enumerate(before_after):
    for layer in layers[i:]:
        for mode in ["W"]:
            for ele in ["mean", "std"]:
                str_title = 'L{},{},{} = '.format(layer, mode, ele)
                b = data1[epoch][b_a[0]][layer][mode][ele]
                a = data1[epoch][b_a[1]][layer][mode][ele]
                # c = str_to_float(a) - str_to_float(b)
                print(str_title, '-Before', b)
                print(str_title, '-After', a)
        for mode in ["S", "L1", "L2"]:
            for ele in ["channel"]:
                str_title = 'L{},{},{} = '.format(layer, mode, ele)
                b = data1[epoch][b_a[0]][layer][mode][ele]
                a = data1[epoch][b_a[1]][layer][mode][ele]
                # c = str_to_float(a) - str_to_float(b)
                print(str_title, '-Before', b)
                print(str_title, '-After', a)
        print()
    print()
    print()

for i, b_a in enumerate(before_after):
    for layer in layers[(i + 1):]:
        for mode in ["W"]:
            for ele in ["mean", "std"]:
                str_title = 'L{},{},{} = '.format(layer, mode, ele)
                b = data1[epoch][b_a[0]][layer][mode][ele]
                a = data1[epoch][b_a[1]][layer][mode][ele]
                c = str_to_float(a) - str_to_float(b)
                print(str_title, c.tolist())
        for mode in ["S", "L1", "L2"]:
            for ele in ["channel"]:
                str_title = 'L{},{},{} = '.format(layer, mode, ele)
                b = data1[epoch][b_a[0]][layer][mode][ele]
                a = data1[epoch][b_a[1]][layer][mode][ele]
                c = str_to_float(a) - str_to_float(b)
                print(str_title, c.tolist())
        print()
    print()
    print()
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
