import json
import numpy as np

j1 = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rankconnect_og2_new2_20200401-223313/grow_rankconnect_og2_new2__json_dic.json"
j2 = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rankconnect_og2_new2_20200330-154756/grow_rankconnect_og2_new2__json_dic.json"
j3 = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rankconnect_og2_new2_20200331-154540/grow_rankconnect_og2_new2__json_dic.json"
j4 = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rankconnect_og2_new2_20200401-232151/grow_rankconnect_og2_new2__json_dic.json"
j5 = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rankconnect_og2_new2_20200401-235354/grow_rankconnect_og2_new2__json_dic.json"
with open(j5, 'r') as f:
    data = json.load(f)


def str_to_float(str):
    num_list = []
    str = str[1:-1]
    str_arry = str.split(', ')
    for num in str_arry:
        num_list.append(float(num))
    return np.asarray(num_list)


# with open(j3, 'r') as f:
#     data1 = json.load(f)
t_grow = 2

layers = ["1", "2", "3"]

l1 = []
l2 = []
l3 = []
for l in layers:
    if l == "1":
        str_list = data[l]["og"]
        # str_list = str_to_float(str_list)
        l1 = str_list
    if l == "2":
        str_list = data[l]["og"]
        # str_list = str_to_float(str_list)
        for i in range(t_grow * 3):
            l2.append(str_list[i * 16])
        f = np.asarray(l2) / 16
        l2 = f.tolist()
    if l == "3":
        str_list = data[l]["og"]
        # str_list = str_to_float(str_list)
        l3 = str_list
print(l1)
print(l2)
print(l3)

l1 = []
l2 = []
l3 = []
for l in layers:
    if l == "1":
        str_list = data[l]["cog"]
        # str_list = str_to_float(str_list)
        l1 = str_list
    if l == "2":
        str_list = data[l]["cog"]
        # str_list = str_to_float(str_list)
        for i in range(t_grow * 3):
            l2.append(str_list[i * 16])
        f = np.asarray(l2) / 16
        l2 = f.tolist()
    if l == "3":
        str_list = data[l]["cog"]
        # str_list = str_to_float(str_list)
        l3 = str_list
print(l1)
print(l2)
print(l3)
# for i in range(0, 3):
#     for layer in layers[i:]:
#         for mode in ["W"]:
#             for ele in ["M_mean", "M_std"]:
#                 str_title = 'L{},{},{} = '.format(layer, mode, ele)
#                 b = data1[epoch]["B0"][layer][mode][ele]
#                 a = data1[epoch]["A0"][layer][mode][ele]
#                 c = float(a) - float(b)
#                 print(str_title, c)
#         for mode in ["S", "L1", "L2"]:
#             for ele in ["layer"]:
#                 str_title = 'L{},{},{} = '.format(layer, mode, ele)
#                 b = data1[epoch]["B0"][layer][mode][ele]
#                 a = data1[epoch]["A0"][layer][mode][ele]
#                 c = float(a) - float(b)
#                 print(str_title, c)
#         print()
#     print()
#     print()
#
# for i in range(0, 3):
#     for layer in layers[i:]:
#         for mode in ["W"]:
#             for ele in ["mean", "std"]:
#                 str_title = 'L{},{},{} = '.format(layer, mode, ele)
#                 b = data1[epoch]["B0"][layer][mode][ele]
#                 a = data1[epoch]["A0"][layer][mode][ele]
#                 # c = str_to_float(a) - str_to_float(b)
#                 print(str_title, '-Before', b)
#                 print(str_title, '-After', a)
#         for mode in ["S", "L1", "L2"]:
#             for ele in ["channel"]:
#                 str_title = 'L{},{},{} = '.format(layer, mode, ele)
#                 b = data1[epoch]["B0"][layer][mode][ele]
#                 a = data1[epoch]["A0"][layer][mode][ele]
#                 # c = str_to_float(a) - str_to_float(b)
#                 print(str_title, '-Before', b)
#                 print(str_title, '-After', a)
#         print()
#     print()
#     print()
#
# for i in range(1, 3):
#     for layer in layers[i:]:
#         for mode in ["W"]:
#             for ele in ["mean", "std"]:
#                 str_title = 'L{},{},{} = '.format(layer, mode, ele)
#                 b = data1[epoch]["B0"][layer][mode][ele]
#                 a = data1[epoch]["A0"][layer][mode][ele]
#                 c = str_to_float(a) - str_to_float(b)
#                 print(str_title, c.tolist())
#         for mode in ["S", "L1", "L2"]:
#             for ele in ["channel"]:
#                 str_title = 'L{},{},{} = '.format(layer, mode, ele)
#                 b = data1[epoch]["B0"][layer][mode][ele]
#                 a = data1[epoch]["A0"][layer][mode][ele]
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
