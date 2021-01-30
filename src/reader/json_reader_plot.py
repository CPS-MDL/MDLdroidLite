import json
import numpy as np
import matplotlib.pyplot as plt

j1 = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_randomMap_og2_new2_20200330-133546/grow_randomMap_og2_new2__json_in_out.json"
standard = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rankconnect_og2_new2_20200331-154540/grow_rankconnect_og2_new2__json_in_out.json"
stand = False

if not stand:
    with open(standard, 'r') as f:
        data = json.load(f)
    # dic = {
    #     'layer': 0,
    #     'size': 1,
    #     'from_epoch': 2,
    #     'to_epoch': 11,
    #     'grow': 0,
    #     'size_from': 0,
    #     'num_epoch': 9,
    # }

    dic = {
        'layer': 0,
        'size': 3,  # 2,3
        'from_epoch': 4,
        'to_epoch': 11,
        'grow': 0,
        'size_from': 0,
        'num_epoch': 7,
    }
else:
    with open(standard, 'r') as f:
        data = json.load(f)

    # dic = {
    #     'layer': 0,
    #     'size': 0,  # 2,3
    #     'from_epoch': 2,
    #     'to_epoch': 11,
    #     'grow': 0,
    #     'size_from': 0,
    #     'num_epoch': 9,
    # }

    dic = {
        'layer': 0,
        'size': 5,  # 4,5
        'from_epoch': 4,
        'to_epoch': 11,
        'grow': 0,
        'size_from': 0,
        'num_epoch': 7,
    }


def str_to_float(str):
    num_list = []
    str = str[1:-1]
    str_arry = str.split(', ')
    for num in str_arry:
        num_list.append(float(num))
    return np.asarray(num_list)


def get_value_from_json_exist(ele='S'):
    in_list = []
    out_list = []
    channel = dic['size']
    for epoch in range(dic['from_epoch'], dic['to_epoch']):
        for batch in range(600):
            out_str = data[str(epoch)][str(batch)][str(dic['layer'])][ele]['exist_out']
            outEle = str_to_float(out_str)[channel]
            out_list.append(outEle)
            in_str = data[str(epoch)][str(batch)][str(dic['layer'] + 1)][ele]['exist_in']
            inEle = str_to_float(in_str)[channel]
            in_list.append(inEle)
    return out_list, in_list


def get_value_from_json(ele='S'):
    in_list = []
    out_list = []
    og_list = []
    channel = dic['size']
    for epoch in range(dic['from_epoch'], dic['to_epoch']):
        for batch in range(0, 600):
            out_str = data[str(epoch)][str(batch)][str(dic['layer'])][ele]['out']
            outEle = str_to_float(out_str)[channel]
            out_list.append(outEle)

            in_str = data[str(epoch)][str(batch)][str(dic['layer'] + 1)][ele]['in']
            inEle = str_to_float(in_str)[channel]
            in_list.append(inEle)

            og_str = data[str(epoch)][str(batch)][str(dic['layer'] + 1)][ele]['og']
            ogEle = str_to_float(og_str)[channel]
            og_list.append(ogEle)
    return out_list, in_list, og_list


def epoch_mean(list, num_epoch=dic['num_epoch']):
    mean_list = []
    batch_size = 600
    for i in range(num_epoch):
        start = i * batch_size
        end = start + batch_size
        m = np.mean(list[start:end])
        mean_list.append(m)
    return mean_list


def plot(out_list=None, in_list=None, og_list=None, max=dic['num_epoch']):
    x = np.arange(dic['from_epoch'], dic['to_epoch'])
    ax = plt.subplot(111)
    if in_list is not None:
        ax.plot(x, in_list, label='in_' + str(dic['size']))
    if out_list is not None:
        ax.plot(x, out_list, label='out_' + str(dic['size']))
    if og_list is not None:
        ax.plot(x, og_list, label='og_' + str(dic['size']))
    ax.legend()
    ax.set_xlabel('Batch')
    ax.set_ylabel('S-score')
    # plt.xticks(np.arange(0, batchs * epochs, step=batchs), np.arange(0, epochs))
    plt.show()


if __name__ == '__main__':
    if not stand:
        out_list, in_list, og_list = get_value_from_json()
        out_mean = epoch_mean(out_list)
        in_mean = epoch_mean(in_list)
        og_mean = epoch_mean(og_list)
        plot(out_mean, in_mean, og_mean)
    else:
        out_list, in_list = get_value_from_json_exist()
        out_mean = epoch_mean(out_list)
        in_mean = epoch_mean(in_list)
        plot(out_list=out_mean, in_list=in_mean)
