import json
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

dic = dict(
    rank_ours_low="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_ours_low_batch_test",
    # standard="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rankconnect_og2_new2_20200331-154540/grow_rankconnect_og2_new2__json_in_out.json",
    standard_10_17_50="/Users/zber/ProgramDev/exp_pyTorch/results/standard_10_17_50_batch_test",
)

acc = []
time = []


def str_to_float(str):
    num_list = []
    str = str[1:-1]
    str_arry = str.split(', ')
    for num in str_arry:
        num_list.append(float(num))
    return np.asarray(num_list)


def read_batch_acc(file_path):
    files = [f for f in glob.glob(file_path + "/*__log.txt")]
    file = files[0]
    acc_str = None
    with open(file, "r") as f:
        for line in f:
            if line.startswith('batch test acc:'):
                start = line.index('[')
                end = line.index(']')
                acc_str = line[start:end + 1]
    ac = str_to_float(acc_str)
    acc.append(ac)


def accumulative_time(batch_time):
    accumulative_times = []
    for i in range(1, len(batch_time) + 1):
        accumulative_times.append(sum(batch_time[:i]))
    return accumulative_times


def read_batch_time(file_path):
    files = [f for f in glob.glob(file_path + "/*__Timer.json")]
    file = files[0]
    with open(file, 'r') as f:
        data = json.load(f)
    batch_time = data['batch']
    accum_times = accumulative_time(batch_time)
    time.append(accum_times)


def plot_save(xs, ys, dic):
    colour_list = ['b', 'g', 'yellow', 'black']
    # path = os.path.join(dir, 'Distance_L{}_{}.png'.format(layer, ('out' if is_out else 'in')))
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 8))
    for x, y, colour, label, in zip(xs, ys, colour_list, dic.keys()):
        ax.plot(x, y, colour, label=label)

    ax.legend()
    plt.show()
    # fig.savefig(path)
    # plt.close(fig)


if __name__ == '__main__':

    for key in dic.keys():
        file_path = dic[key]

        read_batch_acc(file_path)
        read_batch_time(file_path)

    plot_save(time, acc, dic)
