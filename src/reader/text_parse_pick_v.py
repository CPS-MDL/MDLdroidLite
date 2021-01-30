import os
import numpy as np
import matplotlib.pyplot as plt

#################
def str_to_float(str, is_numpy=False):
    num_list = []
    str = str[1:-1]
    str_arry = str.split(', ')
    for num in str_arry:
        num_list.append(float(num))
    if is_numpy:
        return np.asarray(num_list)
    else:
        return num_list


def plot_save(x_y):

    path = os.path.join(dir_name, 'v_graph.png')
    fig, ax = plt.subplots(nrows=num_layers, ncols=1, figsize=(8, 24))
    for i in range(num_layers):
        for j in range(growth_epoch):
            index = i + j*num_layers
            ax[i].plot(x_y[index], label='grow_{}'.format(j))
    for i in range(num_layers):
        ax[i].legend(loc="upper right")
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    # file = "/Users/zber/Documents/FGdroid/result/6-8/df07-newscale.txt"
    # file = "/Users/zber/Documents/FGdroid/result/6-8/df07-oldscale.txt"
    # file = "/Users/zber/Documents/FGdroid/result/6-8/curve_large_bound.txt"
    # file = "/Users/zber/Documents/FGdroid/result/6-8/df07-newscale.txt"
    # file = "/Users/zber/Documents/FGdroid/result/6-8/cost.txt"
    # file = "/Users/zber/Documents/FGdroid/result/6-8/log.txt"
    # file = "/Users/zber/Documents/FGdroid/result/6-8/C100_expCost/grow_rank_cosine_log.txt"
    # file = "/Users/zber/Documents/FGdroid/result/6-8/C100_diffCost_df07/grow_rank_cosine_log_df_06_e20.txt"
    # file = "/Users/zber/ProgramDev/exp_pyTorch/results/GC_Har_MobileNet_20200621-021711/GC_Har_MobileNet__log.txt"

    #

    file = "/Users/zber/ProgramDev/exp_pyTorch/results/GC_Har_MobileNet_20200627-185753/GC_Har_MobileNet__log.txt"
    dir_name = os.path.dirname(file)

    total =[]
    num_layers = 3
    growth_epoch = 5

    with open(file, "r") as f:
        for line in f:
            if "V:" in line:
                start = line.index('*')
                end = line.index('&')
                # v = str_to_float(line[start+3:end-3])
                v = str_to_float(line[start+1:end])
                total.append(v)

    plot_save(total)





            # if "Model new size:" in line:
            #     start = 0
            #     end = line.index(']')
            #     my_file.write(line[start:end+1])
            #     my_file.write('\n')

            # if "=====" in line:
            #     my_file.write(line)
            #     my_file.write('\n')



    # new = 0
    # old = 0
    # with open(target_file, "r") as f:
    #     for line in f:
    #         if 'old' in line:
    #             old+=1
    #         if 'new' in line:
    #             new+=1
    #
    # my_file = open(target_file, "a")
    # my_file.write('num of old is :{}'.format(old))
    # my_file.write('num of new is :{}'.format(new))