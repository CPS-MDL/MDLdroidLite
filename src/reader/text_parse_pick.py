import glob
import os
import numpy as np
import json
import copy

#################
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

    """ /Users/zber/ProgramDev/exp_pyTorch/results/GC_MNIST_LeNet_H5_e05
        /Users/zber/ProgramDev/exp_pyTorch/results/GC_MNIST_LeNet_H6_e06
        /Users/zber/ProgramDev/exp_pyTorch/results/GC_MNIST_LeNet_H5_e06
        /Users/zber/ProgramDev/exp_pyTorch/results/GC_MNIST_LeNet_H2_e06
        /Users/zber/ProgramDev/exp_pyTorch/results/GC_MNIST_LeNet_H6_e05
        /Users/zber/ProgramDev/exp_pyTorch/results/GC_MNIST_LeNet_H2_e05 """
    #
    #
    #
    # file = "/Users/zber/Documents/FGdroid/result/VGG/VGG_BN_1/GC_CIFAR10_VGG__log.txt"
    # file = "/Users/zber/ProgramDev/exp_pyTorch/results/MNIST_H4_2/GC_MNIST_LeNet__log.txt"
    # file = "/Users/zber/ProgramDev/exp_pyTorch/results/LeNet_Horizon/H2/GC_MNIST_LeNet__log.txt"
    # file = "/Users/zber/ProgramDev/exp_pyTorch/results/LeNet_Horizon/H3/GC_MNIST_LeNet__log.txt"
    # file = "/Users/zber/ProgramDev/exp_pyTorch/results/LeNet_Horizon/H4/GC_MNIST_LeNet__log.txt"
    # file = "/Users/zber/ProgramDev/exp_pyTorch/results/LeNet_Horizon/H5/GC_MNIST_LeNet__log.txt"
    # file = "/Users/zber/ProgramDev/exp_pyTorch/results/LeNet_Horizon/H6/GC_MNIST_LeNet__log.txt"
    # file = "/Users/zber/ProgramDev/exp_pyTorch/results/GC_Har_LeNet_v1_best/GC_Har_LeNet__log.txt"

    # Horizon v2 Lenet mnist
    # file = "/Users/zber/ProgramDev/exp_pyTorch/results/v2_Lenet/GC_MNIST_LeNet_TH1_1/GC_MNIST_LeNet__log.txt"
    # file = "/Users/zber/ProgramDev/exp_pyTorch/results/v2_Lenet/GC_MNIST_LeNet_TH2/GC_MNIST_LeNet__log.txt"
    # file = "/Users/zber/ProgramDev/exp_pyTorch/results/v2_Lenet/GC_MNIST_LeNet_TH3/GC_MNIST_LeNet__log.txt"
    # file = "/Users/zber/ProgramDev/exp_pyTorch/results/v2_Lenet/GC_MNIST_LeNet_best4/GC_MNIST_LeNet__log.txt"
    # file = "/Users/zber/ProgramDev/exp_pyTorch/results/v2_Lenet/GC_MNIST_LeNet_TH5/GC_MNIST_LeNet__log.txt"

    # Horizon v2 MBnet Har
    # file="/Users/zber/ProgramDev/exp_pyTorch/results/v2_MBnet/GC_Har_MobileNet_TH1/GC_Har_MobileNet__log.txt"
    # file="/Users/zber/ProgramDev/exp_pyTorch/results/v2_MBnet/GC_Har_MobileNet_TH2/GC_Har_MobileNet__log.txt"
    # file="/Users/zber/ProgramDev/exp_pyTorch/results/v2_MBnet/GC_Har_MobileNet_TH3/GC_Har_MobileNet__log.txt"
    # file="/Users/zber/ProgramDev/exp_pyTorch/results/v2_MBnet/GC_Har_MobileNet_TH4/GC_Har_MobileNet__log.txt"
    # file="/Users/zber/ProgramDev/exp_pyTorch/results/v2_MBnet/GC_Har_MobileNet_TH5/GC_Har_MobileNet__log.txt"
    # file ="/Users/zber/ProgramDev/exp_pyTorch/results/v2_MBnet/GC_Har_MobileNet_best5/GC_Har_MobileNet__log.txt"
    file ="/Users/zber/ProgramDev/exp_pyTorch/results/Har_MB/GC_Har_MobileNet_H3_best1/GC_Har_MobileNet__log.txt"


    target_file = os.path.join(os.path.dirname(file), "size_structure.txt")

    my_file = open(target_file, "w")

    # with open(file, "r") as f:
    #     for line in f:
    #         if 'pick' in line:
    #             my_file.write(line)

    with open(file, "r") as f:
        for line in f:
            if "Model new size:" in line:
            # if "New size is:" in line:
                start = 0
                end = line.index(']')
                my_file.write(line[start:end+1])
                my_file.write('\n')
            # if "Controller growth:" in line:
            #     start = 0
            #     end = line.index(']')
            #     my_file.write(line[start:end+1])
            #     my_file.write('\n')
            #     my_file.write(line)

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
