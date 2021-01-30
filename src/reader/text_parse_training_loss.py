import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import json
import copy

dic_base = dict(
    # standard_10_17_50="/Users/zber/ProgramDev/exp_pyTorch/results/standard_10_17_50/layer1_log.txt",
    # standard_8_14_40="/Users/zber/ProgramDev/exp_pyTorch/results/standard_8_14_40/layer1_log.txt",
    # standard_6_11_30="/Users/zber/ProgramDev/exp_pyTorch/results/standard_6_11_30/layer1_log.txt",
    # standard_4_8_20="/Users/zber/ProgramDev/exp_pyTorch/results/standard_4_8_20/layer1_log.txt",
    # standard_2_5_10="/Users/zber/ProgramDev/exp_pyTorch/results/standard_2_5_10/layer1_log.txt",
    # standard_20_50_500="/Users/zber/ProgramDev/exp_pyTorch/results/grow_standard_20_50_500/layer1_log.txt",

    # rank_baseline="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_baseline_20200430-190631/layer1_log.txt",
    # random="/Users/zber/ProgramDev/exp_pyTorch/results/grow_random_20200430-194619/layer1_log.txt",
    # rank_baseline_LR4="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_baseline_LR4_20200413-001813/layer2_log.txt",
    # copy_n="/Users/zber/ProgramDev/exp_pyTorch/results/grow_copy_n_20200430-191941/layer1_log.txt",
    # bridging="/Users/zber/ProgramDev/exp_pyTorch/results/grow_bridging_20200430-193258/layer1_log.txt",
    # rank_cumulative="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_20200430-213135/layer1_log.txt",
    # rank_cumulative_cosine="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_cosine_20200430-214432/layer1_log.txt",
    # rank_2R="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_2R/layer1_log.txt",

    # # 5-3
    # rank_ours_batch20="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_ours_batch20_no2R/layer1_log.txt",
    # rank_cumulative_batch20="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_batch20_no2R/layer1_log.txt",

    # rank_cumulative="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_20200430-213135/layer1_log.txt",
    # rank_cumulative_cosine="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_cosine_20200430-214432/layer1_log.txt",
    #
    # rank_consine="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_20200501-195838/layer1_log.txt",
    # rank_one_all2="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_one_all2/layer1_log.txt",
    # 5-4
    # rank_ours_batch20="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_ours_batch20/layer1_log.txt",
    # rank_cumulative_batch20="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_batch20/layer1_log.txt",
    # rank_ours_batch100="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_ours_batch100/layer1_log.txt",
    # rank_cumulative_batch100="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_batch100/layer1_log.txt",

    # 5-7
    # rank_ours_normal = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_ours_max_normal_noise/layer1_log.txt",
    # standard_20_50_500="/Users/zber/ProgramDev/exp_pyTorch/results/grow_standard_20_50_500/layer1_log.txt",
    # copy_n_overP="/Users/zber/ProgramDev/exp_pyTorch/results/grow_copy_n_overP/layer1_log.txt",
    # rank_baseline_overP="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_baseline_overP/layer1_log.txt",
    # random_overP="/Users/zber/ProgramDev/exp_pyTorch/results/grow_random_overP/layer1_log.txt",
    # bridging_overP="/Users/zber/ProgramDev/exp_pyTorch/results/grow_bridging_overP/layer1_log.txt",

    # 5-8
    # standard_reg_10_17_50="/Users/zber/ProgramDev/exp_pyTorch/results/standard_10_17_50_reg1/layer1_log.txt",
    # standard_10_17_50="/Users/zber/ProgramDev/exp_pyTorch/results/standard_10_17_50/layer1_log.txt",
    # bridging="/Users/zber/ProgramDev/exp_pyTorch/results/grow_bridging_20200508-223706/layer1_log.txt",
    # rank_baseline="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_baseline_20200508-224355/layer1_log.txt",
    # random="/Users/zber/ProgramDev/exp_pyTorch/results/grow_random_20200508-225047/layer1_log.txt",
    # copy_n="/Users/zber/ProgramDev/exp_pyTorch/results/grow_copy_n_20200508-225737/layer1_log.txt",
    # rank_ours="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_ours_noise_bias_scale/layer1_log.txt",
    # random_reg1="/Users/zber/ProgramDev/exp_pyTorch/results/grow_random_reg0.1/layer1_log.txt",
    # random_reg2="/Users/zber/ProgramDev/exp_pyTorch/results/grow_random_reg0.01/layer1_log.txt",
    # random_reg3="/Users/zber/ProgramDev/exp_pyTorch/results/grow_random_reg0.001/layer1_log.txt",
    # random_reg4="/Users/zber/ProgramDev/exp_pyTorch/results/grow_random_reg0.0001/layer1_log.txt",
    # random_reg5="/Users/zber/ProgramDev/exp_pyTorch/results/grow_random_reg0.00001/layer1_log.txt",
    # random_gate_10_05_test="/Users/zber/ProgramDev/exp_pyTorch/results/grow_random_gate_10_05/layer1_log.txt",
    # random_gate_15_10_test="/Users/zber/ProgramDev/exp_pyTorch/results/grow_random_gate_15-10/layer1_log.txt",

    # low_cosine_LR_05="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_LR_05_20200511-143658/layer1_log.txt",
    # low_cosine_LR_2="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_LR_2_20200511-142440/layer1_log.txt",
    # low_cosine_LR_10="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_LR_10_20200511-182123/layer1_log.txt",
    # low_cosine_LR_20="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_LR20/layer1_log.txt",
    # low_cosine="/Users/zber/ProgramDev/exp_pyTorch/results/grow_low_cosine_20200511-000208/layer1_log.txt",
    # base_line="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_baseline_20200514-134919/layer1_log.txt",
    # base_line_1="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_baseline_20200513-223311/layer1_log.txt",

    # 5-15
    # low_cosine_Sdecay_300 = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_Sdecay_300_20200515-183221/layer1_log.txt",
    # low_cosine_Sdecay_600 = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_Sdecay_600_20200515-172914/layer1_log.txt",
    # low_cosine_converge_e3="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_20200515-222941/layer1_log.txt",
    # low_cosine_converge_e5="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_20200515-224212/layer1_log.txt",
    # low_cosine_asy_converge_e5="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_Asy_converge_E5_20200515-230947/layer1_log.txt",
    # low_cosine_asy_reverse_converge_e5="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_Asy_reverse_converge_E5_20200515-232619/layer1_log.txt",
    # low_cosine_converge_e5_LR = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_converge_E5_150_LR_20200516-000544/layer1_log.txt",
    # low_cosine_max_select_converge="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_max_select_converge_E5_150_20200516-150254/layer1_log.txt",
    # low_cosine_select_converge="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_select_converge_E5_150_20200516-145223/layer1_log.txt",
    # low_cosine_max_select_converge_LR2_="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_max_select_converge_E5_150_LR2_20200516-154034/layer1_log.txt",
    # low_cosine_max_select_converge_LR4="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_max_select_converge_E5_150_LR4_20200516-153312/layer1_log.txt",
    # cosine_max_select_converge_E5_150_LR2_decay="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_max_select_converge_E5_150_LR2_decay_20200516-160510/layer1_log.txt",
    # cosine_max_select_converge_E5_150_LR4_decay="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_max_select_converge_E5_150_LR4_decay_20200516-160142/layer1_log.txt",
    #
    # cosine_max_select_converge_E5_150_LR2_decay_S3="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_max_select_converge_E5_150_LR2_decay_S3_20200516-172456/layer1_log.txt",
    # cosine_max_select_converge_E5_150_LR4_decay_S3="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_max_select_converge_E5_150_LR4_decay_S3_20200516-171602/layer1_log.txt",
    #
    # cosine_max_select_converge_LR2_decay_lambda="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_max_select_converge_E5_150_LR2_decay_Lambda_20200516-182642/layer1_log.txt",
    # cosine_max_select_converge_LR4_decay_lambda="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_max_select_converge_E5_150_LR4_decay_Lambda_20200516-182202/layer1_log.txt",

    # cosine_max_select_converge_copy_optimizer_EXdecay_LR2="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_max_select_converge_copy_optimizer_EXdecay_LR2_20200516-202511/layer1_log.txt",
    # cosine_max_select_converge_renew_optimier_LR2="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_max_select_converge_renew_optimier_LR2_20200516-205802/layer1_log.txt",
    # cosine_max_select_converge_copy_optimizer_LR2="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_max_select_converge_copy_optimizer_LR2_20200516-204614/layer1_log.txt",
    # cosine_converge_LR2decay_LMDdecayAsyR_OPrenew_E10="/Users/zber/ProgramDev/exp_pyTorch/results/cosine_converge_LR2decay_LMDdecayAsyR_OPrenew_E10_20200517-204159/layer1_log.txt",
    # cosine_converge_LR2decay_LMDdecay_OPrenew_E10="/Users/zber/ProgramDev/exp_pyTorch/results/cosine_converge_LR2decay_LMDdecay_OPrenew_E10_20200517-175337/layer1_log.txt",
    # cosine_converge_LR2decay_LMDdecayAsy_OPrenew_E10="/Users/zber/ProgramDev/exp_pyTorch/results/cosine_converge_LR2decay_LMDdecayAsy_OPrenew_E10_20200517-223242/layer1_log.txt",
    # cosine_bad="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_20200517-225847/layer1_log.txt",

    # 5-19
    # good_cosine="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_20200518-151420/layer1_log.txt",
    # cosine_Asy_Xavier="/Users/zber/ProgramDev/exp_pyTorch/results/cosine_Asy_Xavier_20200518-204628/layer1_log.txt",
    # cosine_AsyR_Xavier="/Users/zber/ProgramDev/exp_pyTorch/results/cosine_AsyR_Xavier_20200518-190439/layer1_log.txt",
    # cosine_RI_600="/Users/zber/ProgramDev/exp_pyTorch/results/cosine_RI_step600_20200518-222843/layer1_log.txt",
    # cosine_RI_bias_scale="/Users/zber/ProgramDev/exp_pyTorch/results/cosine_RI_bias_scale_20200519-192346/layer1_log.txt",
    # cosine_RI_back_T10="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_lambda_back_T10_20200519-214351/layer1_log.txt",
    # cosine="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_20200519-223044/layer1_log.txt",
    # cosine_T3="/Users/zber/ProgramDev/exp_pyTorch/results/cosine_lambda_back_T3_20200519-224154/layer1_log.txt",
    # cosine_RI_std="/Users/zber/ProgramDev/exp_pyTorch/results/cosine_RI_Sstd_20200520-000613/layer1_log.txt",
    # cosine_ri="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_20200521-022715/layer1_log.txt",
    # cosine_gate="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_20200522-003301/layer1_log.txt",
    # cosine_RI_gradGate_lambdaR= "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_20200522-012755/layer1_log.txt",
    # cosine_momentum_weightBack="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_20200522-222818/layer1_log.txt",

    # 5-22
    # standard_2_5_10="/Users/zber/ProgramDev/exp_pyTorch/results/standard_2_5_10_20200514-192251/layer1_log.txt",
    # standard_4_8_20="/Users/zber/ProgramDev/exp_pyTorch/results/standard_4_8_20_20200514-192745/layer1_log.txt",
    # standard_6_11_30="/Users/zber/ProgramDev/exp_pyTorch/results/standard_6_11_30_20200514-193258/layer1_log.txt",
    # standard_8_14_40="/Users/zber/ProgramDev/exp_pyTorch/results/standard_8_14_40_20200514-193833/layer1_log.txt",
    standard_10_17_50="/Users/zber/ProgramDev/exp_pyTorch/results/standard_10_17_50_20200514-194428/standard_10_17_50__log.txt",
    # standard_8_17_23="/Users/zber/ProgramDev/exp_pyTorch/results/standard_8_17_23/standard_8_17_23__log.txt",
    #
    # rank_baseline="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_baseline_20200514-195931/layer1_log.txt",
    # random="/Users/zber/ProgramDev/exp_pyTorch/results/grow_random_20200514-200640/layer1_log.txt",
    # copy_n="/Users/zber/ProgramDev/exp_pyTorch/results/grow_copy_n_20200514-201348/layer1_log.txt",
    # bridging="/Users/zber/ProgramDev/exp_pyTorch/results/grow_bridging_20200514-195225/layer1_log.txt",
    # low_cosine="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_20200514-171924/layer1_log.txt",
    # cosine_RI_momentum="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_20200522-233322/layer1_log.txt",

    # 5-25
    # standard_10_5_50="/Users/zber/ProgramDev/exp_pyTorch/results/standard_10_5_50_20200523-224248/layer1_log.txt",
    # ours_R="/Users/zber/ProgramDev/exp_pyTorch/results/grow_ours_R_20200525-222448/layer1_log.txt",
    # random="/Users/zber/ProgramDev/exp_pyTorch/results/grow_random_20200525-215557/layer1_log.txt",
    # ours="/Users/zber/ProgramDev/exp_pyTorch/results/grow_ours20200525-213437/layer1_log.txt",

    # new_numpy="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_20200528-175713/layer1_log.txt",
    # new = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_20200528-170457/layer1_log.txt",
    # old="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_20200528-170111/layer1_log.txt",
    # new1="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_20200528-192040/layer1_log.txt"

    # cosine_L_1_index="/Users/zber/ProgramDev/exp_pyTorch/results/cosine_L_1_index_20200530-192920/layer1_log.txt",
    # cosine_L_L_index="/Users/zber/ProgramDev/exp_pyTorch/results/cosine_L_L_index_20200530-194103/layer1_log.txt",
    # cosine_L_1L_index="/Users/zber/ProgramDev/exp_pyTorch/results/cosine_L_1L_index_20200530-195356/layer1_log.txt",

    # std_10="/Users/zber/Documents/FastCNN/exp_result/loss/standard_2_5_50__log.txt",
    # std_30="/Users/zber/Documents/FastCNN/exp_result/loss/standard_6_15_150__log.txt",
    # std_60="/Users/zber/Documents/FastCNN/exp_result/loss/standard_12_30_300__log.txt",
    # std_100="/Users/zber/Documents/FastCNN/exp_result/loss/standard_20_50_500__log.txt",

    # vgg11 = "/Users/zber/ProgramDev/exp_pyTorch/results/CIFAR10_VGG11_20200510-214917",
    # vgg11_10 = "/Users/zber/ProgramDev/exp_pyTorch/results/CIFAR10_VGG11_10_20200617-210959",
    # vgg11_25 = "/Users/zber/ProgramDev/exp_pyTorch/results/CIFAR10_VGG11_25_20200614-202446",
    # vgg11_40 = "/Users/zber/ProgramDev/exp_pyTorch/results/CIFAR10_VGG11_40_20200618-030323",
    # vgg11_50="/Users/zber/ProgramDev/exp_pyTorch/results/CIFAR10_VGG11_50_20200618-004841",
    # mb_har="/Users/zber/ProgramDev/exp_pyTorch/results/G_Har_MobileNet20200620-174316",
    # mb_har_bn="/Users/zber/ProgramDev/exp_pyTorch/results/G_Har_MobileNetBN_20200620-192435",
    # standard="/Users/zber/ProgramDev/exp_pyTorch/results/Standard_Har_MobileNet_20200620-174936",
    # standard_bn="/Users/zber/ProgramDev/exp_pyTorch/results/Standard_Har_MobileNetBN_20200620-181236",
    new = "/Users/zber/ProgramDev/exp_pyTorch/results/Standard_MNIST_LeNet_20200630-003814/Standard_MNIST_LeNet__log.txt",
    old = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_20200522-233322/grow_rank_cosine__log.txt"

)

num_growth = 5
dic = dic_base
acc = []


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


def print_acc(file):
    global acc
    path = os.path.dirname(file)

    files = [f for f in glob.glob(path + "/*_log.txt")]
    file = files[0]

    with open(file, "r") as f:
        for line in f:
            if line.startswith('Train_loss_mean'):
                acc.append(line[16:-1])


def generator(length, num):
    start = 0
    for i in range(1, 1 + length):
        start = (i - 1) * num
        end = start + num
        yield start, end


def windowz(data, size):
    start = 0
    while start < len(data):
        yield start, start + size
        start += size


def print_acc_batch(file):
    global acc
    # path = os.path.dirname(file)
    # path = file
    a_c = []

    # files = [f for f in glob.glob(path + "/*_log.txt")]
    # file = files[0]

    with open(file, "r") as f:
        for line in f:
            l = None
            mean_list = []
            if line.endswith(';\n'):
                start = line.index('[')
                end = line.index(']')
                l = str_to_float(line[start:end + 1])
                # if 'Epoch_10' in line:
                #     l = str_to_float(line[33:-2])
                # else:
                #     l = str_to_float(line[32:-2])
                for start, end in generator(window_size, num_window):
                    m = np.mean(l[start: end])
                    mean_list.append(m)
                a_c = a_c + mean_list
    acc.append(a_c)


def print_acc_batch_save_dic(file, key):
    global acc_dic
    a_c = []

    with open(file, "r") as f:
        for line in f:
            l = None
            mean_list = []
            if line.endswith(';\n'):
                start = line.index('[')
                end = line.index(']')
                l = str_to_float(line[start:end + 1])
                for start, end in generator(window_size, num_window):
                    m = np.mean(l[start: end])
                    mean_list.append(m)
                a_c = a_c + mean_list
    acc_dic[key] = a_c


def plot_epoch(acc, label_list):
    accuracy_list = []

    for ac in acc:
        accuracy = str_to_float(ac)
        accuracy_list.append(accuracy)
    colour_list = ['r', 'b', 'g', 'orange', 'black', 'purple']

    # path = os.path.join(dir, 'L{}_{}.png'.format(layer, ('out' if is_out else 'in')))
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 8))
    for accuracy, colour, label, in zip(accuracy_list, colour_list, label_list):
        x = np.arange(len(accuracy))
        ax.plot(x, accuracy, colour, label=label)
    ax.legend()
    plt.show()
    # fig.savefig(path)
    # plt.close(fig)


def plot_batch(acc, label_list):
    grow_epochs = [1, 3, 6, 9]
    # grow_epochs = [1, 1.25, 2, 3, 4, 5]

    accuracy_list = acc

    colour_list = ['r', 'b', 'g', 'orange', 'black', 'purple', 'pink']

    # path = os.path.join(dir, 'L{}_{}.png'.format(layer, ('out' if is_out else 'in')))
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 8))
    for accuracy, colour, label, in zip(accuracy_list, colour_list, label_list):
        x = np.arange(len(accuracy))
        ax.plot(x, accuracy, colour, label=label, alpha=0.5, linewidth=2.0)
        # if label == "new":
        #     ax.plot(x, accuracy, colour, label=label, alpha=0.5, linewidth=2.0)
        # else:
        #     ax.plot(x, accuracy, colour, label=label, alpha=0.5)

    for i in grow_epochs:
        point_x = i * (num_batches / window_size) - 1
        plt.axvline(x=point_x, color='grey', linestyle='--', alpha=0.5)

    ax.legend()
    plt.show()


if __name__ == '__main__':
    # num_batches = 230
    # window_size = 23
    # num_window = num_batches // window_size

    num_batches = 600
    window_size = 50
    num_window = num_batches // window_size

    key_list = []
    acc_dic = {}
    for key in dic.keys():
        file = dic[key]
        key_list.append(key)
        print_acc_batch(file)
        # print_acc_batch_save_dic(file, key)
        # print_acc(file)

    plot_batch(acc, key_list)
    # path = "/Users/zber/Documents/FastCNN/exp_result/loss/loss.json"
    # with open(path, 'w') as f:
    #     json.dump(acc_dic, f, indent=4)
