import json
import numpy as np
import matplotlib.pyplot as plt
import os
import glob


def str_to_float(str):
    num_list = []
    str = str[1:-1]
    str_arry = str.split(', ')
    for num in str_arry:
        num_list.append(float(num))
    # return np.asarray(num_list)
    return num_list


def grow_x_y_exist(dic):
    global mode
    global stage
    y = []
    for e in range(1, dic["epoch_to"]):
        for n in range(1, (num_batches // dic["num_batch"]) + 1):
            batches = []
            for b in range(1, dic["num_batch"] + 1):
                if mode == 'VG' and (b * n - 1) == 0:
                    continue

                if mode == 'ACV':
                    s = data[str(e)][str(b * n - 1)][dic["layer"]][mode][stage]
                    f = str_to_float(s)
                elif mode == 'W':
                    s = data[str(e)][str(b * n - 1)][dic["layer"]][mode]['out'][stage]
                    f = str_to_float(s)
                # elif mode == 'W':
                #     s = data[str(e)][str(b * n - 1)][dic["layer"]]['S']["out"]
                #     a = np.asarray(str_to_float(s))
                #     w = data[str(e)][str(b * n - 1)][dic["layer"]]['L1']["out"]
                #     b = np.asarray(str_to_float(w))
                #     # f = np.asarray(s_f) / np.asarray(w_f)
                #     f = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
                elif mode == 'Delta_S':
                    s = data[str(e)][str(b * n - 1)][dic["layer"]][mode]
                    if not s:
                        continue
                    f = str_to_float(s)
                else:
                    s = data[str(e)][str(b * n - 1)][dic["layer"]][mode]["out"]
                    f = str_to_float(s)

                start = 0
                end = dic["grow_base_size"][dic["layer"]]
                f_need = f[start:end]
                f_mean = np.nanmean(f_need)
                batches.append(f_mean)
            y.append(np.nanmean(batches))
    length = len(y)
    x = np.arange(0, length).tolist()
    return x, y


def grow_x_y(dic, gi=1):
    global mode
    global stage
    y = []
    for e in range(dic["epoch_grow"][gi], dic["epoch_to"]):
        for n in range(1, (num_batches // dic["num_batch"]) + 1):
            batches = []
            for b in range(1, dic["num_batch"] + 1):
                if mode == 'VG' and (b * n - 1) == 0:
                    continue
                if mode == 'ACV':
                    s = data[str(e)][str(b * n - 1)][dic["layer"]][mode][stage]
                    f = str_to_float(s)
                elif mode == 'W':
                    s = data[str(e)][str(b * n - 1)][dic["layer"]][mode]['out'][stage]
                    f = str_to_float(s)
                elif mode == 'Delta_S':
                    s = data[str(e)][str(b * n - 1)][dic["layer"]][mode]
                    if not s:
                        continue
                    f = str_to_float(s)
                # elif mode == 'W':
                #     s = data[str(e)][str(b * n - 1)][dic["layer"]]['S']["out"]
                #     a = np.asarray(str_to_float(s))
                #     w = data[str(e)][str(b * n - 1)][dic["layer"]]['L1']["out"]
                #     b = np.asarray(str_to_float(w))
                #     # f = np.asarray(s_f) / np.asarray(w_f)
                #     f = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
                else:
                    s = data[str(e)][str(b * n - 1)][dic["layer"]][mode]["out"]
                    f = str_to_float(s)

                start = dic["grow_base_size"][dic["layer"]] + (gi - 1) * dic["grow_size"][dic["layer"]]
                end = start + dic["grow_size"][dic["layer"]]
                f_need = f[start:end]
                f_mean = np.nanmean(f_need)
                batches.append(f_mean)
            y.append(np.nanmean(batches))
    length = len(y)
    start_x = (dic["epoch_grow"][gi] - 1) * (num_batches // dic["num_batch"])
    x = np.arange(start_x, start_x + length).tolist()
    return x, y


def grow_x_y_exist_in(dic):
    global mode
    y = []
    for e in range(1, dic["epoch_to"]):
        for n in range(1, (num_batches // dic["num_batch"]) + 1):
            batches = []
            for b in range(1, dic["num_batch"] + 1):
                if mode == 'VG' and (b * n - 1) == 0:
                    continue
                s = data[str(e)][str(b * n - 1)][dic["layer"]][mode]["in"]
                f = str_to_float(s)
                start = 0
                end = dic["grow_base_size"][dic["layer"]]
                f_need = f[start:end]
                f_mean = np.nanmean(f_need)
                batches.append(f_mean)
            y.append(np.nanmean(batches))
    length = len(y)
    x = np.arange(0, length)
    return x, y


def grow_x_y_in(dic, gi=1):
    global mode
    y = []
    for e in range(dic["epoch_grow"][gi], dic["epoch_to"]):
        for n in range(1, (num_batches // dic["num_batch"]) + 1):
            batches = []
            for b in range(1, dic["num_batch"] + 1):
                if mode == 'VG' and (b * n - 1) == 0:
                    continue
                s = data[str(e)][str(b * n - 1)][dic["layer"]][mode]["in"]
                f = str_to_float(s)
                start = dic["grow_base_size"][dic["layer"]] + (gi - 1) * dic["grow_size"][dic["layer"]]
                end = start + dic["grow_size"][dic["layer"]]
                f_need = f[start:end]
                if not f_need:
                    print()
                f_mean = np.nanmean(f_need)
                batches.append(f_mean)
            y.append(np.nanmean(batches))
    length = len(y)
    start_x = (dic["epoch_grow"][gi] - 1) * (num_batches // dic["num_batch"])
    x = np.arange(start_x, start_x + length)
    return x, y


def plot_save(x_y, dir, layer='0', is_out=False):
    grow_epochs = [1, 3, 6, 9]
    # grow_epochs = [1, 1.25, 2, 3, 4, 5]
    global mode
    global stage
    label_list = ['Exsiting', 'Grow_1', 'Grow_2', 'Grow_3', 'Grow_4']
    colour_list = ['r--', 'b', 'g', 'yellow', 'black']
    # ax.plot(x, mean, label='C8_mean')
    # ax.plot(x, mean1, label='C12_mean')
    # ax.plot(x, mean2, label='C16_mean')
    # ax.plot(x, mean3, label='C20_mean')
    # ax.plot(x, mean4, label='C24_mean')
    # ax.legend()
    path = os.path.join(dir, '{}{}_L{}_{}.png'.format(mode, stage, layer, ('out' if is_out else 'in')))
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 8))
    for xy, colour, label, in zip(x_y, colour_list, label_list):
        ax.plot(xy[0], xy[1], colour, label=label)
    for i in grow_epochs:
        point_x = i * (num_batches / 50) - 1
        plt.axvline(x=point_x, color='grey', linestyle='--', alpha=0.5)
    ax.legend()
    fig.savefig(path)
    plt.close(fig)


def gen_x_y(is_out=True):
    x_y = []
    if is_out:
        for gi in range(num_growth):
            if gi == 0:
                x_y.append(grow_x_y_exist(dic_layer_out))
            else:
                x_y.append(grow_x_y(dic_layer_out, gi=gi))
    else:
        for gi in range(num_growth):
            if gi == 0:
                x_y.append(grow_x_y_exist_in(dic_layer_in))
            else:
                x_y.append(grow_x_y_in(dic_layer_in, gi=gi))

    # modification
    # old = np.asarray(x_y[0][1])
    # old = np.where(old > 0, 0 , old)
    # new = np.asarray(x_y[1][1])
    # new = np.where(new < 0 , 0, new)
    # x_y[0] = (x_y[0][0], old)
    # x_y[1] = (x_y[1][0], new)

    return x_y


if __name__ == '__main__':
    dic = dict(
        # rank_baseline="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_baseline_6/layer1_log.txt",
        # rank_baseline_gate="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_baseline_gate/layer1_log.txt",
        # rank_baseline_LR3="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_baseline_LR3_20200412-233035/layer2_log.txt",
        # rank_baseline_LR4="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_baseline_LR4_20200413-001813/layer2_log.txt",
        # copy_n="/Users/zber/ProgramDev/exp_pyTorch/results/grow_copy_n_6/layer1_log.txt",
        # copy_one="/Users/zber/ProgramDev/exp_pyTorch/results/grow_copy_one_6/layer1_log.txt",
        # rank_ours="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_ours_11/layer1_log.txt",
        # bridging="/Users/zber/ProgramDev/exp_pyTorch/results/grow_bridging_6/layer1_log.txt",
        # bridging2="/Users/zber/ProgramDev/exp_pyTorch/results/grow_bridging_7/layer1_log.txt",
        # rank_cumulative_cosine="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_cosine_1/layer1_log.txt",
        # rank_cumulative_cosine_gate="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_cosine_gate/layer1_log.txt",
        # rank_cumulative_cosine_LR="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_cosine_LR_20200412-214542/layer1_log.txt",
        # random="/Users/zber/ProgramDev/exp_pyTorch/results/grow_random_1/layer1_log.txt",
        # random_gate="/Users/zber/ProgramDev/exp_pyTorch/results/grow_random_gate/layer1_log.txt",
        # random_LR="/Users/zber/ProgramDev/exp_pyTorch/results/grow_random_LR_20200412-214020/layer1_log.txt",
        # ranklow_n="/Users/zber/ProgramDev/exp_pyTorch/results/grow_ranklow_n_20200408-232641/layer1_log.txt",
        # activation_low="/Users/zber/ProgramDev/exp_pyTorch/results/grow_activation_low_20200409-165613/layer1_log.txt",
        # rank_cumulative_cosine_low="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_cosine_low_20200409-203904/layer1_log.txt",
        # rank_cumulative="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_1/layer1_log.txt",
        # rank_cumulative_low="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_low_1/layer1_log.txt",
        # rank_cumulative_gate="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_gate/layer1_log.txt",
        # rank_cumulative_gate_max_test="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_gate_max_test/layer2_log.txt",
        # standard="/Users/zber/ProgramDev/exp_pyTorch/results/grow_standard_3/grow_standard__json_in_out.json",
        # standard_s="/Users/zber/ProgramDev/exp_pyTorch/results/grow_standard_s/grow_standard__json_dic.json",
        # rank_cumulative_cL_fH="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_convL_fcH/layer1_log.txt",
        # rank_cumulative_cL_fH2="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_convL_fcH2/layer1_log.txt",
        # rank_cumulative_H_normal="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_rank_20200415-161606/layer2_log.txt",
        # rank_cumulative_con1_fc2="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_con1_fc2_f/layer1_log.txt",
        # rank_cumulative_con2_fc1="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_con2_fc1_f/layer1_log.txt",
        # rank_cumulative_conR_fcR="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_conR_fcR/layer1_log.txt",
        # rank_cumulative_con2_fc2="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_con2_fc2/layer1_log.txt",
        # rank_cumulative_con2_fc1="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_con2_fc1/layer1_log.txt",
        # rank_cumulative_con1_fc2="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_con1_fc2/layer1_log.txt",
        # standard_seed="/Users/zber/ProgramDev/exp_pyTorch/results/grow_standard_seed/layer1_log.txt",
        # standard_seed_BN="/Users/zber/ProgramDev/exp_pyTorch/results/grow_standard_seed_BN/layer1_log.txt",
        # rank_cumulative_con2_fc2_in2="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_con2_fc2_in2/layer1_log.txt",
        # rank_cumulative_conR_fcR_in2="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_conR_fcR_in2/layer1_log.txt",
        # rank_cumulative_conN_fcN_in2="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_conN_fcN_in2/layer1_log.txt",
        #
        # rank_cumulative_conR_fcR_in2_out2="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_conR_fcR_in2_out2/layer1_log.txt",
        # rank_cumulative_conN_fcN_in2_out2="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_conN_fcN_in2_out2/layer1_log.txt",
        # rank_cumulative_con2_fc2_in2_out2="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_con2_fc2_in2_out2/layer1_log.txt",

        # rank_cumulative_conR_fcR_inR_ogR="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_conR_fcR_inR_ogR/layer1_log.txt",
        # rank_cumulative_conR_fcR_gateAuto_test="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_conR_fcR_gateAuto/layer1_log.txt",
        # rank_cumulative_con2R_fc2R_gateAuto_test = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_con2R_fc2R_gateAuto/layer1_log.txt",
        # rank_cumulative_con2R_fc2R_gateAuto300_test="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_con2R_fc2R_gateAuto300/layer1_log.txt",
        # rank_cumulative_conR_fcR="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_conR_fcR/layer1_log.txt",

        # rank_cumulative_con2R_fc2R_in2R="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_con2R_fc2R_in2R/layer1_log.txt",
        # rank_cumulative_con2R_fc2R_in2R_gateAutoMax_LR600_test="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_con2R_fc2R_in2R_gateAutoMax_LR600/layer1_log.txt",
        # rank_cumulative_con2R_fc2R_in2R_gateAutoMax_LR300_test="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_con2R_fc2R_in2R_gateAutoMax_LR300/layer1_log.txt",
        # rank_cumulative_con2R_fc2R_in2R_gateAutoMax300_test="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_con2R_fc2R_in2R_gateAutoMax300/layer1_log.txt",
        # rank_cumulative_con2R_fc2R_in2R_LR_600="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_con2R_fc2R_in2R_LR_600/layer1_log.txt",

        # rank_cumulative_con2R_fc2R_gateAutoMax_LR_300_2="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_con2R_fc2R_gateAutoMax_LR_300_2/layer1_log.txt",

        # rank_cumulative_conR_fc1="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_conR_fc1/layer1_log.txt",
        # rank_cumulative_con1_fcR="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_con1_fcR/layer1_log.txt",
        # rank_cumulative_conN_fcN="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_conN_fcN/layer1_log.txt",
        # rank_cumulative_conN_fc1="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_conN_fc1/layer1_log.txt",
        # rank_cumulative_con1_fcN="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_con1_fcN/layer1_log.txt",
        # rank_cumulative_conR_fcR_15="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_conR_fcR_15/layer1_log.txt",
        # rank_cumulative_con2_fc2_15="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_con2_fc2_15/layer1_log.txt",
        # rank_cumulative_og2_in2="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_og2_in2/layer1_log.txt",
        # rank_cumulative_low_normal="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_low_20200415-161006/layer1_log.txt",
        # rank_cumulative_conv_gate_fc_nochange_test="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_conv_gate_fc_nochange_test/layer1_log.txt",
        # rank_cumulative_conv_gateauto_fc_nochange_test="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_conv_gateauto_fc_nochange_test/layer1_log.txt",
        # rank_cumulative_conv_gate_fc_n_test="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_conv_gate_fc_n_test/layer1_log.txt",
        # rank_cumulative_conv_gate_fc_gate_n_test="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_conv_gate_fc_gate_n_test/layer1_log.txt",
        # rank_cumulative_conv_gate_fc_gate_test="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_conv_gate_fc_gate_test/layer1_log.txt",
        # rank_cumulative_conv_gate_fc_gate_ratio_test="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_conv_gate_fc_gate_ratio_test/layer1_log.txt",
        # rank_baseline="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_baseline_20200430-190631/layer1_log.txt",
        # random="/Users/zber/ProgramDev/exp_pyTorch/results/grow_random_20200430-194619/layer1_log.txt",
        # copy_n="/Users/zber/ProgramDev/exp_pyTorch/results/grow_copy_n_20200430-191941/layer1_log.txt",
        # bridging="/Users/zber/ProgramDev/exp_pyTorch/results/grow_bridging_20200430-193258/layer1_log.txt",
        # rank_cumulative="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_20200430-213135/layer1_log.txt",
        # rank_cumulative_cosine="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_cosine_20200430-214432/layer1_log.txt",
        # standard_10_17_50="/Users/zber/ProgramDev/exp_pyTorch/results/standard_10_17_50/layer1_log.txt",
        # standard_s_20_50_100="/Users/zber/ProgramDev/exp_pyTorch/results/grow_standard_20_50_500/layer1_log.txt",

        # 5-5
        # random0="/Users/zber/ProgramDev/exp_pyTorch/results/grow_random_0.1/layer1_log.txt",
        # random1="/Users/zber/ProgramDev/exp_pyTorch/results/grow_random_-1to1/layer1_log.txt",
        # standard_2_5_10="/Users/zber/ProgramDev/exp_pyTorch/results/standard_2_5_10/layer1_log.txt",
        # standard_4_8_20="/Users/zber/ProgramDev/exp_pyTorch/results/standard_4_8_20/layer1_log.txt",
        # standard_6_11_30="/Users/zber/ProgramDev/exp_pyTorch/results/standard_6_11_30/layer1_log.txt",
        # standard_8_14_40="/Users/zber/ProgramDev/exp_pyTorch/results/standard_8_14_40/layer1_log.txt",
        # standard_10_17_50_1 = "/Users/zber/ProgramDev/exp_pyTorch/results/standard_10_17_50_20200508-194338/layer1_log.txt",

        # 5-6
        # rank_ours_dynamic_grow="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_ours_dynamic_grow/layer1_log.txt"
        # rank_ours_BN="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_ours_BN_20200506-175156/layer1_log.txt",
        # rank_ours="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_ours_20200506-000800/layer1_log.txt",
        # rank_ours_normal="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_ours_max_normal_noise/layer1_log.txt",
        # rank_baseline_error = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_baseline_scale_20200507-174546/layer1_log.txt"
        # rank_ours_max_no_noise = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_ours_max_no_noise2/layer1_log.txt",
        # rank_ours_max_uniform_noi ="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_ours_max_uniform_noi/layer1_log.txt",
        # rank_ours_avg = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_ours_avg/layer1_log.txt",
        # rank_baseline_scale="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_baseline_scale_20200508-000823/layer1_log.txt"

        # 5-8
        # standard_10_17_50="/Users/zber/ProgramDev/exp_pyTorch/results/standard_10_17_50/layer1_log.txt",
        # standard_reg_10_17_50="/Users/zber/ProgramDev/exp_pyTorch/results/standard_10_17_50_reg1/layer1_log.txt",
        # bridging="/Users/zber/ProgramDev/exp_pyTorch/results/grow_bridging_20200508-223706/layer1_log.txt",
        # rank_baseline="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_baseline_20200508-224355/layer1_log.txt",
        # random="/Users/zber/ProgramDev/exp_pyTorch/results/grow_random_20200508-225047/layer1_log.txt",
        # copy_n="/Users/zber/ProgramDev/exp_pyTorch/results/grow_copy_n_20200508-225737/layer1_log.txt",
        # rank_ours="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_ours_noise_bias_scale/layer1_log.txt",
        # rank_ours_low="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_ours_noise_bias_scale_low/layer1_log.txt",
        # random_reg1="/Users/zber/ProgramDev/exp_pyTorch/results/grow_random_reg0.1/layer1_log.txt",
        # random_reg2="/Users/zber/ProgramDev/exp_pyTorch/results/grow_random_reg0.01/layer1_log.txt",
        # random_reg3="/Users/zber/ProgramDev/exp_pyTorch/results/grow_random_reg0.001/layer1_log.txt",
        # random_reg4="/Users/zber/ProgramDev/exp_pyTorch/results/grow_random_reg0.0001/layer1_log.txt",
        # random_reg5="/Users/zber/ProgramDev/exp_pyTorch/results/grow_random_reg0.00001/layer1_log.txt",
        # random_gate_10_05_test="/Users/zber/ProgramDev/exp_pyTorch/results/grow_random_gate_10_05/layer1_log.txt",
        # random_gate_15_10_test="/Users/zber/ProgramDev/exp_pyTorch/results/grow_random_gate_15-10/layer1_log.txt",

        # 5-10
        # rank_ours_low="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_ours_low/layer1_log.txt",
        # rank_ours_vg_low="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_ours_vg_low/layer1_log.txt",
        # standard_10_17_50="/Users/zber/ProgramDev/exp_pyTorch/results/standard_10_17_50_20200510-201607/layer1_log.txt",
        # rank_ours_low="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_ours_low/layer1_log.txt",
        # rank_ours_vg_low="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_ours_vg_low/layer1_log.txt",
        # standard_10_17_50="/Users/zber/ProgramDev/exp_pyTorch/results/standard_10_17_50_20200510-201607/layer1_log.txt",
        # low_cosine="/Users/zber/ProgramDev/exp_pyTorch/results/grow_low_cosine_20200511-000208/layer1_log.txt",

        # 5-11
        # low_cosine_e20="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_20e_20200511-041701/layer1_log.txt",
        # rank_low_e20="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_ours_20e_20200511-043252/layer1_log.txt",
        # random_e20="/Users/zber/ProgramDev/exp_pyTorch/results/grow_random_20e_20200511-040126/layer1_log.txt",
        # copy_n_e20="/Users/zber/ProgramDev/exp_pyTorch/results/grow_copy_n_e20_20200512-004939/layer1_log.txt",
        # rank_baseline_e20="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_baseline_e20_20200512-010604/layer1_log.txt",
        # bridging_e20="/Users/zber/ProgramDev/exp_pyTorch/results/grow_bridging_e20_20200512-003340/layer1_log.txt",
        # rank_low_overP="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_ours_overP_20200511-012542/layer1_log.txt",
        # low_cosine_overP="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_overP_20200511-010217/layer1_log.txt",

        # low_cosine_LR_05="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_LR_05_20200511-143658/layer1_log.txt",
        # low_cosine_LR_2="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_LR_2_20200511-142440/layer1_log.txt",
        # low_cosine_LR_10="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_LR_10_20200511-182123/layer1_log.txt",
        # low_cosine_LR_20="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_LR20/layer1_log.txt",
        # low_cosine="/Users/zber/ProgramDev/exp_pyTorch/results/grow_low_cosine_20200511-000208/layer1_log.txt",
        # low_cosine_LR_01 = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_LR_01_20200511-183026/layer1_log.txt",
        # low_cosine_LR_10_1 = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_LR10_1/layer1_log.txt",
        # low_cosine_LR_001="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_LR001/layer1_log.txt",

        # 5-12
        # low_cosine_wd="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_20200512-171940/layer1_log.txt",
        # low_cosine_wd="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_wd_S100/layer1_log.txt",
        # low_cosine_fwd="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_Fwd_S100/layer1_log.txt",
        # low_cosine_LR10="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_LR10_20200512-200458/layer1_log.txt",
        # low_cosine_LR20="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_LR20_20200512-192221/layer1_log.txt",

        # 5-13
        # standard_2_5_10 = "/Users/zber/ProgramDev/exp_pyTorch/results/standard_2_5_10_acv_20200513-002716/layer1_log.txt",
        # standard_4_8_20 = "/Users/zber/ProgramDev/exp_pyTorch/results/standard_4_8_20_acv_20200513-003157/layer1_log.txt",
        # standard_6_11_30 = "/Users/zber/ProgramDev/exp_pyTorch/results/standard_6_11_30_acv_20200513-003718/layer1_log.txt",
        # standard_8_14_40 = "/Users/zber/ProgramDev/exp_pyTorch/results/standard_8_14_40_acv_20200513-004320/layer1_log.txt",
        # standard_10_17_50 = "/Users/zber/ProgramDev/exp_pyTorch/results/standard_10_17_50_acv_20200513-005003/layer1_log.txt",
        # rank_cosine_wd_after = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_wd_after_20200513-012936/layer1_log.txt",
        # rank_cosine_wd_before = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_wd_before_20200513-014531/layer1_log.txt",

        # standard_2_5_10="/Users/zber/ProgramDev/exp_pyTorch/results/standard_2_5_10_20200513-171839/layer1_log.txt",
        # standard_4_8_20="/Users/zber/ProgramDev/exp_pyTorch/results/standard_4_10_20_20200513-172347/layer1_log.txt",
        # standard_4_8_20="/Users/zber/ProgramDev/exp_pyTorch/results/standard_4_8_20_20200513-200616/layer1_log.txt",
        # standard_6_11_30="/Users/zber/ProgramDev/exp_pyTorch/results/standard_6_11_30_20200513-172946/layer1_log.txt",
        # standard_8_14_40="/Users/zber/ProgramDev/exp_pyTorch/results/standard_8_14_40_20200513-173607/layer1_log.txt",
        # standard_10_17_50="/Users/zber/ProgramDev/exp_pyTorch/results/standard_10_17_50_20200513-174306/layer1_log.txt",
        # standard_10_17_50="/Users/zber/ProgramDev/exp_pyTorch/results/standard_10_17_50_20200513-212337/layer1_log.txt",
        # low_cosine_wd_before = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_wd_before_20200513-175200/layer1_log.txt",
        # low_cosine_wd_after = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_wd_after_20200513-180622/layer1_log.txt",

        # low_cosine_LR10 = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_LR10_20200513-182251/layer1_log.txt",
        # low_cosine_LR20 = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_LR20_20200513-183920/layer1_log.txt",
        # low_cosine="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_20200513-190352/layer1_log.txt",

        # standard_2_5_10="/Users/zber/ProgramDev/exp_pyTorch/results/standard_2_5_10_20200513-214705/layer1_log.txt",
        # standard_4_8_20="/Users/zber/ProgramDev/exp_pyTorch/results/standard_4_8_20_20200513-215217/layer1_log.txt",
        # standard_6_11_30="/Users/zber/ProgramDev/exp_pyTorch/results/standard_6_11_30_20200513-215804/layer1_log.txt",
        # standard_8_14_40="/Users/zber/ProgramDev/exp_pyTorch/results/standard_8_14_40_20200513-220428/layer1_log.txt",
        # standard_10_17_50="/Users/zber/ProgramDev/exp_pyTorch/results/standard_10_17_50_20200513-221254/layer1_log.txt",

        # bridging="/Users/zber/ProgramDev/exp_pyTorch/results/grow_bridging_20200513-222520/layer1_log.txt",
        # rank_baseline="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_baseline_20200513-223311/layer1_log.txt",
        # random="/Users/zber/ProgramDev/exp_pyTorch/results/grow_random_20200513-224104/layer1_log.txt",
        # copy_n="/Users/zber/ProgramDev/exp_pyTorch/results/grow_copy_n_20200513-224902/layer1_log.txt",
        # low_cosine="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_20200513-225658/layer1_log.txt",

        # 5-14
        # low_cosine_LR10="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_LR10_20200514-125623/layer1_log.txt",
        # low_cosine_LR20="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_LR20_20200514-130842/layer1_log.txt",
        # low_cosine_wd_after = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_wd_after_20200514-121238/layer1_log.txt",
        # low_cosine_wd_before = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_wd_before_20200514-122705/layer1_log.txt",

        # standard_2_5_10="/Users/zber/ProgramDev/exp_pyTorch/results/standard_2_5_10_20200514-192251/layer1_log.txt",
        # standard_4_8_20="/Users/zber/ProgramDev/exp_pyTorch/results/standard_4_8_20_20200514-192745/layer1_log.txt",
        # standard_6_11_30="/Users/zber/ProgramDev/exp_pyTorch/results/standard_6_11_30_20200514-193258/layer1_log.txt",
        # standard_8_14_40="/Users/zber/ProgramDev/exp_pyTorch/results/standard_8_14_40_20200514-193833/layer1_log.txt",
        # standard_10_17_50="/Users/zber/ProgramDev/exp_pyTorch/results/standard_10_17_50_20200514-194428/layer1_log.txt",
        #
        # rank_baseline="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_baseline_20200514-195931/layer1_log.txt",
        # random="/Users/zber/ProgramDev/exp_pyTorch/results/grow_random_20200514-200640/layer1_log.txt",
        # copy_n="/Users/zber/ProgramDev/exp_pyTorch/results/grow_copy_n_20200514-201348/layer1_log.txt",
        # bridging="/Users/zber/ProgramDev/exp_pyTorch/results/grow_bridging_20200514-195225/layer1_log.txt",
        # low_cosine="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_20200514-171924/layer1_log.txt",
        #
        # low_cosine_LR10="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_LR10_20200514-182026/layer1_log.txt",
        # low_cosine_LR20="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_LR20_20200514-182938/layer1_log.txt",
        # low_cosine_wd="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_wd_20200514-185357/layer1_log.txt",

        # low_cosine_e3 = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_20200514-205126/layer1_log.txt",
        # low_cosine_wd1 = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_20200514-222648/layer1_log.txt"
        # low_cosine_no_hook = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_no_hook/layer1_log.txt",
        # low_cosine_no_hook_e10= "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_no_hook_e10/layer1_log.txt"
        # low_cosine_hook_e10 = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_hook_e10/layer1_log.txt",

        # overP
        # bridging_overP="/Users/zber/ProgramDev/exp_pyTorch/results/grow_bridging_overP_20200514-015356/layer1_log.txt",
        # baseline_overP="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_baseline_overP_20200514-021729/layer1_log.txt",
        # random_overP="/Users/zber/ProgramDev/exp_pyTorch/results/grow_random_overP_20200514-024035/layer1_log.txt",
        # copy_n_overP="/Users/zber/ProgramDev/exp_pyTorch/results/grow_copy_n_overP_20200514-030402/layer1_log.txt",
        # low_cosine_overP="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_overP_20200514-032731/layer1_log.txt",
        # standard_20_50_500="/Users/zber/ProgramDev/exp_pyTorch/results/standard_20_50_500_20200514-010144/layer1_log.txt",

        # LR Scheduler
        # baseline_LRS="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_baseline_LRS_20200514-214327/layer1_log.txt"
        # standard_10_17_50_LRS="/Users/zber/ProgramDev/exp_pyTorch/results/standard_10_17_50_LRS_20200515-165047/layer1_log.txt",
        # baseline_LRS = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_baseline_LRS_20200515-181608/layer1_log.txt",
        # baseline_LRS_overP = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_baseline_LRS_overP_20200515-192916/layer1_log.txt",

        # LR
        # baseline_LR2="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_baseline_20200515-005044/layer1_log.txt",
        # rank_baseline_LR4 = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_baseline_LR4_20200515-163136/layer1_log.txt",
        # low_cosine_Sdecay_300 = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_Sdecay_300_20200515-183221/layer1_log.txt",
        # low_cosine_Sdecay_600 = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_Sdecay_600_20200515-172914/layer1_log.txt",

        # standard_2_5_10="/Users/zber/ProgramDev/exp_pyTorch/results/standard_2_5_10_20200515-144953/layer1_log.txt",
        # standard_4_8_20="/Users/zber/ProgramDev/exp_pyTorch/results/standard_4_8_20_20200515-145445/layer1_log.txt",
        # standard_6_11_30="/Users/zber/ProgramDev/exp_pyTorch/results/standard_6_11_30_20200515-145958/layer1_log.txt",
        # standard_8_14_40="/Users/zber/ProgramDev/exp_pyTorch/results/standard_8_14_40_20200515-150543/layer1_log.txt",
        # standard_10_17_50="/Users/zber/ProgramDev/exp_pyTorch/results/standard_10_17_50_20200515-151155/layer1_log.txt",

        # low_cosine_converge = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_Converge_E2_20200515-215555/layer1_log.txt",
        # low_cosine_converge_before = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_Converge_E2_before_20200515-220914/layer1_log.txt",
        # low_cosine_converge_e3="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_20200515-222941/layer1_log.txt",
        # low_cosine_converge_e5="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_20200515-224212/layer1_log.txt",
        # low_cosine_asy_converge_e5 = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_Asy_converge_E5_20200515-230947/layer1_log.txt",
        # low_cosine_asy_reverse_converge_e5 = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_Asy_reverse_converge_E5_20200515-232619/layer1_log.txt",
        # low_cosine_asy_reverse_converge_e5 = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_Asy_revers_converge_E5_150_20200515-233521/layer1_log.txt",
        # low_cosine_asy_converge_e5 = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_Asy_converge_E5_150_20200515-234617/layer1_log.txt",
        # low_cosine_converge_e5_LR="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_converge_E5_150_LR_20200516-000544/layer1_log.txt",

        # 5-16
        # low_cosine_max_select_converge = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_max_select_converge_E5_150_20200516-150254/layer1_log.txt",
        # low_cosine_select_converge = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_select_converge_E5_150_20200516-145223/layer1_log.txt",
        # low_cosine_max_select_converge_LR2_= "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_max_select_converge_E5_150_LR2_20200516-154034/layer1_log.txt",
        # low_cosine_max_select_converge_LR4 = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_max_select_converge_E5_150_LR4_20200516-153312/layer1_log.txt",

        # cosine_max_select_converge_E5_150_LR4_decay_S3="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_max_select_converge_E5_150_LR4_decay_S3_20200516-161734/layer1_log.txt",
        # cosine_max_select_converge_E5_150_LR2_decay_S3="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_max_select_converge_E5_150_LR2_decay_S3_20200516-161247/layer1_log.txt",
        # cosine_max_select_converge_E5_150_LR2_decay="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_max_select_converge_E5_150_LR2_decay_20200516-160510/layer1_log.txt",
        # cosine_max_select_converge_E5_150_LR4_decay="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_max_select_converge_E5_150_LR4_decay_20200516-160142/layer1_log.txt",
        #
        # cosine_max_select_converge_E5_150_LR2_decay_S3="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_max_select_converge_E5_150_LR2_decay_S3_20200516-172456/layer1_log.txt",
        # # cosine_max_select_converge_E5_150_LR4_decay_S3="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_max_select_converge_E5_150_LR4_decay_S3_20200516-171602/layer1_log.txt",
        # cosine_max_select_converge_LR2_decay_lambda="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_max_select_converge_E5_150_LR2_decay_Lambda_20200516-182642/layer1_log.txt",
        # cosine_max_select_converge_LR4_decay_lambda="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_max_select_converge_E5_150_LR4_decay_Lambda_20200516-182202/layer1_log.txt",
        # renew_optimizer = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_max_select_converge_E5_150_LR2_decay_Lambda_renew_optimizer_20200516-185327/layer1_log.txt",
        # optimizer_new_decay = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_20200516-191250/layer1_log.txt",

        # cosine_max_select_converge_copy_optimizer_EXdecay_LR2="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_max_select_converge_copy_optimizer_EXdecay_LR2_20200516-202511/layer1_log.txt",
        # cosine_max_select_converge_renew_optimier_LR2="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_max_select_converge_renew_optimier_LR2_20200516-205802/layer1_log.txt",
        # cosine_max_select_converge_copy_optimizer_LR2="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_max_select_converge_copy_optimizer_LR2_20200516-204614/layer1_log.txt",
        # converge_copy_optimizer="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_max_select_converge_copy_optimizer_20200516-214928/layer1_log.txt",
        # converge_renew_optimizer="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_max_select_converge_renew_optimizer_20200516-214022/layer1_log.txt",
        # converge_copy_optimizer = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_max_select_converge_copy_optimizer_20200516-220235/layer1_log.txt",

        # baseline_renew_optimizer="/Users/zber/ProgramDev/exp_pyTorch/results/renew_optimizer_grow_rank_baseline_20200516-213829/layer1_log.txt",
        # random_renew_optimizer="/Users/zber/ProgramDev/exp_pyTorch/results/renew_optimizer_grow_random_20200516-214520/layer1_log.txt",
        # copy_renew_optimizer="/Users/zber/ProgramDev/exp_pyTorch/results/renew_optimizer_grow_copy_n_20200516-215209/layer1_log.txt",
        # cosine_renew_optimizer="/Users/zber/ProgramDev/exp_pyTorch/results/renew_optimizer_grow_rank_cosine_20200516-215857/layer1_log.txt",
        # bridging_renew_optimizer="/Users/zber/ProgramDev/exp_pyTorch/results/renew_optimizer_grow_bridging_20200516-213142/layer1_log.txt",

        # weight decay, lambda
        # cosine_weight_decay_lambda="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_weight_decay_lambda_20200516-222256/layer1_log.txt",
        # cosine_weight_decay="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_weight_decay_20200516-221613/layer1_log.txt",

        # cosine_wd_all_lambda_hidden="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_wd_all_lambda_hidden_20200516-231143/layer1_log.txt",

        #
        # cosine_rank="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_20200517-003009/layer1_log.txt",
        # cosine_rank="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_20200517-011419/layer1_log.txt",
        # cosine_5_1 = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_5_1_20200517-012729/layer1_log.txt",
        # cosine_converge_LR2decay_OPrenew = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_converge_LR2decay_OPrenew/layer1_log.txt",
        # cosine_converge_LR2decay_OPcopy="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_converge_LR2decay_OPcopy/layer1_log.txt",
        # cosine_converge_LR2decay_OPavg = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_converge_LR2decay_OPavg/layer1_log.txt",
        # cosine_converge_LR2decay_OPscaleAVG = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_converge_LR2decay_OPscaleAVG_20200517-155545/layer1_log.txt",
        # cosine_converge_LR2decay_OPscale = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_converge_LR2decay_OPscale_20200517-161400/layer1_log.txt",
        # cosine_converge_LR2decay_LMDdecay_OPcopy="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_converge_LR2decay_LMDdecay_OPcopy_20200517-164314/layer1_log.txt",
        # cosine_converge_LR2decay_LMDdecay_OPcopy_E2= "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_converge_LR2decay_LMDdecay_OPcopy_E2_S4_8_20_20200517-171035/layer1_log.txt",
        # cosine_converge_LR2decay_LMDdecay_OPcopy_E5= "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_converge_LR2decay_LMDdecay_OPcopy_E5_S4_8_20_20200517-172649/layer1_log.txt",
        # cosine_converge_LR2decay_LMDdecay_OPrenew="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_converge_LR2decay_LMDdecay_OPrenew_20200517-175337/layer1_log.txt",
        # cosine_converge_LMDnew_OPcopy="/Users/zber/ProgramDev/exp_pyTorch/results/cosine_converge_LMDnew_OPcopy_20200517-183607/layer1_log.txt"
        # cosine_converge_LMDnewDecay_OPcopy = "/Users/zber/ProgramDev/exp_pyTorch/results/cosine_converge_LMDnewDecay_OPcopy_20200517-190458/layer1_log.txt",

        # cosine_converge_LR2decay_LMDdecayAsy_OPrenew="/Users/zber/ProgramDev/exp_pyTorch/results/cosine_converge_LR2decay_LMDdecayAsy_OPrenew_20200517-215549/layer1_log.txt",
        # cosine_converge_LR2decay_LMDdecayAsyR_OPrenew="/Users/zber/ProgramDev/exp_pyTorch/results/cosine_converge_LR2decay_LMDdecayAsyR_OPrenew_20200517-214313/layer1_log.txt",
        # cosine_converge_LR2decay_LMDdecayAsyR_OPrenew_E10="/Users/zber/ProgramDev/exp_pyTorch/results/cosine_converge_LR2decay_LMDdecayAsyR_OPrenew_E10_20200517-204159/layer1_log.txt",
        # cosine_converge_LR2decay_LMDdecayAsy_OPrenew_E10="/Users/zber/ProgramDev/exp_pyTorch/results/cosine_converge_LR2decay_LMDdecayAsy_OPrenew_E10_20200517-175337/layer1_log.txt",
        # cosine_converge_LR2decay_LMDdecayAsy_OPrenew_E10 = "/Users/zber/ProgramDev/exp_pyTorch/results/cosine_converge_LR2decay_LMDdecayAsy_OPrenew_E10_20200517-223242/layer1_log.txt",
        # cosine_bad="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_20200517-225847/layer1_log.txt",

        # 5-18
        # good_cosine = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_20200518-151420/layer1_log.txt",
        # cosine_Asy_Xavier="/Users/zber/ProgramDev/exp_pyTorch/results/cosine_Asy_Xavier_20200518-204628/layer1_log.txt",
        # cosine_AsyR_Xavier="/Users/zber/ProgramDev/exp_pyTorch/results/cosine_AsyR_Xavier_20200518-190439/layer1_log.txt",
        # cosine_RI = "/Users/zber/ProgramDev/exp_pyTorch/results/cosine_RI_20200517-225847/layer1_log.txt",
        # cosine_RI_600 = "/Users/zber/ProgramDev/exp_pyTorch/results/cosine_RI_step600_20200518-222843/layer1_log.txt",
        # cosine_RI_600="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_20200519-002735/layer1_log.txt",
        # cosine_RI="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_20200519-015236/layer1_log.txt",
        # cosine_RI="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_20200519-020328/layer1_log.txt",
        # cosine_RI_600="/Users/zber/ProgramDev/exp_pyTorch/results/cosine_RI_step600_20200518-222843/layer1_log.txt",

        # consine_RI_step600="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_RI_step600_Nonoise_20200519-173020/layer1_log.txt",
        # cosine_RI_step600_noNoise = "/Users/zber/ProgramDev/exp_pyTorch/results/cosine_RI_step600_Nonoise_20200519-175432/layer1_log.txt",
        # standard_10_17_50_Xavier="/Users/zber/ProgramDev/exp_pyTorch/results/standard_10_17_50_Xavier_20200519-171213/layer1_log.txt",

        # 5-19
        # cosine_RI_bias_scale = "/Users/zber/ProgramDev/exp_pyTorch/results/cosine_RI_bias_scale_20200519-194122/layer1_log.txt",
        # cosine_RI_T10 = "/Users/zber/ProgramDev/exp_pyTorch/results/cosine_RI_T-10_20200519-192937/layer1_log.txt",
        # cosine_RI_T = "/Users/zber/ProgramDev/exp_pyTorch/results/cosine_RI_T_20200519-204602/layer1_log.txt",
        # cosine_RI="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_20200519-212733/layer1_log.txt",
        # cosine_RI_back_T10="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_lambda_back_T10_20200519-214351/layer1_log.txt",
        # cosine_RI_T10="/Users/zber/ProgramDev/exp_pyTorch/results/cosine_RI_back_T10_20200519-215807/layer1_log.txt",
        # cosine_lambda_back_T3 = "/Users/zber/ProgramDev/exp_pyTorch/results/cosine_lambda_back_T3_20200519-224154/layer1_log.txt",
        # cosine_lambda05 = "/Users/zber/ProgramDev/exp_pyTorch/results/cosine_lambda05_20200519-230503/layer1_log.txt",

        # 5-20
        # cosine_RI_std="/Users/zber/ProgramDev/exp_pyTorch/results/cosine_RI_Sstd_20200520-000613/layer1_log.txt",
        # cosine_RI_std="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_20200520-174858/layer1_log.txt",
        # cosine_RI_std1="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_20200520-184702/layer1_log.txt"
        # cosine_ri="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_20200520-212325/layer1_log.txt",
        # cosine_ri="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_20200520-210135/layer1_log.txt",
        # cosine_ri="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_20200522-004826/layer1_log.txt",
        # cosine_RI_gradGate_lambdaR="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_20200522-174535/layer1_log.txt",
        # consine_RI_Sstd_weightNew_power="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_20200522-182027/layer1_log.txt",

        # cosine_converge = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_20200522-185207/layer1_log.txt"
        # cosine_momentum = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_20200522-220635/layer1_log.txt",
        # cosine_momentum_weightBack = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_20200522-233322/layer1_log.txt",
        # cosine_momentum = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_20200522-233322/layer1_log.txt",
        # cosine="/Users/zber/ProgramDev/exp_pyTorch/results/grow_ours20200525-185725/grow_rank_cosine__json_in_out.json"，
        # cosine="/Users/zber/ProgramDev/exp_pyTorch/results/grow_ours20200526-223945/layer1_log.txt",
        # new="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_20200528-175713/layer1_log.txt",
        # old="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_20200528-170111/layer1_log.txt",

        #
        # new_grad_step = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_20200530-010944/layer1_log.txt"

        #
        # one_lambda = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_20200530-160804/layer1_log.txt"
        # cosine_L_1_index="/Users/zber/ProgramDev/exp_pyTorch/results/cosine_L_1_index_20200530-192920/layer1_log.txt",
        # cosine_L_L_index="/Users/zber/ProgramDev/exp_pyTorch/results/cosine_L_L_index_20200530-194103/layer1_log.txt",
        # cosine_L_1L_index="/Users/zber/ProgramDev/exp_pyTorch/results/cosine_L_1L_index_20200530-195356/layer1_log.txt",

        # bridging="/Users/zber/ProgramDev/exp_pyTorch/results/grow_bridging_20200508-223706/layer1_log.txt",
        # cosine="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine20200605-200734/layer1_log.txt"
        # rank_bn="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_baseline_BN_20200618-223900/layer1_log.txt"
        # mb_har="/Users/zber/ProgramDev/exp_pyTorch/results/G_Har_MobileNet20200620-174316/analysis_copy.json",
        # mb_har_bn = "/Users/zber/ProgramDev/exp_pyTorch/results/G_Har_MobileNetBN_20200620-192435/analysis_copy.json",
        # standard_mb_har = "/Users/zber/ProgramDev/exp_pyTorch/results/Standard_Har_MobileNet_20200620-174936/analysis_copy.json",
        # mb_har_grow2 = "/Users/zber/ProgramDev/exp_pyTorch/results/G_Har_MobileNet_20200620-180636/analysis_copy.json"
        # standard_mbbn_har = "/Users/zber/ProgramDev/exp_pyTorch/results/Standard_Har_MobileNetBN_20200620-181236/analysis_copy.json",
        new = "/Users/zber/ProgramDev/exp_pyTorch/results/Standard_MNIST_LeNet_20200630-003814/analysis_copy.json"

    )

    # mobiel net
    # net = 'MobileNet'
    # num_batches = 230

    # Lenet
    net = 'LeNet'
    num_batches = 600

    mode = 'S'
    # mode = "Delta_S"
    if mode == 'ACV':
        stage = 'L1'
    elif mode == 'W':
        stage = 'L1'
    else:
        stage = ''
    num_growth = 5

    if net == "MobileNet":
        dic_layer_out = {
            "epoch_from": 1,
            "layer": "1",
            "epoch_to": 11,
            "epoch_grow": [1, 2, 4, 7, 10],  # [1, 1, 1, 1, 1],
            "num_batch": 50,
            "grow_base_size": {"0": 3, "2": 6, "4": 12, "6": 25},
            "grow_size": {"0": 1, "2": 2, "4": 4, "6": 8},
            # "grow_size": {"0": 2, "2": 2, "4": 4, "6": 8},
            "exsit_index": {"0": (0, 3), "2": (0, 6), "4": (0, 12), "6": (0, 25)},
        }

        dic_layer_in = {
            "epoch_from": 1,
            "layer": "1",
            "epoch_to": 11,
            "epoch_grow": [1, 2, 4, 7, 10],  # [1, 1, 1, 1, 1],
            "num_batch": 50,
            "grow_base_size": {"2": 3, "4": 6, "6": 12},
            "grow_size": {"2": 1, "4": 2, "6": 4},
            # "grow_size": {"2": 2, "4": 2, "6": 4},
            "exsit_index": {"2": (0, 3), "4": (0, 6), "6": (0, 12)},
        }

    else:
        dic_layer_out = {
            "epoch_from": 1,
            "layer": "1",
            "epoch_to": 11,
            "epoch_grow": [1, 2, 4, 7, 10],  # [1, 1, 1, 1, 1],
            "num_batch": 50,
            "grow_base_size": {"0": 2, "1": 5, "2": 10},
            "grow_size": {"0": 2, "1": 3, "2": 10},
            "exsit_index": {"0": (0, 2), "1": (0, 5), "2": (0, 10)},
        }

        dic_layer_in = {
            "epoch_from": 1,
            "layer": "1",
            "epoch_to": 11,
            "epoch_grow": [1, 2, 4, 7, 10],  # [1, 2, 4, 7, 10],
            "num_batch": 50,
            "grow_base_size": {"1": 2, "2": 80, "3": 10},
            "grow_size": {"1": 2, "2": 48, "3": 10},
            "exsit_index": {"1": (0, 2), "2": (0, 80), "3": (0, 10)},
        }

    # dic_layer_out["grow_size"] = {"0": 4, "1": 10, "2": 100}
    # dic_layer_out["grow_base_size"] = {"0": 4, "1": 10, "2": 100}
    # dic_layer_in["grow_size"] = {"1": 4, "2": 160, "3": 100}
    # dic_layer_in["grow_base_size"] = {"1": 4, "2": 160, "3": 100}

    # for file in [rank_cumulative_low]:#[ranklow_n,rank_cumulative_cosine_low]:
    # for key in dic_lr.keys():
    #     file = dic_lr[key]
    # [rank_cumulative_cosine,
    # rank_baseline, copy_n, copy_one,bridging,rank_cumulative_cosine,rank_cumulative_cosine,random,
    #          rank_cumulative_cosine_bn]:  # [rank_baseline_bn, copy_n_bn, copy_one_bn, rank_ours_bn, bridging_bn]:#[rank_baseline, copy_n, copy_one, rank_ours]:  # [good1,good2,bad1,bad2,nono_bad1, nono_bad2]: #, good1, good2

    for key in dic.keys():
        print(key, ':')

        if key.startswith('standard'):
            dic_layer_in["epoch_grow"] = [1, 1, 1, 1, 1]
            dic_layer_out["epoch_grow"] = [1, 1, 1, 1, 1]

        # elif key == 'standard_s':
        #     dic_layer_out["grow_size"] = {"0": 4, "1": 10, "2": 100}
        #     dic_layer_out["grow_base_size"] = {"0": 4, "1": 10, "2": 100}
        #     dic_layer_in["grow_size"] = {"1": 4, "2": 160, "3": 100}
        #     dic_layer_in["grow_base_size"] = {"1": 4, "2": 160, "3": 100}
        #
        # elif key.endswith('overP'):
        #     dic_layer_out["grow_size"] = {"0": 4, "1": 10, "2": 100}
        #     dic_layer_out["grow_base_size"] = {"0": 4, "1": 10, "2": 100}
        #     dic_layer_in["grow_size"] = {"1": 4, "2": 160, "3": 100}
        #     dic_layer_in["grow_base_size"] = {"1": 4, "2": 160, "3": 100}

        else:
            dic_layer_in["epoch_grow"] = [1, 2, 4, 7, 10]
            dic_layer_out["epoch_grow"] = [1, 2, 4, 7, 10]
            # dic_layer_in["epoch_grow"] = [1, 1, 1, 1, 2]
            # dic_layer_out["epoch_grow"] = [1, 1, 1, 1, 2]

        path_to_file = dic[key]

        # path = os.path.dirname(path_to_file)

        # files = [f for f in glob.glob(path + "/*__json_in_out.json")]

        # files = [f for f in glob.glob(path_to_file + "/*_copy.json")]

        file = path_to_file

        with open(file, 'r') as f:
            data = json.load(f)
        for out in [True, False]:
            is_out = out
            path = os.path.dirname(file)
            if is_out:
                for layer in dic_layer_out["grow_base_size"].keys():  #
                    dic_layer_out["layer"] = layer
                    x_y = gen_x_y(is_out=is_out)
                    plot_save(x_y, path, layer=layer, is_out=is_out)
                    # target_path = "/Users/zber/Documents/FGdroid/exp_result/S_score_comparison/bridging.json"
                    # with open(target_path, 'w') as f:
                    #     json.dump(x_y, f, indent=4)

            else:
                if mode == 'ACV' or mode == 'W' or mode == 'Delta_S':
                    continue
                for layer in dic_layer_in["grow_base_size"].keys():  #
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
    #                 s_ = data[str(e)][str(b * n)][dic["layer"]][mode]["exist_out"]
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
    #         for mode in [mode, "L1", "L2"]:
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
    #         for mode in [mode, "L1", "L2"]:
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
    #         for mode in [mode, "L1", "L2"]:
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
