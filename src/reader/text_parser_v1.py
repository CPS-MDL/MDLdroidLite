import glob
import os
import numpy as np
import json

# bridging_bad = '/Users/zber/ProgramDev/exp_pyTorch/results/grow_bridging_og2_new2_bad/layer0_log.txt'
# bridging_good = '/Users/zber/ProgramDev/exp_pyTorch/results/grow_bridging_og2_new2_good/layer0_log.txt'
# randomMap_bad = '/Users/zber/ProgramDev/exp_pyTorch/results/grow_randomMap_og2_new2_bad/layer0_log.txt'
# randomMap_good = '/Users/zber/ProgramDev/exp_pyTorch/results/grow_randomMap_og2_new2_good/layer0_log.txt'
# rank_bad = '/Users/zber/ProgramDev/exp_pyTorch/results/grow_rankconnect_og2_new2_bad/layer0_log.txt'
# rank_good = '/Users/zber/ProgramDev/exp_pyTorch/results/grow_rankconnect_og2_new2_good/layer0_log.txt'
# ours_bad = '/Users/zber/ProgramDev/exp_pyTorch/results/grow_rankgroup_og2_new2_bad/layer0_log.txt'
# ours_good = '/Users/zber/ProgramDev/exp_pyTorch/results/grow_rankgroup_og2_new2_good/layer0_log.txt'
# bb = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_bridging_og2_new2_20200327-210217/layer0_log.txt"
#
# bridging_bad = '/Users/zber/ProgramDev/exp_pyTorch/results/grow_bridging_og2_new2_bad/grow_bridging_og2_new2__log.txt'
# bridging_good = '/Users/zber/ProgramDev/exp_pyTorch/results/grow_bridging_og2_new2_good/grow_bridging_og2_new2__log.txt'
# randomMap_bad = '/Users/zber/ProgramDev/exp_pyTorch/results/grow_randomMap_og2_new2_bad/grow_randomMap_og2_new2__log.txt'
# randomMap_good = '/Users/zber/ProgramDev/exp_pyTorch/results/grow_randomMap_og2_new2_good/grow_randomMap_og2_new2__log.txt'
# rank_bad = '/Users/zber/ProgramDev/exp_pyTorch/results/grow_rankconnect_og2_new2_bad/grow_rankconnect_og2_new2__log.txt'
# rank_good = '/Users/zber/ProgramDev/exp_pyTorch/results/grow_rankconnect_og2_new2_good/grow_rankconnect_og2_new2__log.txt'
# ours_bad = '/Users/zber/ProgramDev/exp_pyTorch/results/grow_rankgroup_og2_new2_bad/grow_rankgroup_og2_new2__log.txt'
# ours_good = '/Users/zber/ProgramDev/exp_pyTorch/results/grow_rankgroup_og2_new2_good/grow_rankgroup_og2_new2__log.txt'
# bb = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_bridging_og2_new2_20200327-210217/layer0_log.txt"
# """
# /Users/zber/ProgramDev/exp_pyTorch/results/grow_bridging_og2_new2_bad
# /Users/zber/ProgramDev/exp_pyTorch/results/grow_bridging_og2_new2_good
# /Users/zber/ProgramDev/exp_pyTorch/results/grow_randomMap_og2_new2_bad
# /Users/zber/ProgramDev/exp_pyTorch/results/grow_randomMap_og2_new2_good
# /Users/zber/ProgramDev/exp_pyTorch/results/grow_rankconnect_og2_new2_bad
# /Users/zber/ProgramDev/exp_pyTorch/results/grow_rankconnect_og2_new2_good
# /Users/zber/ProgramDev/exp_pyTorch/results/grow_rankgroup_og2_new2_bad
# /Users/zber/ProgramDev/exp_pyTorch/results/grow_rankgroup_og2_new2_good
# """
#
# good1 = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rankconnect_ogn_newn_2_1/layer0_log.txt"
# good2 = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rankconnect_ogn_newn_2/layer0_log.txt"
# good3 = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rankconnect_ogn_newn_3/layer0_log.txt"
#
# bad2 = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rankconnect_ogn_newn_4down/layer0_log.txt"
# bad1 = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rankconnect_ogn_newn_bad/layer0_log.txt"
#
# nono_bad2 = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rankconnect_ognon-scale_newnon-scale_bad2/layer0_log.txt"
# nono_bad1 = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rankconnect_ognon-scale_newnon-scale_bad1/layer0_log.txt"

dic_base = dict(
    # rank_baseline="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_baseline_6/layer1_log.txt",
    # rank_baseline_before="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_baseline_5/layer1_log.txt",
    # rank_baseline_gate="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_baseline_gate/layer1_log.txt",
    # rank_baseline_LR3="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_baseline_LR3_20200412-233035/layer2_log.txt",
    # rank_baseline_LR4="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_baseline_LR4_20200413-001813/layer2_log.txt",
    # copy_n="/Users/zber/ProgramDev/exp_pyTorch/results/grow_copy_n_6/layer1_log.txt",
    # copy_one="/Users/zber/ProgramDev/exp_pyTorch/results/grow_copy_one_6/layer1_log.txt",
    # rank_ours="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_ours_11/layer1_log.txt",
    # bridging="/Users/zber/ProgramDev/exp_pyTorch/results/grow_bridging_6/layer1_log.txt",
    # bridging2 = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_bridging_7/layer1_log.txt",
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

    # rank_cumulative_con2_fc2="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_con2_fc2/layer1_log.txt",
    # rank_cumulative_con2_fc1="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_con2_fc1/layer1_log.txt",
    # rank_cumulative_con1_fc2="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_con1_fc2/layer1_log.txt",
    #
    # rank_cumulative_conR_fcR="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_conR_fcR/layer1_log.txt",
    # rank_cumulative_conR_fc1="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_conR_fc1/layer1_log.txt",
    # rank_cumulative_con1_fcR="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_con1_fcR/layer1_log.txt",
    #
    # rank_cumulative_conN_fcN="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_conN_fcN/layer1_log.txt",
    # rank_cumulative_conN_fc1="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_conN_fc1/layer1_log.txt",
    # rank_cumulative_con1_fcN="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_con1_fcN/layer1_log.txt",
    #
    # rank_cumulative_conR_fcR_15="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_conR_fcR_15/layer1_log.txt",
    # rank_cumulative_con2_fc2_15="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_con2_fc2_15/layer1_log.txt",

    # rank_cumulative_con2_fc2_in2="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_con2_fc2_in2/layer1_log.txt",
    # rank_cumulative_conR_fcR_in2="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_conR_fcR_in2/layer1_log.txt",
    # rank_cumulative_conN_fcN_in2="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_conN_fcN_in2/layer1_log.txt",
    # rank_cumulative_conR_fcR_in2_out2="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_conR_fcR_in2_out2/layer1_log.txt",
    # rank_cumulative_conN_fcN_in2_out2="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_conN_fcN_in2_out2/layer1_log.txt",
    # rank_cumulative_con2_fc2_in2_out2="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_con2_fc2_in2_out2/layer1_log.txt",
    #
    # rank_cumulative_conR_fcR_inR_ogR="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_conR_fcR_inR_ogR/layer1_log.txt",
    # rank_cumulative_conR_fcR_gateAuto_test="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_conR_fcR_gateAuto/layer1_log.txt",
    # rank_cumulative_con2R_fc2R_gateAuto_test="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_con2R_fc2R_gateAuto/layer1_log.txt",
    # rank_cumulative_con2R_fc2R_in2_out2_gateAuto_test="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_con2R_fc2R_in2_out2_gateAuto/layer1_log.txt",
    # rank_cumulative_con2R_fc2R_gateAuto300_test="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_con2R_fc2R_gateAuto300/layer1_log.txt",
    #
    # rank_cumulative_con2R_fc2R_gaTe_test="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_con2R_fc2R_gaTe/layer1_log.txt",
    # rank_cumulative_con2R_fc2R_gate_Auto_test="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_con2R_fc2R_gate_Auto/layer1_log.txt",
    # rank_cumulative_con2R_fc2R_gateAutoMax300_test="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_con2R_fc2R_gateAutoMax300/layer1_log.txt",
    # rank_cumulative_con2R_fc2R_gateAutoMax200_test="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_con2R_fc2R_gateAutoMax200/layer1_log.txt",
    # rank_cumulative_con2R_fc2R_gateAutoMax100_test="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_con2R_fc2R_gateAutoMax100/layer1_log.txt",
    # rank_cumulative_con2R_fc2R_gateMax_test="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_con2R_fc2R_gateMax/layer1_log.txt",
    #
    # rank_cumulative_con2R_fc2R_LR_exsit="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_20200428-001128/layer1_log.txt",

    # rank_cumulative_con2R_fc2R_gate_LR_min_test="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_gate_LR_min/layer1_log.txt",
    # rank_cumulative_con2R_fc2R_LR300="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_con2R_fc2R_LR300/layer1_log.txt",

    # ve_con2R_fc2R_in2R_gateAutoMax_LR300_1_test="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_con2R_fc2R_in2R_gateAutoMax_LR300_1/layer1_log.txt",
    # rank_cumulative_con2_fc2="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_og2_in2/layer1_log.txt",
    # rank_cumulative_con1_fc2="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_con1_fc2_f/layer1_log.txt",
    # rank_cumulative_con2_fc1="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_con2_fc1_f/layer1_log.txt",

    # rank_cumulative_gate="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_gate/layer1_log.txt",
    # rank_cumulative_H_normal="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_rank_20200415-161606/layer2_log.txt",
    # rank_cumulative_conv_gate_fc_nochange_test="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_conv_gate_fc_nochange_test/layer1_log.txt",
    # rank_cumulative_conv_gateauto_fc_nochange_test="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_conv_gateauto_fc_nochange_test/layer1_log.txt",
    # rank_cumulative_conv_gate_fc_n_test="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_conv_gate_fc_n_test/layer1_log.txt",
    # rank_cumulative_conv_gate_fc_gate_n_test="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_conv_gate_fc_gate_n_test/layer1_log.txt",
    # rank_cumulative_conv_gate_fc_gate_test="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_conv_gate_fc_gate_test/layer1_log.txt",
    # rank_cumulative_conv_gate_fc_gate_ratio_test="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_conv_gate_fc_gate_ratio_test/layer1_log.txt",

    # rank_cumulative_gate_ratio_test="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_gate_ratio_test/layer1_log.txt",
    # rank_cumulative_gate_Fratio_test="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_gate_Fratio_test/layer1_log.txt",
    # rank_cumulative_gate_Fratio5_test="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_gate_Fratio0.5_test/layer1_log.txt",
    # rank_cumulative_gate_Fratio7_test="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_gate_Fratio0.7_test/layer1_log.txt",
    # rank_cumulative_gate_Fratio2_test="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_gate_Fratio0.2_test/layer1_log.txt",
    # rank_cumulative_low_normal="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_low_20200415-161006/layer1_log.txt",
    # rank_cumulative_gate_max_test="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_gate_max_test/layer2_log.txt",
    # rank_cumulative_LR_rank="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_LR_rank/layer1_log.txt",
    # rank_cumulative_in_test="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_innochange_addtest/layer2_log.txt",
    # rank_cumulative_inn_test="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_inN_addtest/layer2_log.txt",
    # rank_cumulative_in2_og2_test="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_ogn_inn/layer1_log.txt",
    # rank_cumulative_low="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_low_1/layer1_log.txt",
    # standard="/Users/zber/ProgramDev/exp_pyTorch/results/grow_standard_3/grow_standard__json_in_out.json",
    # standard_s="/Users/zber/ProgramDev/exp_pyTorch/results/grow_standard_s/grow_standard__json_dic.json",
    # rank_cumulative_cL_fH="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_convL_fcH/layer1_log.txt",
    # rank_cumulative_cL_fH2="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_convL_fcH2/layer1_log.txt",

    # 30 April, 2019
    # standard_s_20_500_50="/Users/zber/ProgramDev/exp_pyTorch/results/grow_standard_20_50_500/layer1_log.txt",
    # standard_10_17_50="/Users/zber/ProgramDev/exp_pyTorch/results/standard_10_17_50/layer1_log.txt",
    # standard_8_14_40="/Users/zber/ProgramDev/exp_pyTorch/results/standard_8_14_40/layer1_log.txt",
    # standard_6_11_30="/Users/zber/ProgramDev/exp_pyTorch/results/standard_6_11_30/layer1_log.txt",
    # standard_4_8_20="/Users/zber/ProgramDev/exp_pyTorch/results/standard_4_8_20/layer1_log.txt",
    # standard_2_5_10="/Users/zber/ProgramDev/exp_pyTorch/results/standard_2_5_10/layer1_log.txt",
    # rank_baseline="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_baseline_20200430-190631/layer1_log.txt",
    # random="/Users/zber/ProgramDev/exp_pyTorch/results/grow_random_20200430-194619/layer1_log.txt",
    # copy_n="/Users/zber/ProgramDev/exp_pyTorch/results/grow_copy_n_20200430-191941/layer1_log.txt",
    # bridging="/Users/zber/ProgramDev/exp_pyTorch/results/grow_bridging_20200430-193258/layer1_log.txt",
    # rank_cumulative="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_20200430-213135/layer1_log.txt",
    # rank_cumulative_cosine="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_cosine_20200430-214432/layer1_log.txt",

    # 5-1
    # rank_consine="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_20200501-195838/layer1_log.txt",
    # rank_cumulative2R_output_no_2R="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative2R_output_no2R/layer1_log.txt",
    # rank_cumulative_gate_autoadd_test="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_gate_autoadd/layer1_log.txt",
    # rank_2R="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_2R/layer1_log.txt",

    # 5-3
    # rank_one_all2="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_one_all2/layer1_log.txt",
    # standard_s_random_overP="/Users/zber/ProgramDev/exp_pyTorch/results/grow_random_overP/layer1_log.txt",
    # standard_s_rank_baseline_overP="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_baseline_overP/layer1_log.txt",
    # standard_s_copy_n="/Users/zber/ProgramDev/exp_pyTorch/results/grow_copy_n_20200503-183009/layer1_log.txt",
    # standard_s_bridging="/Users/zber/ProgramDev/exp_pyTorch/results/grow_bridging_20200503-191340/layer1_log.txt",
    # rank_ours_batch20="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_ours_batch20_no2R/layer1_log.txt",
    # rank_cumulative_batch20="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_batch20_no2R/layer1_log.txt",

    # 5-5
    # standard_2_5_10="/Users/zber/ProgramDev/exp_pyTorch/results/standard_2_5_10_20200505-211708/layer1_log.txt",
    # standard_4_8_20="/Users/zber/ProgramDev/exp_pyTorch/results/standard_4_8_20_20200505-212418/layer1_log.txt",
    # standard_6_11_30="/Users/zber/ProgramDev/exp_pyTorch/results/standard_6_11_30_20200505-213355/layer1_log.txt",
    # standard_8_14_40="/Users/zber/ProgramDev/exp_pyTorch/results/standard_8_14_40_20200505-214554/layer1_log.txt",
    # standard_10_17_50_1="/Users/zber/ProgramDev/exp_pyTorch/results/standard_10_17_50_20200508-194338/layer1_log.txt",

    # 5-6
    # rank_ours_dynamic_grow="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_ours_dynamic_grow/layer1_log.txt",
    # rank_ours_dynamic_grow_bias_big = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_ours_20200506-222539/layer1_log.txt",
    # rank_ours_dynamic_grow1_bias_small =  "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_ours_20200506-221445/layer1_log.txt",
    # rank_ours_BN="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_ours_BN_20200506-175156/layer1_log.txt",
    # rank_ours="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_ours_20200506-000800/layer1_log.txt",

    # 5-7
    # rank_ours_normal = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_ours_max_normal_noise/layer1_log.txt",
    # rank_ours_normal_ten = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_ours_max_normal_noise10/layer1_log.txt",
    # rank_ours_max_no_noise = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_ours_max_no_noise2/layer1_log.txt",
    # rank_ours_max_uniform_noi ="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_ours_max_uniform_noi/layer1_log.txt",
    # rank_ours_avg="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_ours_avg/layer1_log.txt",
    # rank_ours_all_max="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_ours_all_max/layer1_log.txt",

    # rank_ours_max_normal_noies = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_ours_max_normal_noies/layer1_log.txt",
    # rank_ours_select_max="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_ours_select_max/layer1_log.txt",
    # rank_baseline_scale= "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_baseline_scale_20200508-000823/layer1_log.txt"

    # 5-8
    # standard_10_17_50="/Users/zber/ProgramDev/exp_pyTorch/results/standard_10_17_50/layer1_log.txt",
    # standard_reg_10_17_50="/Users/zber/ProgramDev/exp_pyTorch/results/standard_10_17_50_reg1/layer1_log.txt",
    # bridging="/Users/zber/ProgramDev/exp_pyTorch/results/grow_bridging_20200508-223706/layer1_log.txt",
    # rank_baseline="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_baseline_20200508-224355/layer1_log.txt",
    # random="/Users/zber/ProgramDev/exp_pyTorch/results/grow_random_20200508-225047/layer1_log.txt",
    # copy_n="/Users/zber/ProgramDev/exp_pyTorch/results/grow_copy_n_20200508-225737/layer1_log.txt",
    # rank_ours = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_ours_noise_bias_scale/layer1_log.txt",
    # random_reg1="/Users/zber/ProgramDev/exp_pyTorch/results/grow_random_reg0.1/layer1_log.txt",
    # random_reg2="/Users/zber/ProgramDev/exp_pyTorch/results/grow_random_reg0.01/layer1_log.txt",
    # random_reg3="/Users/zber/ProgramDev/exp_pyTorch/results/grow_random_reg0.001/layer1_log.txt",
    # random_reg4="/Users/zber/ProgramDev/exp_pyTorch/results/grow_random_reg0.0001/layer1_log.txt",
    # random_reg5="/Users/zber/ProgramDev/exp_pyTorch/results/grow_random_reg0.00001/layer1_log.txt",
    # random_gate_10_05_test="/Users/zber/ProgramDev/exp_pyTorch/results/grow_random_gate_10_05/layer1_log.txt",
    # random_gate_15_10_test="/Users/zber/ProgramDev/exp_pyTorch/results/grow_random_gate_15-10/layer1_log.txt",

    # E20
    # low_cosine_e20="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_20e_20200511-041701/layer1_log.txt",
    # rank_low_e20="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_ours_20e_20200511-043252/layer1_log.txt",
    # random_e20="/Users/zber/ProgramDev/exp_pyTorch/results/grow_random_20e_20200511-040126/layer1_log.txt",
    # copy_n_e20="/Users/zber/ProgramDev/exp_pyTorch/results/grow_copy_n_e20_20200512-004939/layer1_log.txt",
    # rank_baseline_e20="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_baseline_e20_20200512-010604/layer1_log.txt",
    # bridging_e20="/Users/zber/ProgramDev/exp_pyTorch/results/grow_bridging_e20_20200512-003340/layer1_log.txt",

    # 5-10
    # rank_ours_low="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_ours_low/layer1_log.txt",
    # rank_ours_vg_low="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_ours_vg_low/layer1_log.txt",
    # standard_10_17_50="/Users/zber/ProgramDev/exp_pyTorch/results/standard_10_17_50_20200510-201607/layer1_log.txt",
    # low_cosine="/Users/zber/ProgramDev/exp_pyTorch/results/grow_low_cosine_20200511-000208/layer1_log.txt",
    # rank_ours_cosine="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_20200511-001645/layer1_log.txt",
    # rank_ours_rank="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_ours_rank/layer1_log.txt",

    # 5-11
    # rank_low_overP = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_ours_overP_20200511-012542/layer1_log.txt",
    # low_cosine_overP = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_overP_20200511-010217/layer1_log.txt",
    # low_cosine_LR_05 = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_LR_05_20200511-143658/layer1_log.txt",
    # low_cosine_LR_2 = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_LR_2_20200511-142440/layer1_log.txt",
    # low_cosine="/Users/zber/ProgramDev/exp_pyTorch/results/grow_low_cosine_20200511-000208/layer1_log.txt",
    # low_cosine_LR_10="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_LR_10_20200511-182123/layer1_log.txt",
    # low_cosine_LR_20="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_LR20/layer1_log.txt",

    # bridging_overP="/Users/zber/ProgramDev/exp_pyTorch/results/grow_bridging_overP_20200514-015356/layer1_log.txt",
    # baseline_overP="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_baseline_overP_20200514-021729/layer1_log.txt",
    # random_overP="/Users/zber/ProgramDev/exp_pyTorch/results/grow_random_overP_20200514-024035/layer1_log.txt",
    # copy_n_overP="/Users/zber/ProgramDev/exp_pyTorch/results/grow_copy_n_overP_20200514-030402/layer1_log.txt",
    # low_cosine_overP="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_overP_20200514-032731/layer1_log.txt",
    # standard_20_50_500="/Users/zber/ProgramDev/exp_pyTorch/results/standard_20_50_500_20200514-010144/layer1_log.txt",

    # standard_2_5_10="/Users/zber/ProgramDev/exp_pyTorch/results/standard_2_5_10_20200514-192251/layer1_log.txt",
    # standard_4_8_20="/Users/zber/ProgramDev/exp_pyTorch/results/standard_4_8_20_20200514-192745/layer1_log.txt",
    # standard_6_11_30="/Users/zber/ProgramDev/exp_pyTorch/results/standard_6_11_30_20200514-193258/layer1_log.txt",
    # standard_8_14_40="/Users/zber/ProgramDev/exp_pyTorch/results/standard_8_14_40_20200514-193833/layer1_log.txt",
    # standard_10_17_50="/Users/zber/ProgramDev/exp_pyTorch/results/standard_10_17_50_20200514-194428/layer1_log.txt",

    # rank_baseline="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_baseline_20200514-195931/layer1_log.txt",
    # random="/Users/zber/ProgramDev/exp_pyTorch/results/grow_random_20200514-200640/layer1_log.txt",
    # copy_n="/Users/zber/ProgramDev/exp_pyTorch/results/grow_copy_n_20200514-201348/layer1_log.txt",
    # bridging="/Users/zber/ProgramDev/exp_pyTorch/results/grow_bridging_20200514-195225/layer1_log.txt",
    # low_cosine="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_20200514-171924/layer1_log.txt",

    # low_cosine_LR10="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_LR10_20200514-182026/layer1_log.txt",
    # low_cosine_LR20="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_LR20_20200514-182938/layer1_log.txt",
    # low_cosine_wd="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_wd_20200514-185357/layer1_log.txt",

    # cosine_max_select_converge_E5_150_LR2_decay="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_max_select_converge_E5_150_LR2_decay_20200516-160510/layer1_log.txt",
    # cosine_max_select_converge_E5_150_LR4_decay="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_max_select_converge_E5_150_LR4_decay_20200516-160142/layer1_log.txt",
    #
    # cosine_max_select_converge_E5_150_LR2_decay_S3="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_max_select_converge_E5_150_LR2_decay_S3_20200516-172456/layer1_log.txt",
    # cosine_max_select_converge_E5_150_LR4_decay_S3="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_max_select_converge_E5_150_LR4_decay_S3_20200516-171602/layer1_log.txt",

    # lambda -> 1
    # cosine_max_select_converge_LR2_decay_lambda = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_max_select_converge_E5_150_LR2_decay_Lambda_20200516-182642/layer1_log.txt",
    # cosine_max_select_converge_LR4_decay_lambda = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_max_select_converge_E5_150_LR4_decay_Lambda_20200516-182202/layer1_log.txt",
    # cosine_max_select_converge_copy_optimizer_EXdecay_LR2="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_max_select_converge_copy_optimizer_EXdecay_LR2_20200516-202511/layer1_log.txt",
    # cosine_max_select_converge_renew_optimier_LR2="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_max_select_converge_renew_optimier_LR2_20200516-205802/layer1_log.txt",
    # cosine_max_select_converge_copy_optimizer_LR2="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_max_select_converge_copy_optimizer_LR2_20200516-204614/layer1_log.txt",

    # converge_renew_optimizer="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_max_select_converge_renew_optimizer_20200516-214022/layer1_log.txt",
    # converge_copy_optimizer="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_max_select_converge_copy_optimizer_20200516-220235/layer1_log.txt",

    # baseline_renew_optimizer="/Users/zber/ProgramDev/exp_pyTorch/results/renew_optimizer_grow_rank_baseline_20200516-213829/layer1_log.txt",
    # random_renew_optimizer="/Users/zber/ProgramDev/exp_pyTorch/results/renew_optimizer_grow_random_20200516-214520/layer1_log.txt",
    # copy_renew_optimizer="/Users/zber/ProgramDev/exp_pyTorch/results/renew_optimizer_grow_copy_n_20200516-215209/layer1_log.txt",
    # cosine_renew_optimizer="/Users/zber/ProgramDev/exp_pyTorch/results/renew_optimizer_grow_rank_cosine_20200516-215857/layer1_log.txt",
    # bridging_renew_optimizer="/Users/zber/ProgramDev/exp_pyTorch/results/renew_optimizer_grow_bridging_20200516-213142/layer1_log.txt",

    # 5-16
    # weight decay + lambda
    # cosine_weight_decay_lambda="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_weight_decay_lambda_20200516-222256/layer1_log.txt",
    # cosine_weight_decay="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_weight_decay_20200516-221613/layer1_log.txt",
    # low_cosine_Sdecay_600="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_Sdecay_600_20200515-172914/layer1_log.txt",
    # low_cosine_max_select_converge="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_max_select_converge_E5_150_20200516-150254/layer1_log.txt",
    # cosine_wd_all_lambda_hidden="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_wd_all_lambda_hidden_20200516-231143/layer1_log.txt",
    # cosine_max_select_converge_E5_150_LR2_decay="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_max_select_converge_E5_150_LR2_decay_20200516-160510/layer1_log.txt",
    # cosine_max_select_converge_LR2_decay_lambda="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_max_select_converge_E5_150_LR2_decay_Lambda_20200516-182642/layer1_log.txt",
    # cosine_max_select_converge_copy_optimizer_EXdecay_LR2="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_max_select_converge_copy_optimizer_EXdecay_LR2_20200516-202511/layer1_log.txt",
    # cosine_max_select_converge_renew_optimier_LR2="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_max_select_converge_renew_optimier_LR2_20200516-205802/layer1_log.txt",

    # 5-17
    # cosine_converge_LR2decay_OPrenew = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_converge_LR2decay_OPrenew/layer1_log.txt",
    # cosine_converge_LR2decay_OPcopy="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_converge_LR2decay_OPcopy/layer1_log.txt",
    # cosine_converge_LR2decay_OPavg = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_converge_LR2decay_OPavg/layer1_log.txt",
    # cosine_converge_LR2decay_OPscaleAVG = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_converge_LR2decay_OPscaleAVG_20200517-155545/layer1_log.txt",

    # cosine_converge_LR2decay_LMDdecayAsy_OPrenew="/Users/zber/ProgramDev/exp_pyTorch/results/cosine_converge_LR2decay_LMDdecayAsy_OPrenew_20200517-215549/layer1_log.txt",
    # cosine_converge_LR2decay_LMDdecayAsyR_OPrenew="/Users/zber/ProgramDev/exp_pyTorch/results/cosine_converge_LR2decay_LMDdecayAsyR_OPrenew_20200517-214313/layer1_log.txt",

    # cosine_converge_LR2decay_LMDdecayAsyR_OPrenew_E10="/Users/zber/ProgramDev/exp_pyTorch/results/cosine_converge_LR2decay_LMDdecayAsyR_OPrenew_E10_20200517-204159/layer1_log.txt",
    # cosine_converge_LR2decay_LMDdecay_OPrenew_E10="/Users/zber/ProgramDev/exp_pyTorch/results/cosine_converge_LR2decay_LMDdecay_OPrenew_E10_20200517-175337/layer1_log.txt",
    # cosine_converge_LR2decay_LMDdecayAsy_OPrenew_E10="/Users/zber/ProgramDev/exp_pyTorch/results/cosine_converge_LR2decay_LMDdecayAsy_OPrenew_E10_20200517-223242/layer1_log.txt",

    # good_cosine="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_20200518-151420/layer1_log.txt",
    # cosine_Asy_Xavier="/Users/zber/ProgramDev/exp_pyTorch/results/cosine_Asy_Xavier_20200518-204628/layer1_log.txt",
    # cosine_AsyR_Xavier="/Users/zber/ProgramDev/exp_pyTorch/results/cosine_AsyR_Xavier_20200518-190439/layer1_log.txt",
    # cosine_RI_600 = "/Users/zber/ProgramDev/exp_pyTorch/results/cosine_RI_step600_20200518-222843/layer1_log.txt",

    # 5-19
    # standard_10_17_50_Xavier="/Users/zber/ProgramDev/exp_pyTorch/results/standard_10_17_50_Xavier_20200519-171213/layer1_log.txt",
    # cosine_RI_step600="/Users/zber/ProgramDev/exp_pyTorch/results/cosine_RI_step600_20200518-222843/layer1_log.txt",
    # cosine_RI_step600_noNoise="/Users/zber/ProgramDev/exp_pyTorch/results/cosine_RI_step600_Nonoise_20200519-175432/layer1_log.txt",
    # cosine_RI_bias_scale="/Users/zber/ProgramDev/exp_pyTorch/results/cosine_RI_bias_scale_20200519-192346/layer1_log.txt",

    # cosine_RI_Sstd="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_20200522-165406/layer1_log.txt",
    # cosine_RI_Sstd_power="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_20200522-174535/layer1_log.txt",
    # consine_RI_Sstd_weightNew_power="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_20200522-182027/layer1_log.txt",
    # cosine_RI_momentum = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_20200522-233322/layer1_log.txt",

    #
    # cosine_L_1_index="/Users/zber/ProgramDev/exp_pyTorch/results/cosine_L_1_index_20200530-192920/layer1_log.txt",
    # cosine_L_L_index="/Users/zber/ProgramDev/exp_pyTorch/results/cosine_L_L_index_20200530-194103/layer1_log.txt",
    cosine_L_1_index="/Users/zber/ProgramDev/exp_pyTorch/results/cosine_L_1_index_20200530-192920/layer1_log.txt",
    cosine_L_L_index="/Users/zber/ProgramDev/exp_pyTorch/results/cosine_L_L_index_20200530-194103/layer1_log.txt",
    cosine_L_1L_index="/Users/zber/ProgramDev/exp_pyTorch/results/cosine_L_1L_index_20200530-195356/layer1_log.txt",

)
# dic_lr = dict(
#     bridging_lr="/Users/zber/ProgramDev/exp_pyTorch/results/grow_bridging_LR_20200409-211253/layer1_log.txt",
#     ranklow_one_lr="/Users/zber/ProgramDev/exp_pyTorch/results/grow_ranklow_one_LR_20200409-215528/layer1_log.txt",
#     rank_cumulative_cosine_low_lr="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_cosine_LR_20200409-212445/layer1_log.txt",
#     rank_cumulative_LR="/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_LR_20200410-001857/layer1_log.txt",
# )

# rank_baseline_bn = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_baseline_BN_1/layer1_log.txt"
# copy_n_bn = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_copy_n_BN_1/layer1_log.txt"
# copy_one_bn = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_copy_one_BN_1/layer1_log.txt"
# rank_ours_bn = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_ours_BN_1/layer1_log.txt"
# bridging_bn = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_bridging_BN_1/layer1_log.txt"
# rank_cumulative_cosine_bn = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cumulative_cosine_BN_1/layer1_log.txt"
# random_bn = '/Users/zber/ProgramDev/exp_pyTorch/results/grow_random_bn_1/layer1_log.txt'

# path_list = []
# for i in range(1, 6):
#     p = base_path.format(i=i)
#     path_list.append(p)

num_growth = 5
dic = dic_base


def str_to_float(str):
    num_list = []
    str = str[1:-1]
    str_arry = str.split(', ')
    for num in str_arry:
        num_list.append(float(num))
    return np.asarray(num_list)


def print_acc(file, key):
    acc = []
    acc_str = ''
    if key.startswith('standard'):
        pass
    else:
        with open(file, "r") as f:
            for line in f:
                if line.startswith('Test'):
                    start = line.index('(')
                    end = line.index('%')
                    acc.append(line[start + 1:end])

    path = os.path.dirname(file)

    files = [f for f in glob.glob(path + "/*__log.txt")]

    with open(files[0], "r") as f:
        for line in f:
            if line.startswith('Accuracy:'):
                start = line.index('[')
                end = line.index(']')
                acc_str = line[start:end + 1]

    line = 0
    delta_acc = []

    # if not key.endswith('test'):
    #     while line < len(acc):
    #         pre_acc = 0
    #         for i in [0, 1, 3, 5, 6, 7]:  #:
    #             # for i in [0, 1, 3, 5]:  #:
    #             if i == 7:
    #                 print('{}'.format(acc[line + i]), end='\n')
    #             else:
    #                 print('{} -> '.format(acc[line + i]), end='')
    #
    #             if pre_acc == 0:
    #                 pre_acc = acc[line + i]
    #             else:
    #                 delta = float(pre_acc) - float(acc[line + i])
    #                 pre_acc = acc[line + i]
    #                 delta_acc.append(delta)
    #         line += 8
    # else:
    #     while line < len(acc):
    #         pre_acc = 0
    #         for i in [0, 1, 3, 5, 6]:  #:
    #             if i == 6:
    #                 print('{}'.format(acc[line + i]), end='\n')
    #             else:
    #                 print('{} -> '.format(acc[line + i]), end='')
    #
    #             if pre_acc == 0:
    #                 pre_acc = acc[line + i]
    #             else:
    #                 delta = float(pre_acc) - float(acc[line + i])
    #                 pre_acc = acc[line + i]
    #                 delta_acc.append(delta)
    #         line += 7

    print('acc: ' + acc_str)
    print('acc drop : ' + str(np.sum(delta_acc) / 4))


def print_all_acc(file):
    acc = []
    with open(file, "r") as f:
        for line in f:
            if line.startswith('Accuracy:'):
                print(line)
                break
    print()
    print()


def grow_x_y_exist(dic):
    y = []
    for e in range(1, dic["epoch_to"]):
        for n in range(1, (600 // dic["num_batch"]) + 1):
            batches = []
            for b in range(1, dic["num_batch"] + 1):
                s = data[str(e)][str(b * n - 1)][dic["layer"]]["S"]["out"]
                f = str_to_float(s)
                start = 0
                end = dic["grow_base_size"][dic["layer"]]
                f_need = f[start:end]
                f_mean = np.mean(f_need)
                batches.append(f_mean)
            y.append(np.mean(batches))
    length = len(y)
    x = np.arange(0, length)
    return x, y


def grow_x_y(dic, gi=1):
    y = []
    for e in range(dic["epoch_grow"][gi], dic["epoch_to"]):
        for n in range(1, (600 // dic["num_batch"]) + 1):
            batches = []
            for b in range(1, dic["num_batch"] + 1):
                s = data[str(e)][str(b * n - 1)][dic["layer"]]["S"]["out"]
                f = str_to_float(s)
                start = dic["grow_base_size"][dic["layer"]] + (gi - 1) * dic["grow_size"][dic["layer"]]
                end = start + dic["grow_size"][dic["layer"]]
                f_need = f[start:end]
                f_mean = np.mean(f_need)
                batches.append(f_mean)
            y.append(np.mean(batches))
    length = len(y)
    start_x = (dic["epoch_grow"][gi] - 1) * (600 // dic["num_batch"])
    x = np.arange(start_x, start_x + length)
    return x, y


def grow_x_y_exist_in(dic):
    y = []
    for e in range(1, dic["epoch_to"]):
        for n in range(1, (600 // dic["num_batch"]) + 1):
            batches = []
            for b in range(1, dic["num_batch"] + 1):
                s = data[str(e)][str(b * n - 1)][dic["layer"]]["S"]["in"]
                f = str_to_float(s)
                start = 0
                end = dic["grow_base_size"][dic["layer"]]
                f_need = f[start:end]
                f_mean = np.mean(f_need)
                batches.append(f_mean)
            y.append(np.mean(batches))
    length = len(y)
    x = np.arange(0, length)
    return x, y


def grow_x_y_in(dic, gi=1):
    y = []
    for e in range(dic["epoch_grow"][gi], dic["epoch_to"]):
        for n in range(1, (600 // dic["num_batch"]) + 1):
            batches = []
            for b in range(1, dic["num_batch"] + 1):
                s = data[str(e)][str(b * n - 1)][dic["layer"]]["S"]["in"]
                f = str_to_float(s)
                start = dic["grow_base_size"][dic["layer"]] + (gi - 1) * dic["grow_size"][dic["layer"]]
                end = start + dic["grow_size"][dic["layer"]]
                f_need = f[start:end]
                f_mean = np.mean(f_need)
                batches.append(f_mean)
            y.append(np.mean(batches))
    length = len(y)
    start_x = (dic["epoch_grow"][gi] - 1) * (600 // dic["num_batch"])
    x = np.arange(start_x, start_x + length)
    return x, y


# for f in [bb]:
#     print_acc(f)


def distance_x_y(file_key, dic, gi=1, is_out=False):
    y = []
    global mode
    key = "out" if is_out else "in"

    # if file_key.startswith('standard'):
    #     observe_epoch = 10
    # else:
    observe_epoch = 1

    for e in range(dic["epoch_grow"][gi], dic["epoch_grow"][gi] + observe_epoch):
        for n in range(1, (600 // dic["num_batch"]) + 1):
            batches = []
            for b in range(1, dic["num_batch"] + 1):
                s = data[str(e)][str(b * n - 1)][dic["layer"]][mode][key]
                if s == "[]":
                    continue
                f = str_to_float(s)
                start = dic["grow_base_size"][dic["layer"]] + (gi - 1) * dic["grow_size"][dic["layer"]]
                f_exist = f[:start]
                f_need = f[start:]
                f_exist_mean = np.mean(f_exist)
                f_new_mean = np.mean(f_need)
                batches.append(f_exist_mean - f_new_mean)
            if batches:
                y.append(np.mean(batches))
    length = len(y)
    x = np.arange(length)
    return x, y


def distance_x_y_std(file_key, dic, is_out=False):
    global num_growth
    dic_layer = {}

    if file_key.startswith('standard'):
        observe_epoch = 10
    else:
        observe_epoch = 1

    out_in = "out" if is_out else "in"
    for gi in range(1, num_growth):
        dic_layer[str(gi)] = []
        for e in range(dic["epoch_grow"][gi], dic["epoch_grow"][gi] + observe_epoch):
            for n in range(600):
                s = data[str(e)][str(n)][dic["layer"]]["S"][out_in]
                f = str_to_float(s)
                start = dic["grow_base_size"][dic["layer"]] + (gi - 1) * dic["grow_size"][dic["layer"]]
                f_exist = f[:start]
                f_need = f[start:]
                f_exist_mean = np.mean(f_exist)
                f_new_mean = np.mean(f_need)
                distance = f_exist_mean - f_new_mean
                dic_layer[str(gi)].append(distance)
    s_list = []
    for i in range(len(dic_layer['1'])):
        m_list = []
        for key in dic_layer.keys():
            m_list.append(dic_layer[key])
        s_list.append(np.std(m_list))
    return np.mean(s_list)


def distance_x_y_std(file_key, dic, is_out=False):
    global num_growth
    dic_layer = {}

    if file_key.startswith('standard'):
        observe_epoch = 10
    else:
        observe_epoch = 1

    out_in = "out" if is_out else "in"
    for gi in range(1, num_growth):
        dic_layer[str(gi)] = []
        for e in range(dic["epoch_grow"][gi], dic["epoch_grow"][gi] + observe_epoch):
            for n in range(600):
                s = data[str(e)][str(n)][dic["layer"]]["S"][out_in]
                f = str_to_float(s)
                start = dic["grow_base_size"][dic["layer"]] + (gi - 1) * dic["grow_size"][dic["layer"]]
                f_exist = f[:start]
                f_need = f[start:]
                f_exist_mean = np.mean(f_exist)
                f_new_mean = np.mean(f_need)
                distance = f_exist_mean - f_new_mean
                dic_layer[str(gi)].append(distance)
    s_list = []
    for i in range(len(dic_layer['1'])):
        m_list = []
        for key in dic_layer.keys():
            m_list.append(dic_layer[key])
        s_list.append(np.std(m_list))
    return np.mean(s_list)


def distance_x_y_std(file_key, dic, is_out=False):
    global num_growth
    dic_layer = {}

    if file_key.startswith('standard'):
        observe_epoch = 10
    else:
        observe_epoch = 1

    out_in = "out" if is_out else "in"
    for gi in range(1, num_growth):
        dic_layer[str(gi)] = []
        for e in range(dic["epoch_grow"][gi], dic["epoch_grow"][gi] + observe_epoch):
            for n in range(600):
                s = data[str(e)][str(n)][dic["layer"]]["S"][out_in]
                f = str_to_float(s)
                start = dic["grow_base_size"][dic["layer"]] + (gi - 1) * dic["grow_size"][dic["layer"]]
                f_exist = f[:start]
                f_need = f[start:]
                f_exist_mean = np.mean(f_exist)
                f_new_mean = np.mean(f_need)
                distance = f_exist_mean - f_new_mean
                dic_layer[str(gi)].append(distance)
    s_list = []
    for i in range(len(dic_layer['1'])):
        m_list = []
        for key in dic_layer.keys():
            m_list.append(dic_layer[key])
        s_list.append(np.std(m_list))
    return np.mean(s_list)


def generator(window, length):
    for i in range(0, length, window):
        yield i, i + window


def x_y_std(file_key, dic, is_out=False):
    global num_growth
    global mode
    s_list = []

    out_in = "out" if is_out else "in"

    for e in range(dic["epoch_from"], dic["epoch_to"]):
        for n in range(600):
            s = data[str(e)][str(n)][dic["layer"]][mode][out_in]
            if s == "[]":
                continue
            f = str_to_float(s)
            std = np.std(f)
            s_list.append(std)

    s_window = []
    for start, end in generator(50, len(s_list)):
        s_window.append(np.mean(s_list[start: end]))

    return s_window, np.mean(s_list)


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
    return x_y


if __name__ == '__main__':
    is_simple = False
    mode = 'S'

    # dic_layer_out = {
    #     "epoch_from": 1,
    #     "layer": "1",
    #     "epoch_to": 11,
    #     "epoch_grow": [1, 2, 4, 7, 10],  # [1, 1, 1, 1, 1]
    #     "num_batch": 1,
    #     "grow_base_size": {"0": 2, "1": 5, "2": 10},
    #     "grow_size": {"0": 2, "1": 3, "2": 10},
    #     "exsit_index": {"0": (0, 2), "1": (0, 5), "2": (0, 10)},
    # }
    #
    # dic_layer_in = {
    #     "epoch_from": 1,
    #     "layer": "1",
    #     "epoch_to": 11,
    #     "epoch_grow": [1, 2, 4, 7, 10],  # [1, 1, 1, 1, 1],
    #     "num_batch": 1,
    #     "grow_base_size": {"1": 2, "2": 80, "3": 10},
    #     "grow_size": {"1": 2, "2": 48, "3": 10},
    #     "exsit_index": {"1": (0, 2), "2": (0, 80), "3": (0, 10)},
    # }

    # dic_content = {'out': {'0': {'Grow_1': [], 'Grow_2': [], 'Grow_3': [], 'Grow_4': []},
    #                        '1': {'Grow_1': [], 'Grow_2': [], 'Grow_3': [], 'Grow_4': []},
    #                        '2': {'Grow_1': [], 'Grow_2': [], 'Grow_3': [], 'Grow_4': []}},
    #                'in': {'1': {'Grow_1': [], 'Grow_2': [], 'Grow_3': [], 'Grow_4': []},
    #                       '2': {'Grow_1': [], 'Grow_2': [], 'Grow_3': [], 'Grow_4': []},
    #                       '3': {'Grow_1': [], 'Grow_2': [], 'Grow_3': [], 'Grow_4': []}}}

    dic_layer_out = {
        "epoch_from": 1,
        "layer": "1",
        "epoch_to": 11,
        "epoch_grow": [1, 2, 4, 7, 10],  # [1, 1, 1, 1, 1],
        "num_batch": 1,
        "grow_base_size": {"0": 2, "1": 5, "2": 10},
        "grow_size": {"0": 2, "1": 3, "2": 10},
        "exsit_index": {"0": (0, 2), "1": (0, 5), "2": (0, 10)},
        "control_batches": 600
    }

    dic_layer_in = {
        "epoch_from": 1,
        "layer": "1",
        "epoch_to": 11,
        "epoch_grow": [1, 2, 4, 7, 10],  # [1, 2, 4, 7, 10],
        "num_batch": 1,
        "grow_base_size": {"1": 2, "2": 80, "3": 10},
        "grow_size": {"1": 2, "2": 48, "3": 10},
        "exsit_index": {"1": (0, 2), "2": (0, 80), "3": (0, 10)},
        "control_batches": 600
    }

    dic_element = {'Grow_1': [], 'Grow_2': [], 'Grow_3': [], 'Grow_4': []}
    dic_content = {'out': {'0': {'delta': [], 'delta-std': [], 'std': []},
                           '1': {'delta': [], 'delta-std': [], 'std': []},
                           '2': {'delta': [], 'delta-std': [], 'std': []}, },
                   'in': {'1': {'delta': [], 'delta-std': [], 'std': []},
                          '2': {'delta': [], 'delta-std': [], 'std': []},
                          '3': {'delta': [], 'delta-std': [], 'std': []}, }}

    for key in dic.keys():

        if key.startswith('standard'):
            dic_layer_in["epoch_grow"] = [1, 1, 1, 1, 1]
            dic_layer_out["epoch_grow"] = [1, 1, 1, 1, 1]
            if key == 'standard_s':
                dic_layer_out["grow_size"] = {"0": 4, "1": 10, "2": 100}
                dic_layer_out["grow_base_size"] = {"0": 4, "1": 10, "2": 100}
                dic_layer_in["grow_size"] = {"1": 4, "2": 160, "3": 100}
                dic_layer_in["grow_base_size"] = {"1": 4, "2": 160, "3": 100}

            num = key[-2:]
            if num.isdigit():
                num_int = int(num)
                num_growth = num_int / 10
                num_growth = int(num_growth)
            else:
                num_growth = 5

        elif key.endswith('overP'):
            dic_layer_out["grow_size"] = {"0": 4, "1": 10, "2": 100}
            dic_layer_out["grow_base_size"] = {"0": 4, "1": 10, "2": 100}
            dic_layer_in["grow_size"] = {"1": 4, "2": 160, "3": 100}
            dic_layer_in["grow_base_size"] = {"1": 4, "2": 160, "3": 100}

        else:
            dic_layer_in["epoch_grow"] = [1, 2, 4, 7, 10]
            dic_layer_out["epoch_grow"] = [1, 2, 4, 7, 10]

        print(key, ':')
        print_acc(dic[key], key)
        print("")
        if is_simple:
            continue

        path_to_file = dic[key]

        path = os.path.dirname(path_to_file)

        files = [f for f in glob.glob(path + "/*__json_in_out.json")]

        filename = files[0]

        with open(filename, 'r') as f:
            data = json.load(f)
        y = []
        for out in [True, False]:
            is_out = out
            if is_out:

                # for layer in ["0", "1", "2"]:  #
                #     dic_layer_out["layer"] = layer
                #     x_y = gen_x_y(is_out=is_out)
                #     existing = np.asarray(x_y[0][1])
                #     sum_y = []
                #     for i, label in enumerate(dic_element.keys()):
                #         y = np.asarray(x_y[i + 1][1])
                #         length = len(y)
                #         delta_y = np.abs(existing[-length:] - y).tolist()
                #         sum_y = sum_y + delta_y
                #
                #         # print('Out,Layer{},{}:{:.5f}'.format(layer, label, delta_y))
                #     mean_y = np.mean(sum_y)
                #     dic_content['out'][layer].append((key, mean_y))
                # print()

                for layer in ["0", "1", "2"]:  #
                    dic_layer_out["layer"] = layer
                    layer_delta = []
                    layer_std = []
                    for gi in range(1, num_growth):
                        x_y = distance_x_y(file_key=key, dic=dic_layer_out, gi=gi, is_out=is_out)
                        data_list = np.asarray(x_y[1])
                        delta = np.mean(data_list[:50]) - data_list[-1]
                        # std = np.std(data_list)
                        layer_delta.append(delta)
                        # layer_std.append(std)
                    mean = np.mean(layer_delta)
                    # delta_std = distance_x_y_std(file_key=key, dic=dic_layer_out, is_out=is_out)
                    _, std = x_y_std(file_key=key, dic=dic_layer_out, is_out=is_out)

                    dic_content['out'][layer]["delta"].append((key, mean))
                    # dic_content['out'][layer]["delta-std"].append((key, delta_std))
                    dic_content['out'][layer]["std"].append((key, std))
            else:

                # for layer in ["1", "2", "3"]:  #
                #     dic_layer_in["layer"] = layer
                #     x_y = gen_x_y(is_out=is_out)
                #     existing = np.asarray(x_y[0][1])
                #     sum_y = []
                #     for i, label in enumerate(dic_element.keys()):
                #         y = np.asarray(x_y[i + 1][1])
                #         length = len(y)
                #         delta_y = np.abs(existing[-length:] - y).tolist()
                #         sum_y = sum_y + delta_y
                #
                #         # print('Out,Layer{},{}:{:.5f}'.format(layer, label, delta_y))
                #     mean_y = np.mean(sum_y)
                #     dic_content['in'][layer].append((key, mean_y))
                for layer in ["1", "2", "3"]:
                    dic_layer_in["layer"] = layer
                    layer_delta = []
                    layer_std = []
                    for gi in range(1, num_growth):
                        x_y = distance_x_y(file_key=key, dic=dic_layer_in, gi=gi, is_out=is_out)
                        data_list = np.asarray(x_y[1])
                        delta = np.mean(data_list[:50]) - data_list[-1]
                        layer_delta.append(delta)
                    mean = np.mean(layer_delta)
                    # std = distance_x_y_std(file_key=key, dic=dic_layer_in, is_out=is_out)
                    _, std = x_y_std(file_key=key, dic=dic_layer_out, is_out=is_out)
                    dic_content['in'][layer]['delta'].append((key, mean))
                    dic_content['in'][layer]['std'].append((key, std))

        print()

    if not is_simple:
        for i_o in ["out", "in"]:
            for l in dic_content[i_o].keys():
                print('{} - L{} - Delta:'.format(i_o, l))
                list_c = dic_content[i_o][l]['delta']
                list_c.sort(key=lambda tup: tup[1], reverse=True)
                for mode, value in list_c:
                    print('{}: {:.8f}'.format(mode, value))
                print()
                print()

        for i_o in ["out", "in"]:
            for l in dic_content[i_o].keys():
                print('{} - L{} - STD:'.format(i_o, l))
                list_c = dic_content[i_o][l]['std']
                list_c.sort(key=lambda tup: tup[1])
                for mode, value in list_c:
                    print('{}: {:.8f}'.format(mode, value))
                print()
                print()

        # for i_o in ["out", "in"]:
        #     for l in dic_content[i_o].keys():
        #         print('{} - L{} - Delta-STD:'.format(i_o, l))
        #         list_c = dic_content[i_o][l]['delta-std']
        #         list_c.sort(key=lambda tup: tup[1])
        #         for mode, value in list_c:
        #             print('{}: {:.8f}'.format(mode, value))
        #         print()
        #         print()

    # for key in dic.keys():
    #     print(key, ':')
    #     print_acc(dic[key])
    #
    #     path_to_file = dic[key]
    #
    #     path = os.path.dirname(path_to_file)
    #
    #     files = [f for f in glob.glob(path + "**/*__json_in_out.json")]
    #
    #     filename = files[0]
    #
    #     with open(filename, 'r') as f:
    #         data = json.load(f)
    #     y = []
    #     for out in [True, False]:
    #         is_out = out
    #         if is_out:
    #             for layer in ["0", "1", "2"]:  #
    #                 dic_layer_out["layer"] = layer
    #                 x_y = gen_x_y(is_out=is_out)
    #                 existing = np.asarray(x_y[0][1])
    #                 for i, label in enumerate(['Grow_1', 'Grow_2', 'Grow_3', 'Grow_4']):
    #                     y = np.asarray(x_y[i + 1][1])
    #                     length = len(y)
    #                     delta_y = np.mean(np.abs(existing[-length:] - y))
    #
    #                     # print('Out,Layer{},{}:{:.5f}'.format(layer, label, delta_y))
    #                     dic_content['out'][layer][label].append((key, delta_y))
    #             print()
    #         else:
    #             for layer in ["1", "2", "3"]:  #
    #                 dic_layer_in["layer"] = layer
    #                 x_y = gen_x_y(is_out=is_out)
    #                 existing = np.asarray(x_y[0][1])
    #                 for i, label in enumerate(['Grow_1', 'Grow_2', 'Grow_3', 'Grow_4']):
    #                     y = np.asarray(x_y[i + 1][1])
    #                     length = len(y)
    #                     delta_y = np.mean(np.abs(existing[-length:] - y))
    #                     # print('In,Layer{},{:8s}:{:.5f}'.format(layer, label, delta_y))
    #                     dic_content['in'][layer][label].append((key, delta_y))
    #     print()
    #
    # for i_o in ["out", "in"]:
    #     for l in dic_content[i_o].keys():
    #         for label in dic_content[i_o][l].keys():
    #             print('{}, L{},{}'.format(i_o, l, label))
    #             list_c = dic_content[i_o][l][label]
    #             list_c.sort(key=lambda tup: tup[1])
    #             for mode, value in list_c:
    #                 print('{}: {:.8f}'.format(mode, value))
    #             print()
    #             print()
