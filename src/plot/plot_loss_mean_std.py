import torch
import os
import numpy as np
import matplotlib.pyplot as plt


# kwargs = {'num_classes': 6,
#           'num_channel': 8,
#           'out1': t_channel,
#           'out2': t_channel,
#           'k_size': (1, 3),  # 3->576, 7->288
#           'c_stride': (1, 1),h
#           'p_stride': (1, 2),
#           'fc1': 256,
#           'fc2': 300}
#
# model = CNN.Net(**kwargs)
# model.load_state_dict(torch.load(path_to_model))
#
# model = torch.load(path_to_model)
# conv1_weight = model.conv1.weight.data.numpy()
# print(conv1_weight)
# print(conv1_weight.shape)
# conv1_plot = np.reshape(conv1_weight, (t_channel ** 2, 1, t_kernel))
#
# plt.scatter(x, y, s=area, c=colors, alpha=0.5)
# plt.show()


# models = []
# in_channels = 8
#
# epochs_arr = [0, 3, 18]


# epochs is an array with length = 3
def plot_conv_weights(epochs, in_channels):
    fig, axs = plt.subplots(3)
    n_fig = 0
    for i in epochs:
        path_to_model_location = 'EMG_CNN_C24_K320200102-191749/model_epoch_{}.ckpt'.format(i)
        path_to_model = os.path.join(res_dir, path_to_model_location)
        model = torch.load(path_to_model)
        conv1_weight = model.conv2.weight.data.numpy()
        conv1_plot = np.reshape(conv1_weight, (in_channels, -1))
        for fm in conv1_plot:
            axs[n_fig].plot(fm)
        n_fig += 1
    plt.show()


def plot_fully_weights(epochs, in_channels, x, colors, markers):
    i_color = 0
    for i in epochs:
        path_to_model_location = 'EMG_CNN_C8_K320200102-145312/model_epoch_{}.ckpt'.format(i)
        path_to_model = os.path.join(res_dir, path_to_model_location)
        model = torch.load(path_to_model)
        fc_weight = model.fc2.weight.data.numpy()
        fc_plot = np.reshape(fc_weight, (in_channels), order='F')
        plt.scatter(x, fc_plot, s=5, c=colors[i_color], marker=markers[i_color])
        i_color += 1
    plt.show()


def loss_mean_std(log_loc, save_to_location):
    for txt_loc in log_loc:
        with open(txt_loc, 'rt') as f:
            data = f.readlines()
        for line in data:
            if line.__contains__('Batch_100'):
                log = open(save_to_location, a)
                log.write(line)
                log.close()


def mean_std():
    log_dir = '/results/loss_mean_std/mean_std_log.txt'
    log = open(log_dir, 'a')
    array = ['8', '12', '16', '20', '24', 'standard']
    for c in array:
        m = []
        s = []
        for e in range(1, 21):
            val = eval('mean_std2.C_{}_Epoch_{}_Batch_100'.format(c, e))
            mean = np.mean(val)
            std = np.std(val)
            m.append(mean)
            s.append(std)
        log.write('c{}_mean = {}\n'.format(c, m))
        log.write('c{}_std = {}\n'.format(c, s))
    log.close()


if __name__ == '__main__':
    loc_h_5 = '/Users/zber/ProgramDev/exp_pyTorch/results/EMG_grow_h_score_10_20_50_20200316-190005'
    loc_h_10 = '/Users/zber/ProgramDev/exp_pyTorch/results/MNIST_grow_h_score_10_20200316-173732'
    loc_h_15 = '/Users/zber/ProgramDev/exp_pyTorch/results/MNIST_grow_h_score_15_20200316-173613'
    loc_l2_5 = '/Users/zber/ProgramDev/exp_pyTorch/results/EMG_grow_l2_10_20_50_20200316-190050'
    loc_l2_10 = '/Users/zber/ProgramDev/exp_pyTorch/results/MNIST_grow_l2_10_20200316-172924'
    loc_l2_15 = '/Users/zber/ProgramDev/exp_pyTorch/results/MNIST_grow_l2_15_20200316-173310'
    alex_h = '/Users/zber/ProgramDev/exp_pyTorch/results/Alex_h_score/layers_gradient.npy'
    alex_l2 = '/Users/zber/ProgramDev/exp_pyTorch/results/Alex_l2/layers_gradient.npy'
    emg_h = '/Users/zber/ProgramDev/exp_pyTorch/results/EMG_grow_h_score_10_20_50_20200316-190005'
    emg_l2 = '/Users/zber/ProgramDev/exp_pyTorch/results/EMG_grow_l2_20200317-152918/layers_gradient.npy'
    emg_h = '/Users/zber/ProgramDev/exp_pyTorch/results/EMG_grow_H_score_20200316-232138/layers_gradient.npy'
    mnist_h_score = '/Users/zber/ProgramDev/exp_pyTorch/results/MNIST_grow_l2_20200317-154725/layers_h_score.npy'
    mnist_grow_l2 = '/Users/zber/ProgramDev/exp_pyTorch/results/MNIST_grow_l2_20200317-154725/layers_l2.npy'
    mnist_batch_loss = '/Users/zber/ProgramDev/exp_pyTorch/results/MNIST_grow_l2_20200317-154725/batch_loss.npy'
    mnist_19_l2 = '/Users/zber/ProgramDev/exp_pyTorch/results/MNIST_grow_l2_20200317-180413/layers_l2.npy'
    mnist_19_h_score = '/Users/zber/ProgramDev/exp_pyTorch/results/MNIST_grow_l2_20200317-181405/layers_h_score.npy'
    mnist_19_l2 = '/Users/zber/ProgramDev/exp_pyTorch/results/MNIST_grow_l2_20200317-181405/layers_l2.npy'
    mnist_19_batch_loss = '/Users/zber/ProgramDev/exp_pyTorch/results/MNIST_grow_l2_20200317-181405/batch_loss.np'
    minist_seed = '/Users/zber/ProgramDev/exp_pyTorch/results/MNIST_grow_l2_20200317-184533/layers_l2.npy'
    mnist_h = '/Users/zber/ProgramDev/exp_pyTorch/results/MNIST_grow_new_h_20200318-220033/layers_h_score.npy'
    mnist_new_h = '/Users/zber/ProgramDev/exp_pyTorch/results/MNIST_grow_new_h_20200318-225230/layers_h_score.npy'

    y = [1.8969427472352982, 1.778080599308014, 1.729607492685318, 1.7016785153746605, 1.6827078901131949,
     1.6688102243012852, 1.658258900472096, 1.6498083553214868, 1.6429001125582943, 1.6372364889184634, 1.6323625492869,
     1.6281411509050263, 1.6244667312884942, 1.621248304531688, 1.618370602382554, 1.6157581536347667,
     1.613385146610877, 1.6112201593319575, 1.6092198672629239, 1.6074101016024749, 1.6057309548533152,
     1.6041747708212246, 1.6027296933229418, 1.6013821390767893, 1.6001152249733608, 1.5989405487592403,
     1.5978231077282516, 1.5967905986947672, 1.5958086080455232, 1.5948849928114148]
    # parameters need to set
    epochs = 20
    batchs = 600
    layers = 4

    data_batch_loss = np.load(mnist_batch_loss)
    batch_loss = np.reshape(data_batch_loss, (-1))

    data = np.load(mnist_new_h)

    data = np.reshape(data, (epochs, layers, batchs))
    conv1_mean = np.reshape(np.mean(data[:, 0, :], axis=1), (-1))
    conv2_mean = np.reshape(np.mean(data[:, 1, :], axis=1), (-1))
    fc_mean = np.reshape(np.mean(data[:, 2, :], axis=1), (-1))

    conv1 = np.reshape(data[:, 0, :], (-1))
    conv2 = np.reshape(data[:, 1, :], (-1))
    fc = np.reshape(data[:, 2, :], (-1))

    # alex = np.load(alex_l2)
    # alex = np.reshape(alex, (5, 8, 500))

    # conv1 = np.reshape(alex[:, 0, :], (-1))
    # conv1_max = np.reshape(np.argmax(conv1), (1))
    # conv2 = np.reshape(alex[:, 1, :], (-1))
    # conv2_max = np.reshape(np.argmax(conv2), (1))
    # conv3 = np.reshape(alex[:, 2, :], (-1))
    # conv3_max = np.reshape(np.argmax(conv3), (1))
    # conv4 = np.reshape(alex[:, 3, :], (-1))
    # conv5 = np.reshape(alex[:, 4, :], (-1))
    # fc1 = np.reshape(alex[:, 5, :], (-1))
    # fc2 = np.reshape(alex[:, 6, :], (-1))
    # fc3 = np.reshape(alex[:, 7, :], (-1))

    # conv1_total = np.load(os.path.join(emg_l2, 'conv1_total.npy'))
    # conv2_total = np.load(os.path.join(emg_l2, 'conv2_total.npy'))
    # fc_total = np.load(os.path.join(emg_l2, 'fc_total.npy'))
    # c1 = np.reshape([conv1_total[i] for i in range(20)], (1, 590))
    # c1 = c1[0]
    # c2 = np.reshape([conv2_total[i] for i in range(5)], (1, 590))
    # c2 = c2[0]
    # fc1 = np.reshape([fc_total[i] for i in range(5)], (1, 590))
    # fc1 = fc1[0]

    a = [-0.02372000040486455, -0.009660000912845135, -0.060750002041459084, -0.019289998337626457, -0.033440001308918]

    b = [-0.012290000217035413, -0.024789996445178986, -0.04670000076293945, 0.001490000169724226, -0.0014100000262260437]
    x = np.arange(0,5)
    x1 = np.arange(0, epochs)
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(x, a, label='good')
    ax.plot(x, b, label='bad')
    # ax.plot(x, conv1_grad_mean_20, label='20%_conv1')
    # ax.plot(x, conv1_grad_mean_30, label='30%_conv1')
    # ax.plot(x, conv1_grad_mean_40, label='40%_conv1')
    # ax.plot(x, conv1_grad_mean_50, label='50%_conv1')
    # ax.plot(x, conv1_grad_mean_60, label='60%_conv1')
    # ax.plot(x, conv1_grad_mean_70, label='70%_conv1')
    # ax.plot(x, conv1_grad_mean_80, label='80%_conv1')
    # ax.plot(x, conv1_grad_mean_90, label='90%_conv1')
    # ax.plot(x, conv1_grad_mean_100, label='100%_conv1')

    # ax.plot(x, conv2_grad_mean_10, label='10%_conv2')
    # ax.plot(x, conv2_grad_mean_10, label='20%_conv2')
    # ax.plot(x, conv2_grad_mean_30, label='30%_conv2')
    # ax.plot(x, conv2_grad_mean_40, label='40%_conv2')
    # ax.plot(x, conv2_grad_mean_50, label='50%_conv2')
    # ax.plot(x, conv2_grad_mean_60, label='60%_conv2')
    # ax.plot(x, conv2_grad_mean_70, label='70%_conv2')
    # ax.plot(x, conv2_grad_mean_80, label='80%_conv2')
    # ax.plot(x, conv2_grad_mean_90, label='90%_conv2')
    # ax.plot(x, conv2_grad_mean_100, label='100%_conv2')

    # ax.plot(x, fc1_grad_mean_10, label='10%_fc1')
    # ax.plot(x, fc1_grad_mean_20, label='20%_fc1')
    # ax.plot(x, fc1_grad_mean_30, label='30%_fc1')
    # ax.plot(x, fc1_grad_mean_40, label='40%_fc1')
    # ax.plot(x, fc1_grad_mean_50, label='50%_fc1')
    # ax.plot(x, fc1_grad_mean_60, label='60%_fc1')
    # ax.plot(x, fc1_grad_mean_70, label='70%_fc1')
    # ax.plot(x, fc1_grad_mean_80, label='80%_fc1')
    # ax.plot(x, fc1_grad_mean_90, label='90%_fc1')
    # ax.plot(x, fc1_grad_mean_100, label='100%_fc1')
    # ax.plot(x, conv1_grad_mean, label='con1', marker='.')
    # ax.plot(x, conv2_grad_mean, label='con2', marker='.')
    #
    # ax.plot(x1, conv1_mean, label='c1_mean')
    # ax.plot(x1, conv2_mean, label='c2_mean')
    # ax.plot(x1, fc_mean, label='fc_mean')

    # ax.plot(x, conv1, label='c1')
    # ax.plot(x, conv2, label='c2')
    # ax.plot(x, fc, label='fc1')

    # ax.plot(x, batch_loss, label='loss')

    # ax.plot(x, c1, label='fc1')
    # ax.plot(x, c2, label='fc2')
    # ax.plot(x, fc1, label='fc3')

    ax.legend()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Gradient')
    # plt.xticks(np.arange(0, batchs * epochs, step=batchs), np.arange(0, epochs))
    plt.show()

"""
if __name__ == '__main__':
    mean = [0.06843648, 0.08067124, 0.07266106, 0.07712139, 0.06911331, 0.065912634, 0.063376695, 0.06709916,
            0.060152154, 0.062089033, 0.0637455, 0.05785312, 0.05370382, 0.04949081, 0.047913887, 0.044987604,
            0.03901023, 0.04051204, 0.035585966, 0.037096784]
    std = [0.16113718, 0.18532501, 0.2021417, 0.21206434, 0.22032146, 0.22731647, 0.23200996, 0.23608667, 0.24111068,
           0.24460573, 0.24782473, 0.25288516, 0.2563186, 0.2593256, 0.26228786, 0.26552522, 0.26899916, 0.2715774,
           0.27498424, 0.2766955]

    mean1 = [0.020946177, 0.010676813, 0.0058721933, 0.0041978774, 0.0037910342, 0.00036570404, -0.0027134905,
             -0.00012287719, -0.004120827, -0.0065825307, -0.0036826015, -0.0030005097, -0.007331864, -0.0099135,
             -0.008794166, -0.014046585, -0.013884955, -0.012787424, -0.011711882, -0.012944959]
    std1 = [0.11524946, 0.12711023, 0.13344416, 0.13871656, 0.14361015, 0.14666522, 0.15056589, 0.15401013, 0.15708135,
            0.16033265, 0.16370343, 0.1660621, 0.16825804, 0.17107008, 0.1775911, 0.1801614, 0.18292561, 0.18603194,
            0.1895064, 0.19052953]

    mean2 = [0.046715524, 0.04440918, 0.043146543, 0.0462507, 0.04378225, 0.0354776, 0.03057083, 0.029954253,
             0.02957255,
             0.026159301, 0.023012996, 0.02266285, 0.021738494, 0.018663753, 0.014856701, 0.012795077, 0.008862994,
             0.008324706, 0.010779845, 0.013865359]
    std2 = [0.12720191, 0.13638467, 0.14315341, 0.14661944, 0.15156038, 0.15754718, 0.16213939, 0.16968884, 0.17326152,
            0.17946121, 0.18459713, 0.18745008, 0.1913205, 0.19431351, 0.19888669, 0.20279595, 0.20515059, 0.2078465,
            0.21174201, 0.21275346]

    mean3 = [0.041618463, 0.03835969, 0.031268608, 0.019312698, 0.0155725265, 0.009461803, 0.0057371194, 0.009935248,
            0.0036355224, 0.0042693657, 0.0025028682, 0.0029259354, 0.0028006341, -0.0037072308, -0.00057414407,
            -0.002379968, -0.004573618, -0.0047585764, -0.006297678, -0.008768776]
    std3 = [0.106787995, 0.11932907, 0.1277132, 0.13700533, 0.14383173, 0.14944659, 0.15373552, 0.15726721, 0.16224536,
           0.16699417, 0.17022726, 0.17413418, 0.1782033, 0.18186578, 0.18558022, 0.18995857, 0.19140533, 0.19501266,
           0.1975793, 0.20036492]

    mean4 = [0.07933043, 0.0856617, 0.089531355, 0.09476099, 0.09045592, 0.08684247, 0.083911985, 0.0823975, 0.08123537,
            0.08205103, 0.08066932, 0.07299003, 0.06831393, 0.06306687, 0.0639327, 0.052690823, 0.05450894, 0.05517054,
            0.055134133, 0.050859466, 0.047393207, 0.04877215, 0.046281137, 0.04088269, 0.044352774, 0.04021485,
            0.042991873, 0.041305795, 0.037002116, 0.042962674, 0.03007979, 0.03470056, 0.03308264, 0.03693215,
            0.030468665, 0.035638846, 0.030913748, 0.025886795, 0.02562764, 0.026278723]
    std4 = [0.14692181, 0.17011659, 0.1848163, 0.19620606, 0.20569539, 0.21121845, 0.21394984, 0.21597695, 0.2200552,
           0.22461376, 0.22808911, 0.23127578, 0.2354718, 0.23778133, 0.2412507, 0.24287695, 0.2446567, 0.24780048,
           0.25075388, 0.25239167, 0.25551805, 0.2577055, 0.26090783, 0.26441026, 0.26537362, 0.2668876, 0.26991332,
           0.27176934, 0.27529132, 0.2763566, 0.28104958, 0.2830737, 0.28489643, 0.28780803, 0.29222313, 0.2936217,
           0.29462475, 0.29629022, 0.29910898, 0.30122894]
    mean5 = [0.07933043, 0.0856617, 0.089531355, 0.09476099, 0.09045592, 0.08684247, 0.083911985, 0.0823975, 0.08123537,
            0.08205103, 0.08066932, 0.07299003, 0.06831393, 0.06306687, 0.0639327, 0.052690823, 0.05450894, 0.05517054,
            0.055134133, 0.050859466]
    std5 = [0.14692181, 0.17011659, 0.1848163, 0.19620606, 0.20569539, 0.21121845, 0.21394984, 0.21597695, 0.2200552,
           0.22461376, 0.22808911, 0.23127578, 0.2354718, 0.23778133, 0.2412507, 0.24287695, 0.2446567, 0.24780048,
           0.25075388, 0.25239167]

    x = np.arange(20)
    x1 = np.arange(40)
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(x, mean, label='C8_mean')
    ax.plot(x, mean1, label='C24_mean')
    ax.plot(x, std, label='C8_std')
    ax.plot(x, std1, label='C24_std')
    ax.plot(x, mean2, label='C12_mean')
    ax.plot(x, std2, label='C12_std')
    ax.plot(x, mean3, label='C20_mean')
    ax.plot(x, std3, label='C20_std')
    ax.plot(x1, mean4, label='C8_mean_E40')
    ax.plot(x1, std4, label='C8_std_E40')
    ax.plot(x, mean5, label='C8_mean_E20')
    ax.plot(x, std5, label='C8_std_E20')
    ax.legend()
    plt.show()

    # plt.plot(mean, c='g')
    # plt.plot(std, c='r')
    # plt.plot(mean1, c='g')
    # plt.plot(std2, c='r')
    # plt.show()
"""

"""
if __name__ == '__mai__':
    # load model
    parent_dir = os.path.dirname(os.getcwd())
    res_dir = os.path.join(parent_dir, 'results/')
    path_to_model_location = 'EMG_CNN_C8_K320200102-145312/model_epoch_0.ckpt'
    path_to_model = os.path.join(res_dir, path_to_model_location)

    # settings
    t_channel = 8
    t_kernel = 3
    kwargs = {'num_classes': 6,
              'num_channel': 8,
              'out1': t_channel,
              'out2': t_channel,
              'k_size': (1, 3),  # 3->576, 7->288
              'c_stride': (1, 1),
              'p_stride': (1, 2),
              'fc1': 256,
              'fc2': 300}

    # x axis
    x = []
    i = 0.0
    while i < 180.0:
        i = float('%.1f' % i)
        x.append(i)
        i = i + 0.1

    epoch_array = [0, 10, 19]
    colors = ['r', 'g', 'b']
    markers = ['o', 'x', '+']
    in_channels = 24
    plot_conv_weights(epochs=epoch_array, in_channels=in_channels)
    # plot_fully_weights(epochs=epoch_array, in_channels=in_channels, x=x, colors=colors, markers=markers)
# print(model.conv2.weight.data)
# print(model.fc1.weight.data)
# print(model.fc2.weight.data)

# print('acv')
"""
