import torch
import matplotlib.pyplot as plt
import os
import numpy as np
from utils import find_layers
from model.CNN import LeNet5_GROW_STD


def gen_hist_from_dic(dic, path='', epoch=0):
    for layer in dic['before'].keys():
        bins = np.arange(0, 0.8, 0.02)
        fig, ax = plt.subplots(nrows=1, ncols=1)
        plt.xlim([0, 0.8])
        color = ['b', 'r', 'k']

        for key, c in zip(dic.keys(), color):
            ax.hist(dic[key][layer], bins=bins, alpha=0.3, label=key, color=c, lw=1, edgecolor='black')

        plt.title('L_{}'.format(layer))
        plt.xlabel('weight range')
        plt.ylabel('count')
        plt.legend(loc='upper right')
        pic_path = os.path.join(path, 'E_{}L{}.pdf'.format(epoch, layer))
        fig.savefig(pic_path)
        plt.close(fig)


def plot_hist(path_to, l_index='0', bins=20):
    m = torch.load(path_to, map_location=torch.device('cpu'))
    # layer = m.classifier._modules['0']
    layer = m.features._modules[l_index]
    weight = layer.weight.data.cpu().numpy()
    f_weight = np.reshape(weight, (-1)).tolist()
    plt.hist(f_weight, bins)
    plt.show()


def save_hist(module, dir, name, bins=20):
    data = module.weight.data.cpu().numpy()
    data = np.reshape(data, (-1))
    path = os.path.join(dir, name)
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.hist(data, bins)
    fig.savefig(path)
    plt.close(fig)


# dic = {'before':, 'after', 'finish' }
def save_weight_to_dic(layers, hist_dic, key='before'):
    dic = {}
    for index, layer in enumerate(layers):
        weight = layer.weight.data.cpu().flatten().numpy()
        weight = np.reshape(np.abs(weight), (-1))
        dic[index] = weight.tolist()
    hist_dic[key] = dic


def save_hist(fig, path):
    fig.savefig(path)
    plt.close(fig)


if __name__ == '__main__':
    model_path = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_standard_seed/model_grow_standard_seed_.ckpt"
    STD_REG_PATH = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_std_dense_reg/model_grow_rank_cumulative_.ckpt"
    model_s_path = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_standard_20_50_500/model_grow_standard_20_50_500_.ckpt"

    path = os.path.dirname(model_s_path)

    model = LeNet5_GROW_STD()
    # model = torch.load(model_path)
    model.load_state_dict(torch.load(model_s_path, map_location=torch.device('cpu')))
    layers = find_layers(model)
    hist_dic = {}
    save_weight_to_dic(layers, hist_dic, key='before')
    gen_hist_from_dic(hist_dic, path=path, epoch=0)


    # standard = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_standard_og2_new2_20200327-220602/model_grow_standard_og2_new2_.ckpt"
    # rand = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_randomMap_og2_new2_20200327-232518/model_grow_randomMap_og2_new2_.ckpt"
    # ours = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rankgroup_og2_new2_20200327-232242/model_grow_rankgroup_og2_new2_.ckpt"
    # rank = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rankconnect_og2_new2_20200327-232005/model_grow_rankconnect_og2_new2_.ckpt"
    # bridging = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_bridging_og2_new2_20200327-232754/model_grow_bridging_og2_new2_.ckpt"
    # standard_bn10 = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_standard_og2_new2_BN_20200328-150430/model_grow_standard_og2_new2_BN_.ckpt"
    # standard_bn50 = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_standard_og2_new2_BN_FC50/model_grow_standard_og2_new2_BN_.ckpt"
    # ours_bn = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_rankgroup_og2_new2_BN/model_grow_rankgroup_og2_new2_BN_.ckpt"
    # bridging_03 = "/Users/zber/ProgramDev/exp_pyTorch/results/grow_bridging_og2_new2_03/model_grow_bridging_og2_new2_.ckpt"
    # standard, rand, ours, rank,
    # for m in [standard]:
    #     plot_hist(m, bins=50)
    # a = np.random.normal(0, 0.8, 1000)
    # dic = {
    #     '0': {'a': a}
    #
    # }
    # gen_hist_from_dic(dic)
