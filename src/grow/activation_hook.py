from utils import find_layers
import torch
import copy

activation = {}
pre_acv = {}
act_hook_dic = {}


def save_acv():
    global pre_acv
    global activation
    pre_acv = copy.deepcopy(activation)


def get_activation(name):
    def hook(self, input, output):
        global activation
        activation[name] = {}
        if output.dim() > 2:
            acv1 = torch.norm(output, p=1, dim=(2, 3))
        else:
            # acv1 = torch.norm(output, p=1, dim=1, keepdim=True)
            acv1 = torch.abs(output)

        acv1 = torch.mean(acv1, dim=0)
        activation[name]["L1"] = acv1.tolist()

        if output.dim() > 2:
            acv2 = torch.norm(output, p=2, dim=(2, 3))
        else:
            # acv2 = torch.norm(output, p=2, dim=(0)) / output.shape[0]
            acv2 = torch.abs(output)

        acv2 = torch.mean(acv2, dim=0)
        activation[name]["L2"] = acv2.tolist()

    return hook


def register_acv_hook(model):
    global act_hook_dic
    layers = find_layers(model)[:-1]
    for i, module in enumerate(layers):
        act_hook_dic[str(i)] = module.register_forward_hook(get_activation(name=str(i)))
    return act_hook_dic


def retrun_acv_dic():
    global activation
    return activation


def return_preacv_dic():
    global pre_acv
    return pre_acv


def remove_acv_hook():
    global act_hook_dic
    for key in act_hook_dic.keys():
        act_hook_dic[key].remove()
