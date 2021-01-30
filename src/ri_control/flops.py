import torch

from model.summary import summary
from model.CNN import LeNet5, LeNet5_GROW_P
from model.mobileNet import NetS
from model.VGG import VGG, VGG_BN
from utils import device

dataset = ""


def set_dataset(set="mnist"):
    global dataset
    dataset = set


def create_model(input_shape, channels=[], model_name="LeNet"):
    global dataset

    if model_name == 'LeNet':
        # LeNet

        if dataset == "MNIST":
            para_model = {'out1': channels[0], 'out2': channels[1], 'fc1': channels[2]}
            net = LeNet5_GROW_P(**para_model)

        elif dataset == "Har":
            width = 128
            kwargs = {
                'in_channel': 9,
                'out1_channel': channels[0],
                'out2_channel': channels[1],
                'fc': channels[2],
                'out_classes': 6,
                'kernel_size': 14,
                'flatten_factor': 5
            }
            ff = obtain_ff(kwargs, width)
            kwargs['flatten_factor'] = ff
            net = LeNet5(**kwargs)

        elif dataset == "EMG":
            input_shape = (8, 1, 100)
            width = 100
            kwargs = {
                'in_channel': 8,
                'out1_channel': channels[0],
                'out2_channel': channels[1],
                'fc': channels[2],
                'out_classes': 6,
                'kernel_size': 14,
                'flatten_factor': 5
            }

            ff = obtain_ff(kwargs, width)
            kwargs['flatten_factor'] = ff
            net = LeNet5(**kwargs)

        elif dataset == 'myHealth':
            width = 100
            input_shape = (23, 1, 100)
            kwargs = {
                'in_channel': 23,
                'out1_channel': channels[0],
                'out2_channel': channels[1],
                'fc': channels[2],
                'out_classes': 11,
                'kernel_size': 14,
                'flatten_factor': 5,
            }

            ff = obtain_ff(kwargs, width)
            kwargs['flatten_factor'] = ff
            net = LeNet5(**kwargs)

        elif dataset in ['FinDroid', 'Fin_Sitting', 'Fin_SitRun']:
            width = 150
            input_shape = (6, 1, 150)
            kwargs = {
                'in_channel': 6,
                'out1_channel': channels[0],
                'out2_channel': channels[1],
                'fc': channels[2],
                'out_classes': 6,
                'kernel_size': 14,
                'flatten_factor': 5,
            }
            ff = obtain_ff(kwargs, width)
            kwargs['flatten_factor'] = ff
            net = LeNet5(**kwargs)

    elif model_name == 'VGG':
        vgg_list = [channels[0], 'M', channels[1], 'M', channels[2], channels[3], 'M', channels[4], channels[5], 'M', channels[6], channels[7], 'M']
        net = VGG_BN(vgg_list)

    elif model_name == 'MobileNet':
        if dataset == "Har":
            width = 128

            kwargs = {
                'in_channel': 9,
                'out1_channel': channels[0],
                'out2_channel': channels[1],
                'out3_channel': channels[2],
                'out4_channel': channels[3],
                'out_classes': 6,
                'kernel_size': 14,
                'avg_factor': 2
            }

            ff = obtain_ff_mb(kwargs, width)
            kwargs['avg_factor'] = ff
            net = NetS(**kwargs)

        elif dataset == "EMG":
            width = 100
            kwargs = {
                'in_channel': 8,
                'out1_channel': channels[0],
                'out2_channel': channels[1],
                'out3_channel': channels[2],
                'out4_channel': channels[3],
                'out_classes': 6,
                'kernel_size': 12,
                'avg_factor': 2
            }

            ff = obtain_ff_mb(kwargs, width)
            kwargs['avg_factor'] = ff
            net = NetS(**kwargs)

        elif dataset == 'myHealth':
            width = 100
            kwargs = {
                'in_channel': 23,
                'out1_channel': channels[0],
                'out2_channel': channels[1],
                'out3_channel': channels[2],
                'out4_channel': channels[3],
                'out_classes': 11,
                'kernel_size': 12,
                'avg_factor': 2
            }
            ff = obtain_ff_mb(kwargs, width)
            kwargs['avg_factor'] = ff
            net = NetS(**kwargs)

        elif dataset in ['FinDroid', 'Fin_Sitting', 'Fin_SitRun']:

            width = 150

            kwargs = {
                'in_channel': 6,
                'out1_channel': channels[0],
                'out2_channel': channels[1],
                'out3_channel': channels[2],
                'out4_channel': channels[3],
                'out_classes': 6,
                'kernel_size': 14,
                'avg_factor': 2
            }

            ff = obtain_ff_mb(kwargs, width)
            kwargs['avg_factor'] = ff

            net = NetS(**kwargs)

    return net


def find_modules(model):
    modules = []
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            modules.append(module)
    return modules


def grab_input_weight_shape(model_name, channels, input_shape=(1, 28, 28)):
    ins = []
    outs = []
    strides = []
    groups = []
    paddings = []

    # create model
    model = create_model(input_shape=input_shape, channels=channels, model_name=model_name)
    model = model.to(device)
    s = summary(model, tuple(input_shape))
    layers = find_modules(model)
    for key in s.keys():
        if key.startswith('Conv2d') or key.startswith('Linear'):
            ins.append(s[key]['input_shape'][1:])

    for module in layers:
        outs.append(module.weight.data.shape)
        if isinstance(module, torch.nn.Conv2d):
            strides.append(module.stride)
            groups.append(module.groups)
            paddings.append(module.padding)
        else:
            strides.append((0, 0))
            groups.append(0)
            paddings.append((0, 0))
    return ins, outs, strides, groups, paddings


def calculate_flops(input_shape, conv_filter, stride=1, padding=1, mode='conv'):
    if len(input_shape) == 1:
        return 2 * input_shape[0] * conv_filter[0]
    n = conv_filter[1] * conv_filter[2] * conv_filter[3]
    flops_per_instance = n + (n - 1)
    num_instances_per_filter = ((input_shape[1] - conv_filter[2] + 2 * padding) / stride) + 1  # for rows
    num_instances_per_filter *= ((input_shape[1] - conv_filter[2] + 2 * padding) / stride) + 1  # multiplying with cols
    flops_per_filter = num_instances_per_filter * flops_per_instance
    total_flops_per_layer = flops_per_filter * conv_filter[0]

    # multiply with number of filters
    # if total_flops_per_layer / 1e9 > 1:  # for Giga Flops
    #     print(total_flops_per_layer / 1e9, '{}'.format('GFlops'))
    # else:
    #     print(total_flops_per_layer / 1e6, '{}'.format('MFlops'))
    return total_flops_per_layer


def calculate_flops_v2(input_shape, conv_filter, stride=(1, 1), padding=(1, 1), group=1, activation='relu'):
    if len(input_shape) == 1:
        return 2 * input_shape[0] * conv_filter[0]
        # return 0
    n = conv_filter[1] * conv_filter[2] * conv_filter[3]
    flops_per_instance = n + (n - 1)
    num_instances_per_filter = ((input_shape[1] - conv_filter[2] + 2 * padding[0]) / stride[0]) + 1  # for rows
    num_instances_per_filter *= ((input_shape[2] - conv_filter[3] + 2 * padding[1]) / stride[1]) + 1  # multiplying with cols
    flops_per_filter = num_instances_per_filter * flops_per_instance
    total_flops_per_layer = flops_per_filter * (conv_filter[0] // group)  # multiply with number of filters

    if activation == 'relu':
        # Here one can add number of flops required
        # Relu takes 1 comparison and 1 multiplication
        # Assuming for Relu: number of flops equal to length of input vector
        total_flops_per_layer += conv_filter[0] * input_shape[1] * input_shape[2]
    # if total_flops_per_layer / 1e9 > 1:  # for Giga Flops
    #     print(total_flops_per_layer / 1e9, '{}'.format('GFlops'))
    # else:
    #     print(total_flops_per_layer / 1e6, '{}'.format('MFlops'))
    return total_flops_per_layer


def calculate_total_flops(model_name, channels, input_shape):
    # calculate total flops

    # get all info needed
    ins, outs, strides, groups, paddings = grab_input_weight_shape(model_name, channels, input_shape)
    flops = 0

    for i, o, s, p, g in zip(ins, outs, strides, paddings, groups):
        flops += calculate_flops_v2(i, o, s, p, g)

    return flops


def obtain_ff(dic, width=128):
    if 'padding' in dic.keys():
        after_first = (width + dic['padding'][1] * 2 - dic['kernel_size'] + 1) // 2
        ff = (after_first + dic['padding'][1] * 2 - dic['kernel_size'] + 1) // 2
    else:
        after_first = (width - dic['kernel_size'] + 1) // 2
        ff = (after_first - dic['kernel_size'] + 1) // 2
    return ff


def obtain_ff_mb(dic, width=128):
    strides = [2, 1, 2, 2]
    for i in range(4):
        width = (width - dic['kernel_size']) // strides[i] + 1
    return width


if __name__ == "__main__":
    input_shape = (5, 1, 3)  # Format:(channels, rows,cols)

    conv_filter = (3, 5, 1, 3)
    print(calculate_flops(input_shape, conv_filter))

    # para_model = {'out1': 20, 'out2': 50, 'fc1': 500}
    #
    # model = LeNet5_GROW_P(**para_model)
    #
    # ins, outs = grab_input_weight_shape(model, (1, 28, 28))
    # #
    # flops = 0
    # for i, o in zip(ins, outs):
    #     flops += calculate_flops(i, o)
    # #
    # print(flops)
