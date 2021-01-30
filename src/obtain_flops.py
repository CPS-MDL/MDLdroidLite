import torch

from ri_control.ri_controller import grab_input_weight_shape
from model.CNN import LeNet5_GROW_P, LeNet5
from model.VGG import VGG, VGG_BN
from model.summary import summary
from model.mobileNet import NetS
from model.TCN import TCN
from ptflops import get_model_complexity_info


# input_shape = (3, 300, 300)  # Format:(channels, rows,cols)
# conv_filter = (64, 3, 3, 3)  # Format: (num_filters, channels, rows, cols)
# stride = 1
# padding = 1
# activation = 'relu'
#
# if conv_filter[1] == 0:
#     n = conv_filter[2] * conv_filter[3]  # vector_length
# else:
#     n = conv_filter[1] * conv_filter[2] * conv_filter[3]  # vector_length
#
# flops_per_instance = n + (n - 1)  # general defination for number of flops (n: multiplications and n-1: additions)
#
# num_instances_per_filter = ((input_shape[1] - conv_filter[2] + 2 * padding) / stride) + 1  # for rows
# num_instances_per_filter *= ((input_shape[1] - conv_filter[2] + 2 * padding) / stride) + 1  # multiplying with cols
#
# flops_per_filter = num_instances_per_filter * flops_per_instance
# total_flops_per_layer = flops_per_filter * conv_filter[0]  # multiply with number of filters
#
# if activation == 'relu':
#     # Here one can add number of flops required
#     # Relu takes 1 comparison and 1 multiplication
#     # Assuming for Relu: number of flops equal to length of input vector
#     total_flops_per_layer += conv_filter[0] * input_shape[1] * input_shape[2]
#
# if total_flops_per_layer / 1e9 > 1:  # for Giga Flops
#     print(total_flops_per_layer / 1e9, '{}'.format('GFlops'))
# else:
#     print(total_flops_per_layer / 1e6, '{}'.format('MFlops'))


def find_modules(model):
    modules = []
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            modules.append(module)
    return modules


def calculate_total_flops(model, input_shape=(1, 28, 28), ):
    # calculate total flops

    ins, outs, strides, groups, paddings = grab_input_weight_shape(model, input_shape)
    flops = 0

    for i, o, s, p, g in zip(ins, outs, strides, paddings, groups):
        flops += calculate_flops(i, o, s, p, g)

    return flops


def grab_input_weight_shape(model, input_shape=(1, 28, 28)):
    ins = []
    outs = []
    strides = []
    groups = []
    paddings = []
    s = summary(model, input_shape)
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


def calculate_flops(input_shape, conv_filter, stride=(1, 1), padding=(1, 1), group=1, activation='relu'):
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

    model_name = 'VGG'  # 'LeNet', 'VGG' , 'MobileNet'
    method = ['ours', 'ptflops']
    # method = ['ours']

    dataset = "FinDroid"

    # method = 'ptflops' # 'ptflops'

    if model_name == 'LeNet':
        # LeNet

        if dataset == "MNIST":
            # para_model = {'out1': 16, 'out2': 29, 'fc1': 50}
            # para_model = {'out1': 18, 'out2': 25, 'fc1': 43}
            # para_model = {'out1': 12, 'out2': 39, 'fc1': 42}
            # para_model = {'out1': 13, 'out2': 33, 'fc1': 50}
            # para_model = {'out1': 17, 'out2': 23, 'fc1': 41}
            # para_model = {'out1': 20, 'out2': 23, 'fc1': 41}
            # para_model = {'out1': 17, 'out2': 24, 'fc1': 42}

            # para_model = {'out1': 10, 'out2': 25, 'fc1': 40}

            # para_model = {'out1': 5, 'out2': 39, 'fc1': 51}
            # para_model = {'out1': 50, 'out2': 33, 'fc1': 13}
            # para_model = {'out1': 20, 'out2': 45, 'fc1': 269}
            para_model = {'out1': 5, 'out2': 25, 'fc1': 43}
            net = LeNet5_GROW_P(**para_model)
            input_shape = (1, 28, 28)
            # old update
            # [17, 23, 41] 1.82M, 25.79K
            # [17, 24, 42] 1.87 M, 27.27 k
            # [20, 23, 41] 2.13 M  27.59 k

            # [13, 33, 50] 1.84 M   38.06 k
            # [16, 29, 50] 2.04 M   35.8 k
            # v2 [5, 39, 51]  0.85 M   37.44 k
            # v2 [10, 25, 40] 1.15M  22.98k  98.78

        elif dataset == "Har":
            input_shape = (9, 1, 128)
            width = 128
            # kwargs = {
            #     'in_channel': 9,
            #     'out1_channel': 12,
            #     'out2_channel': 18,
            #     'fc': 50,
            #     'out_classes': 6,
            #     'kernel_size': 14,
            #     'flatten_factor': 5
            # }

            kwargs = {
                'in_channel': 9,
                'out1_channel': 5,
                'out2_channel': 16,
                'fc': 25,
                'out_classes': 6,
                'kernel_size': 14,
                'flatten_factor': 5
            }

            # [12, 18, 50, 6]  0.66 M  24.72 k
            # [10, 16, 33] 0.52 M 15.38 k
            # [5, 16, 25] 0.27M  10.75 k
            ff = obtain_ff(kwargs, width)
            kwargs['flatten_factor'] = ff
            net = LeNet5(**kwargs)

        elif dataset == "EMG":
            input_shape = (8, 1, 100)
            width = 100
            kwargs = {
                'in_channel': 8,
                'out1_channel': 17,
                'out2_channel': 38,
                'fc': 52,
                'out_classes': 6,
                'kernel_size': 14,
                'flatten_factor': 5
            }

            ff = obtain_ff(kwargs, width)
            kwargs['flatten_factor'] = ff
            # [20, 50, 500] 2.0 MMac 394.82 k
            # [16, 42, 261] 1.22 MMac 177.52 k
            # [12, 24, 25]  0.5 MMac 14.59 k
            # [17, 38, 52, 6] 0.94 M 41.01 k
            # uniform
            # torch.manual_seed(7518)

            # normal
            # torch.manual_seed(9162)

            net = LeNet5(**kwargs)

        elif dataset == 'myHealth':
            param = {
                'lr': 0.0005,
                'epoch': 20,
                'batch_size': 64,
                'is_bn': False,
            }
            width = 100
            input_shape = (23, 1, 100)
            kwargs = {
                'in_channel': 23,
                'out1_channel': 8,
                'out2_channel': 22,
                'fc': 48,
                'out_classes': 11,
                'kernel_size': 14,
                'flatten_factor': 5,
            }
            ff = obtain_ff(kwargs, width)
            kwargs['flatten_factor'] = ff
            net = LeNet5(**kwargs)

            # [20, 50, 500] 2.74 MMac  401.52 k
            # [18, 40, 112] 1.76 MMac  84.49 k
            # [7, 15, 25] 0.5 MMac  9.68 k
            # [8, 22, 48, 11] 0.63 MMac 21.5 k

        elif dataset == 'FinDroid':
            width = 150
            input_shape = (6, 1, 150)
            kwargs = {
                'in_channel': 6,
                'out1_channel':8,
                'out2_channel': 35,
                'fc': 54,
                'out_classes': 6,
                'kernel_size': 14,
                'flatten_factor': 5,
            }
            ff = obtain_ff(kwargs, width)
            kwargs['flatten_factor'] = ff
            net = LeNet5(**kwargs)

            # [20, 50, 500]  3.38 M   694.26 k

            # [15, 46, 132]  1.76 M   175.85 k
            # [5, 15, 39]    0.27 M   17.56 k
            # [12, 18, 50, 6] 0.65 M 18.86 k

    elif model_name == 'VGG':
        # VGG
        input_shape = (3, 32, 32)
        net = VGG_BN([18, 'M', 37, 'M', 77, 77, 'M', 86, 86, 'M', 86, 86, 'M'])
        #[15, 'M', 51, 'M', 70, 84, 'M', 109, 107, 'M', 89, 87, 'M']
        # net = VGG('VGG11')
        # net = VGG('VGG11_s1')
        ins, outs, strides, groups, paddings = grab_input_weight_shape(net, input_shape)
        # VGG11 full size: 306.42 MMac, 9.23 M
        # Prunning [15, 'M', 51, 'M', 70, 84, 'M', 109, 107, 'M', 89, 87, 'M']: 22.73 MMac, 0.43782 M
        # v2 [15, 'M', 30, 'M', 49, 49, 'M', 86, 86, 'M', 86, 86, 'M'] 11.96 MMac 279.24 k k
        # v1 [17, 'M', 30, 'M', 52, 52, 'M', 86, 84, 'M', 83, 80, 'M'] 12.76 MMac 273.46 k
        # v1  [18, 'M', 37, 'M', 77, 77, 'M', 86, 86, 'M', 86, 86, 'M'] 19.52 MMac 347.3 k


    elif model_name == "TCN":
        # TCN
        # input_shape = (4, 100)
        # net = TCN(input_size=4, output_size=3, num_channels=[20, 20, 20], kernel_size=15, dropout=0.0)

        n_classes = 10
        input_channels = 1
        seq_length = int(784 / input_channels)
        steps = 0
        channel_sizes = [25] * 8
        kernel_size = 7
        net = TCN(input_channels, n_classes, channel_sizes, kernel_size=kernel_size, dropout=0.0)
        input_shape = (1, 784)

        # net = TCN(input_size=9, output_size=6, num_channels=[36, 72, 144], kernel_size=14, dropout=0.0)
        # input_shape = (1, 784)
        # net = TCN(input_size=4, output_size=3, num_channels=[20, 20, 20,20, 20, 20,20, 20, 20], kernel_size=15, dropout=0.0)

    elif model_name == 'MobileNet':
        if dataset == "Har":
            param = {
                'lr': 0.0005,
                'epoch': 20,
                'batch_size': 64,
                'is_bn': False,
            }
            input_shape = (9, 1, 128)
            width = 128

            # kwargs = {
            #     'in_channel': 9,
            #     'out1_channel': 9,
            #     'out2_channel': 16,
            #     'out3_channel': 35,
            #     'out4_channel': 86,
            #     'out_classes': 6,
            #     'kernel_size': 14,
            #     'avg_factor': 2
            # }

            kwargs = {
                'in_channel': 9,
                'out1_channel': 20,
                'out2_channel': 26,
                'out3_channel': 91,
                'out4_channel': 126,
                'out_classes': 6,
                'kernel_size': 14,
                'avg_factor': 2
            }

            # kwargs['out1_channel'] = 32
            # kwargs['out2_channel'] = 64
            # kwargs['out3_channel'] = 128
            # kwargs['out4_channel'] = 256

            ff = obtain_ff_mb(kwargs, width)
            kwargs['avg_factor'] = ff
            net = NetS(**kwargs)

            # [32, 64, 128, 256] 1.14 M   51.72 k
            # [19, 34, 63, 126]  0.49 M   15.51 k
            # [10, 30, 48, 58]   0.26 M   7.37 k
            # [9, 16, 35, 86, 6] 0.2 M  6.21 k

        elif dataset == "EMG":
            input_shape = (8, 1, 100)
            width = 100
            kwargs = {
                'in_channel': 8,
                'out1_channel': 18,
                'out2_channel': 17,
                'out3_channel': 38,
                'out4_channel': 47,
                'out_classes': 6,
                'kernel_size': 12,
                'avg_factor': 2
            }
            # kwargs['out1_channel'] = 32
            # kwargs['out2_channel'] = 64
            # kwargs['out3_channel'] = 128
            # kwargs['out4_channel'] = 256

            ff = obtain_ff_mb(kwargs, width)
            kwargs['avg_factor'] = ff
            # uniform
            # torch.manual_seed(7518)

            # normal
            # torch.manual_seed(9162)

            net = NetS(**kwargs)
            print('EMG')
            # model = NetS()

            # [32, 64, 128, 256] 0.74M  50.31 k
            # [24, 38, 34, 68] 0.34 M 8.39 k
            # [20, 35, 40, 52] 0.29 M  7.56 k
            # [18, 17, 38, 47, 6] 0.22 MMac 5.63 k
        elif dataset == 'myHealth':
            width = 100
            input_shape = (23, 1, 100)

            kwargs = {
                'in_channel': 23,
                'out1_channel': 20,
                'out2_channel': 26,
                'out3_channel': 43,
                'out4_channel': 91,
                'out_classes': 11,
                'kernel_size': 12,
                'avg_factor': 2
            }

            # kwargs['out1_channel'] = 32
            # kwargs['out2_channel'] = 64
            # kwargs['out3_channel'] = 128
            # kwargs['out4_channel'] = 256
            ff = obtain_ff_mb(kwargs, width)
            kwargs['avg_factor'] = ff
            net = NetS(**kwargs)

            # [32, 64, 128, 256] 1.26 M  57.35 k
            # [23, 41, 77, 153]  0.78 M  25.61 k
            # [20, 48, 51, 53]   0.66 M  13.65 k
            # [20, 26, 43, 91 ,11] 0.6 M 13.15 k

        elif dataset == 'FinDroid':
            param = {
                'lr': 0.0005,
                'epoch': 20,
                'batch_size': 64,
                'is_bn': False,
            }

            width = 150
            input_shape = (6, 1, 150)

            kwargs = {
                'in_channel': 6,
                'out1_channel': 8,
                'out2_channel': 14,
                'out3_channel': 35,
                'out4_channel': 41,
                'out_classes': 6,
                'kernel_size': 14,
                'avg_factor': 2
            }

            # kwargs['out1_channel'] = 32
            # kwargs['out2_channel'] = 64
            # kwargs['out3_channel'] = 128
            # kwargs['out4_channel'] = 256

            ff = obtain_ff_mb(kwargs, width)
            kwargs['avg_factor'] = ff

            net = NetS(**kwargs)
            full_size = [10, 20, 128, 256]

            # [32, 64, 128, 256] 50.37k 1.41 M
            # [28, 52, 101, 202] 33.21k 1.03 M
            # [10, 20, 61, 54] 7.16k 0.27 M
            # [16, 17, 34, 83, 6] 6.46 k  0.32 M
            # [8, 14, 35, 41, 6]

    # start calculation
    if 'ours' in method:
        # calculate flops
        ins, outs, strides, groups, paddings = grab_input_weight_shape(net, input_shape)
        flops = 0

        for i, o, s, p, g in zip(ins, outs, strides, paddings, groups):
            flops += calculate_flops(i, o, s, p, g)

        print(flops)
    if 'ptflops' in method:
        macs, params = get_model_complexity_info(net, input_shape, as_strings=True,
                                                 print_per_layer_stat=True, verbose=True)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))
        print('complete')
