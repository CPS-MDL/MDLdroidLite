import torch

from ri_control.ri_controller import grab_input_weight_shape
from model.CNN import LeNet5_GROW_P, LeNet5
from model.VGG import VGG, VGG_BN
from model.summary import summary
from model.mobileNet import NetS
from model.TCN import TCN
from ptflops import get_model_complexity_info


def find_modules(model):
    modules = []
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            modules.append(module)
    return modules


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

    model_name = 'TCN'  # 'LeNet', 'VGG' , 'MobileNet' ,'TCN'

    method = ['ptflops']

    dataset = "OP"

    # method = 'ptflops' # 'ptflops'

    if model_name == 'LeNet':
        # LeNet

        if dataset == "Har":
            input_shape = (9, 1, 128)
            width = 128
            kwargs = {
                'in_channel': 9,
                'out1_channel': 36,
                'out2_channel': 72,
                'fc': 300,
                'out_classes': 6,
                'kernel_size': 64,
                'flatten_factor': 32,
                'padding': [31, 32]
            }
            net = LeNet5(**kwargs)

        elif dataset == "EMG":
            input_shape = (8, 1, 20)
            width = 20
            kwargs = {
                'in_channel': 8,
                'out1_channel': 36,
                'out2_channel': 72,
                'fc': 300,
                'out_classes': 6,
                'kernel_size': 9,
                'flatten_factor': 5,
                'padding': [4, 4]
            }

            net = LeNet5(**kwargs)

        elif dataset == "Fall":
            input_shape = (1, 1, 604)
            width = 604
            kwargs = {
                'in_channel': 1,
                'out1_channel': 36,
                'out2_channel': 72,
                'fc': 300,
                'out_classes': 8,
                'kernel_size': 42,
                'flatten_factor': 151,
                'padding': [21, 21]
            }

            net = LeNet5(**kwargs)

        elif dataset == "PAMA":
            input_shape = (9, 1, 512)
            width = 512
            kwargs = {
                'in_channel': 9,
                'out1_channel': 36,
                'out2_channel': 72,
                'fc': 300,
                'out_classes': 7,
                'kernel_size': 64,
                'flatten_factor': 128,
                'padding': [32, 32]
            }

            net = LeNet5(**kwargs)

        elif dataset == 'myHealth':
            width = 100
            input_shape = (23, 1, 100)
            kwargs = {
                'in_channel': 23,
                'out1_channel': 36,
                'out2_channel': 72,
                'fc': 300,
                'out_classes': 11,
                'kernel_size': 35,
                'flatten_factor': 25,
                'padding': [18, 17]
            }

            net = LeNet5(**kwargs)

        elif dataset == 'OP':
            width = 23
            input_shape = (77, 1, 23)
            kwargs = {
                'in_channel': 77,
                'out1_channel': 36,
                'out2_channel': 72,
                'fc': 300,
                'out_classes': 11,
                'kernel_size': 10,
                'flatten_factor': 23//4 +1,
                'padding': [5, 5]
            }

            net = LeNet5(**kwargs)

        elif dataset == 'FinDroid':
            width = 150
            input_shape = (6, 1, 150)
            kwargs = {
                'in_channel': 6,
                'out1_channel': 5,
                'out2_channel': 15,
                'fc': 39,
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

    elif model_name == 'VGG':
        # VGG
        input_shape = (3, 32, 32)
        net = VGG_BN('VGG11_bn_s1')
        # net = VGG('VGG11')
        # net = VGG('VGG11_s1')
        ins, outs, strides, groups, paddings = grab_input_weight_shape(net, input_shape)

    elif model_name == "TCN":
        # TCN
        # input_shape = (4, 100)
        # net = TCN(input_size=4, output_size=3, num_channels=[20, 20, 20], kernel_size=15, dropout=0.0)

        if dataset == "Har":
            input_shape = (9, 128)
            width = 128
            kwargs = {
                'in_channel': 9,
                'out1_channel': 10,
                'out2_channel': 30,
                'out3_channel': 48,
                'out4_channel': 158,
                'out_classes': 6,
                'kernel_size': 14,
                'avg_factor': 2
            }

        elif dataset == "EMG":
            input_shape = (8, 20)
            width = 20
            kwargs = {
                'in_channel': 8,
                'out1_channel': 20,
                'out2_channel': 35,
                'out3_channel': 40,
                'out4_channel': 52,
                'out_classes': 6,
                'kernel_size': 3,
                'avg_factor': 2
            }

        elif dataset == "Fall":
            input_shape = (1,  604)
            width = 604
            kwargs = {
                'in_channel': 1,
                'out1_channel': 20,
                'out2_channel': 35,
                'out3_channel': 40,
                'out4_channel': 52,
                'out_classes': 8,
                'kernel_size': 14,
                'avg_factor': 2
            }

            print('Fall')

        elif dataset == "PAMA":
            input_shape = (9, 512)
            width = 512

            kwargs = {
                'in_channel': 9,
                'out1_channel': 20,
                'out2_channel': 35,
                'out3_channel': 40,
                'out4_channel': 52,
                'out_classes': 7,
                'kernel_size': 14,
                'avg_factor': 2
            }

        elif dataset == 'myHealth':
            width = 100
            input_shape = (23, 100)

            kwargs = {
                'in_channel': 23,
                'out1_channel': 20,
                'out2_channel': 48,
                'out3_channel': 51,
                'out4_channel': 53,
                'out_classes': 11,
                'kernel_size': 12,
                'avg_factor': 2
            }
        elif dataset == "OP":
            width = 23
            input_shape = (77, 23)

            kwargs = {
                'in_channel': 77,
                'out1_channel': 20,
                'out2_channel': 48,
                'out3_channel': 51,
                'out4_channel': 53,
                'out_classes': 11,
                'kernel_size': 3,
                'avg_factor': 2
            }

        n_classes = kwargs['out_classes']
        input_channels = kwargs['in_channel']
        seq_length = width
        channel_sizes = [36, 36]
        kernel_size = kwargs['kernel_size']
        net = TCN(input_channels, n_classes, channel_sizes, kernel_size=kernel_size, dropout=0.0)

    elif model_name == 'MobileNet':
        if dataset == "Har":
            input_shape = (9, 1, 128)
            width = 128

            kwargs = {
                'in_channel': 9,
                'out1_channel': 10,
                'out2_channel': 30,
                'out3_channel': 48,
                'out4_channel': 158,
                'out_classes': 6,
                'kernel_size': 14,
                'avg_factor': 2
            }

            kwargs['out1_channel'] = 32
            kwargs['out2_channel'] = 64
            kwargs['out3_channel'] = 128
            kwargs['out4_channel'] = 256

            ff = obtain_ff_mb(kwargs, width)
            kwargs['avg_factor'] = ff
            net = NetS(**kwargs)

            # [32, 64, 128, 256] 1.14 M   51.72 k
            # [19, 34, 63, 126]  0.49 M   15.51 k
            # [10, 30, 48, 58]   0.28 M   12.77 k


        elif dataset == "EMG":
            input_shape = (8, 1, 20)
            width = 20
            kwargs = {
                'in_channel': 8,
                'out1_channel': 20,
                'out2_channel': 35,
                'out3_channel': 40,
                'out4_channel': 52,
                'out_classes': 6,
                'kernel_size': 3,
                'avg_factor': 2
            }
            kwargs['out1_channel'] = 32
            kwargs['out2_channel'] = 64
            kwargs['out3_channel'] = 128
            kwargs['out4_channel'] = 256

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

        elif dataset == "Fall":
            input_shape = (1, 1, 604)
            width = 604
            kwargs = {
                'in_channel': 1,
                'out1_channel': 20,
                'out2_channel': 35,
                'out3_channel': 40,
                'out4_channel': 52,
                'out_classes': 8,
                'kernel_size': 14,
                'avg_factor': 2
            }
            kwargs['out1_channel'] = 32
            kwargs['out2_channel'] = 64
            kwargs['out3_channel'] = 128
            kwargs['out4_channel'] = 256

            ff = obtain_ff_mb(kwargs, width)
            kwargs['avg_factor'] = ff
            # uniform
            # torch.manual_seed(7518)

            # normal
            # torch.manual_seed(9162)

            net = NetS(**kwargs)
            print('Fall')

        elif dataset == "PAMA":
            input_shape = (9, 1, 512)
            width = 512

            kwargs = {
                'in_channel': 9,
                'out1_channel': 20,
                'out2_channel': 35,
                'out3_channel': 40,
                'out4_channel': 52,
                'out_classes': 7,
                'kernel_size': 14,
                'avg_factor': 2
            }
            kwargs['out1_channel'] = 32
            kwargs['out2_channel'] = 64
            kwargs['out3_channel'] = 128
            kwargs['out4_channel'] = 256

            ff = obtain_ff_mb(kwargs, width)
            kwargs['avg_factor'] = ff
            net = NetS(**kwargs)
            print('PAMA')

        elif dataset == 'myHealth':
            width = 100
            input_shape = (23, 1, 100)

            kwargs = {
                'in_channel': 23,
                'out1_channel': 20,
                'out2_channel': 48,
                'out3_channel': 51,
                'out4_channel': 53,
                'out_classes': 11,
                'kernel_size': 12,
                'avg_factor': 2
            }

            kwargs['out1_channel'] = 32
            kwargs['out2_channel'] = 64
            kwargs['out3_channel'] = 128
            kwargs['out4_channel'] = 256
            ff = obtain_ff_mb(kwargs, width)
            kwargs['avg_factor'] = ff
            net = NetS(**kwargs)

            # [32, 64, 128, 256] 1.26 M  57.35 k
            # [23, 41, 77, 153]  0.78 M  25.61 k
            # [20, 48, 51, 53]   0.66 M  13.65 k

        elif dataset == 'OP':
            width = 23
            input_shape = (77, 1, 23)

            kwargs = {
                'in_channel': 77,
                'out1_channel': 20,
                'out2_channel': 48,
                'out3_channel': 51,
                'out4_channel': 53,
                'out_classes': 11,
                'kernel_size': 3,
                'avg_factor': 2
            }

            kwargs['out1_channel'] = 32
            kwargs['out2_channel'] = 64
            kwargs['out3_channel'] = 128
            kwargs['out4_channel'] = 256
            ff = obtain_ff_mb(kwargs, width)
            kwargs['avg_factor'] = ff
            net = NetS(**kwargs)


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
                'out1_channel': 10,
                'out2_channel': 20,
                'out3_channel': 61,
                'out4_channel': 54,
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
        print(f"Paras: {params}")
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))
        print('complete')
