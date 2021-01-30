from model.CNN import LeNet5_GROW_P
from torch.nn import init
import math
import torch


def reset_parameters(self):
    init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    if self.bias is not None:
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.bias, -bound, bound)


if __name__ == '__main__':
    para_list = [
        {'out1': 2, 'out2': 5, 'fc1': 10},
        {'out1': 4, 'out2': 10, 'fc1': 20},
        {'out1': 6, 'out2': 11, 'fc1': 30},
        {'out1': 8, 'out2': 14, 'fc1': 40},
        {'out1': 10, 'out2': 17, 'fc1': 50},
    ]

    for para in para_list:
        model = LeNet5_GROW_P(**para)
        mode = 'fan_in'
        nonlinearity = 'leaky_relu'
        print('Model_{}_{}_{}'.format(para['out1'], para['out2'], para['fc1']))
        i = 0
        for module in model.features.children():
            torch.manual_seed(307)
            classname = module.__class__.__name__
            if classname.find('Conv') != -1:
                i = i + 1
                tensor = module.weight
                print('Conv{} Bound:'.format(i))
                a = math.sqrt(5)
                fan = init._calculate_correct_fan(tensor, mode)
                gain = init.calculate_gain(nonlinearity, a)
                std = gain / math.sqrt(fan)
                weight_bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
                init.uniform_(module.weight, -weight_bound, weight_bound)
                weight_sd = torch.std(module.weight)

                if module.bias is not None:
                    fan_in, _ = init._calculate_fan_in_and_fan_out(tensor)
                    bias_bound = 1 / math.sqrt(fan_in)
                    init.uniform_(module.bias, -bias_bound, bias_bound)
                    bias_sd = torch.std(module.bias)

                print('{:1.5f}|{:1.5f}|{:1.5f}|{:1.5f}|'.format(weight_bound, weight_sd, bias_bound, bias_sd))

        for module in model.classifier.children():
            torch.manual_seed(1535)
            classname = module.__class__.__name__
            if classname.find('Linear') != -1:
                i = i + 1
                tensor = module.weight
                print('linear{} Bound:'.format(i))
                a = math.sqrt(5)
                fan = init._calculate_correct_fan(tensor, mode)
                gain = init.calculate_gain(nonlinearity, a)
                std = gain / math.sqrt(fan)
                weight_bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
                init.uniform_(module.weight, -weight_bound, weight_bound)
                weight_sd = torch.std(module.weight)

                if module.bias is not None:
                    fan_in, _ = init._calculate_fan_in_and_fan_out(tensor)
                    bias_bound = 1 / math.sqrt(fan_in)
                    init.uniform_(module.bias, -bias_bound, bias_bound)
                    bias_sd = torch.std(module.bias)

                print('{:1.5f}|{:1.5f}|{:1.5f}|{:1.5f}|'.format(weight_bound, weight_sd, bias_bound, bias_sd))
        print()
        print()
