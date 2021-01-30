import torch
import torch.nn as nn
import torch.nn.functional as F

hook_tensor = torch.zeros(1, 5, 1, 1)


class LeNet5_GROW(nn.Module):
    def __init__(self):
        super(LeNet5_GROW, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=5, stride=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(2, 5, kernel_size=5, stride=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(5 * 16, 10, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(10, 10, bias=False),
        )

    def forward(self, x):
        x = self.features(x)
        out1 = x.view(x.size(0), -1)
        out = self.classifier(out1)
        # out2 = F.softmax(x, dim=1)
        return out


def hook_maker(old_shape, new_shape, dim=4):
    t_old = torch.ones(old_shape)
    t_new = torch.empty(new_shape).fill_(0.5)
    t_new[:old_shape] = t_old
    if dim == 4:
        t_new = torch.reshape(t_new, (1, -1, 1, 1))
        t_new = t_new.requires_grad_(False)
    else:
        t_new = torch.reshape(t_new, (1, -1))
        t_new = t_new.requires_grad_(False)

    def hook(self, input, output):
        # input is a tuple of packed inputs
        # output is a Tensor. output.data is the Tensor we are interested
        print('output: ', output)
        output.mul_(t_new)
        print('output after: ', output)

    return hook


def hk(self, input, output):
    global hook_tensor
    # input is a tuple of packed inputs
    # output is a Tensor. output.data is the Tensor we are interested
    print('output: ', output)
    output.mul_(hook_tensor)
    print('output after: ', output)


if __name__ == '__main__':
    # torch.manual_seed(155)
    model = LeNet5_GROW()
    tensor = torch.zeros(1, 5, 1, 1, requires_grad=False)
    hook_fn = hook_maker(3, 5)

    # hook_fn1 = hook_maker(tensor)
    hook = {}
    for name, module in model.named_modules():
        if name == 'features.3':
            hook[name] = module.register_forward_hook(hk)
    r = torch.ones(5, 1, 28, 28)
    output = model(r)
    hook_tensor = torch.ones(1, 5, 1, 1)
    output = model(r)

    for key in hook.keys():
        hook[key].remove()

    print('')
