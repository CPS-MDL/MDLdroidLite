"""
Pruning a ConvNet by filters iteratively
"""

import torch
import torch.nn as nn

from prune.methods import filter_prune
from model.models import LeNet
from main import generate_data_loader
from utils import dir_path, test, write_log


if __name__ == '__main__':
    # Hyper Parameters
    param = {
        'pruning_perc': 3.,
        'batch_size': 100,
        'test_batch_size': 100,
        'num_epochs': 5,
        'learning_rate': 0.0005,
        'weight_decay': 5e-4,
    }

    # Data loaders
    """
    train_dataset = datasets.MNIST(root='../data/', train=True, download=True,
                                   transform=transforms.ToTensor())
    loader_train = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=param['batch_size'], shuffle=True)
    
    test_dataset = datasets.MNIST(root='../data/', train=False, download=True,
                                  transform=transforms.ToTensor())
    loader_test = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=param['test_batch_size'], shuffle=True)
    """
    loader_train, loader_test = generate_data_loader(batch_size=param['batch_size'], dataset='MNIST')

    # create model
    model = LeNet()

    path = dir_path('Prune_LeNet5')

    # train model

    # train_test(model, trainloader=loader_train, testloader=loader_test, learning_rate=param['learning_rate'],
    #            epoch=param['num_epochs'], dic_path=path,str_channel='LeNet')

    # criterion
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=param['learning_rate'])

    # Load the pretrained model path_to_model = /Users/zber/ProgramDev/exp_pyTorch/results/Prune_LeNet520200207-000948/model.ckpt
    net = LeNet()
    net.load_state_dict(torch.load('/Users/zber/ProgramDev/exp_pyTorch/results/Prune_LeNet520200207-000948/model.ckpt'))
    if torch.cuda.is_available():
        print('CUDA enabled.')
        net.cuda()
    print("--- Pretrained network loaded ---")
    write_log("--- Pretrained network test ---",path['path_to_log'])
    test(net, loader_test, criterion, path['path_to_log'])

    # prune the weights
    masks = filter_prune(net, param['pruning_perc'])
    net.set_masks(masks)
    print("--- {}% parameters pruned ---".format(param['pruning_perc']))
    write_log('--- {}% parameters pruned and test---'.format(param['pruning_perc']),path['path_to_log'])
    test(net, loader_test, criterion, path['path_to_log'])


"""
# Retraining
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(net.parameters(), lr=param['learning_rate'],
                                weight_decay=param['weight_decay'])

train(net, criterion, optimizer, param, loader_train)

# Check accuracy and nonzeros weights in each layer
print("--- After retraining ---")
test(net, loader_test)
prune_rate(net)

# Save and load the entire model
torch.save(net.state_dict(), 'models/convnet_pruned.pkl')
"""