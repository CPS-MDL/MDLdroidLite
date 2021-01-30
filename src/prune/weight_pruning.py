"""
Pruning a MLP by weights with one shot
"""

import torch
import torch.nn as nn

from prune.methods import weight_prune
from model.models import LeNet_prune
from main import generate_data_loader, train_test
from utils import dir_path, test

# Hyper Parameters
param = {
    'pruning_perc': 90.,
    'batch_size': 100,
    'test_batch_size': 100,
    'num_epochs': 20,
    'learning_rate': 0.0005,
}

# Data loaders
dataset = 'MNIST'  # CIFAR10,MNIST,EMG
criterion = nn.CrossEntropyLoss()

loader_train, loader_test = generate_data_loader(batch_size=param['batch_size'], dataset=dataset)

# Create model
model = LeNet_prune()

# Create dir and get path
path = dir_path('LeNet_prune')

# Train model
train_test(model, loader_train, loader_test, param['learning_rate'], param['num_epochs'], path, 'LeNet_prune')

# or load the pretrained model
# Load the pretrained model
# model.load_state_dict(
#     torch.load('/Users/zber/ProgramDev/exp_pyTorch/results/LeNet_prune20200213-012555/model_LeNet_prune_FC8.ckpt'))
if torch.cuda.is_available():
    print('CUDA ensabled.')
    model.cuda()
print("--- Pretrained network loaded ---")
test(model, loader_test, criterion, path['path_to_log'])

# prune the weights
pruning_perc = 10.
while True:
    masks = weight_prune(model, pruning_perc)
    model.set_masks(masks)
    print("--- {}% parameters pruned ---".format(pruning_perc))
    _, ac = test(model, loader_test, criterion, path['path_to_log'])
    if ac < 98.5:
        break
    model.load_state_dict(
        torch.load('/Users/zber/ProgramDev/exp_pyTorch/results/LeNet_prune20200213-012555/model_LeNet_prune_FC8.ckpt'))
    if torch.cuda.is_available():
        print('CUDA ensabled.')
        model.cuda()
    pruning_perc += 1

"""
# Retraining
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=param['learning_rate'])

train(model, criterion, optimizer, param, loader_train)


# Check accuracy and nonzeros weights in each layer
print("--- After retraining ---")
test(model, loader_test)
prune_rate(model)


# Save and load the entire model
torch.save(model.state_dict(), 'models/mlp_pruned.pkl')
"""
