from grow.ri_adaption_old import RIAdaption
from utils import find_layers
from torch.optim import Adam
from model.CNN import LeNet5_GROW
from main import generate_data_loader
import torch.nn as nn
from utils import device

model = LeNet5_GROW()
optimizer = Adam(model.parameters(), lr=0.01)
ri = RIAdaption(model, optimizer, 'ours')
g_list = [(0, 1), (1, 3), (2, 3)]

trainloader, testloader = generate_data_loader(batch_size=100, dataset='MNIST', is_main=False)
criterion = nn.CrossEntropyLoss()
for i, (inputs, target) in enumerate(trainloader):
    model.train()
    inputs = inputs.to(device)
    target = target.to(device)
    optimizer.zero_grad()
    _, output = model(inputs)
    loss = criterion(output, target)
    loss.backward()

    model = ri.grow(g_list)

print('a')
