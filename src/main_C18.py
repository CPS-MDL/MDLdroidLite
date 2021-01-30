import torch
import torch.nn as nn
from model import mobileNetv1, CNN, TCN, LSTM
import time
import data_loader
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import datetime

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(train_loader, model, criterion, optimizer, to_log, print_freq=100, data_type='image', epoch=0):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    train_loss = []

    # switch to train mode
    model.train()

    # prepare a txt file for logging
    log = open(to_log, 'a')

    start = time.time()

    for i, (inputs, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - start)

        # prepare input and target
        if data_type == 'image':
            inputs = inputs.to(device)
            target = target.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # compute output
        output = model(inputs)
        loss = criterion(output, target)
        train_loss.append(loss.item())
        loss.backward()
        optimizer.step()
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # compute gradient and do SGD step

        # measure elapsed time
        batch_time.update(time.time() - start)
        start = time.time()

        if i % print_freq == 0:
            str = ('Epoch: [{0}][{1}/{2}]\t'
                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                   'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                   'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                   'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
            print(str)
            log.write(str + '\n')
    return train_loss

    # print('The total training time is {}'.format(time.time() - start))
    # log.write('The total training time is {}\n'.format(time.time() - start))
    # log.close()


# custom weights initialization called on nets
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.05)
        m.bias.data.normal_(0.0, 0.05)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def test(model, test_loader, criterion, to_log):
    log = open(to_log, 'a')
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target)  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)
    format_str = '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), acc
    )
    print(format_str)
    log.write(format_str)
    log.close()
    return test_loss.item(), acc


def validate(val_loader, model, criterion, to_log, data_type='image', print_freq=100):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # prepare a txt file for logging
    log = open(to_log, 'a')

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (inputs, target) in enumerate(val_loader):

        if data_type != 'image':
            inputs = inputs.to(device)
            target = target.to(device)

        # compute output
        output = model(inputs)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            str = ('Test: [{0}/{1}]\t'
                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                   'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                   'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1, top5=top5))
            print(str)
            log.write(str + '\n')

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
    log.write(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
    log.close()
    return top1.avg


"""
# Train the model
def train(data_loader=None, model=None, criterion=None, optimizer=None, num_epochs=20):
    total_step = len(data_loader)
    for epoch in range(num_epochs):
        for i, (x, y) in enumerate(data_loader):
            x = x.to(device)
            y = y.to(device)

            # Forward pass
            outputs = model(x)
            loss = criterion(outputs, y)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 20 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
    return model
"""
"""
# Test the model
def test(model=None, test_loader=None):
    with torch.no_grad():
        correct = 0
        total = 0
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))
        



# def EMG(model_name='lstm', path_to_x=None, path_to_y=None):
#     kwargs = {'num_classes': 6, 'num_channel': 8,
#               'out1': 8, 'out2': 8, 'out3': 16, 'out4': 16, 'out5': 16, 'f1': 300, 'f2': 300}
#     EMG = data_loader.Config(path_x=path_to_x, path_y=path_to_y, input_width=150, num_classes=6)
#     data_loader = EMG.data_loader()
#     if model_name == 'lstm':
#         model = LSTM.RNN(input_size=EMG.channel, hidden_size=128, num_layers=2, num_classes=EMG.num_classes).to(device)
#     if model_name == 'mobileNet':
#         model = mobileNet.MobileNetV2(n_class=EMG.num_classes, input_size=EMG.input_width, width_mult=1.,
#                                       channel=EMG.channel)
#     if model_name == 'AlexNet':
#         model = AlexNet.AlexNet(**kwargs)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=EMG.learning_rate)
#     model = train(model=model, data_loader=data_loader, criterion=criterion, optimizer=optimizer)
#     test(model=model, test_loader=data_loader)
#     return model
"""


# Save the model checkpoint
def saveModel(model=None, path_to_model=None, mode='part'):
    if mode == 'part':
        torch.save(model.state_dict(), path_to_model)
    if mode == 'entire':
        torch.save(model, path_to_model)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


# Calculate the size of parameters.
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # dataset and model setting
    dataset = 'MNIST'  # CIFAR10,MNIST,EMG
    model_name = 'CNN'  # mobileNet , LSTM, CNN
    model = None
    trainloader = None
    testloader = None

    # global training settins
    learning_rate = 0.0005
    epoch = 20
    batch_size = 100
    shuffle = True
    t_channel = 24 # [8,12,16,24] for experiment，or standard
    CONS_FLATTEN_MULTIPLIER = 144
    FULLY_LAYER_MULTIPLIER = 16
    # t_fc = {8: 1152, 12: 1728, 16: 2304, 24: 3456}
    t_kernel = 3  # [3,7,9] for experiment

    # dataset setting
    num_classes = None
    num_channels = None
    width = None
    kwargs = None

    # 2 con, 2 pooling, 1 fully

    # log folder setting
    parent_dir = os.path.dirname(os.getcwd())
    res_dir = os.path.join(parent_dir, 'results/')
    res_dir = res_dir + '{}_{}_C{}_K{}_B{}_'.format(dataset, model_name, t_channel, t_kernel, batch_size)
    res_dir = res_dir + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    path_to_model = os.path.join(res_dir, 'model.ckpt')
    path_to_log = os.path.join(res_dir, 'log.txt')
    path_to_parameter = os.path.join(res_dir, 'parameter.txt')

    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    # dataset
    if dataset == 'CIFAR10':
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        kwargs = {'num_classes': 6, 'num_channel': 8,
                  'out1': 8, 'out2': 8, 'out3': 16, 'out4': 16, 'out5': 16, 'f1': 2304, 'f2': 300}

        trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                                download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=0)
        testset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                               download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=0)

    if dataset == 'MNIST':
        num_classes = 10
        num_channels = 1
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        if model_name == 'CNN':
            # standard
            # kwargs = {'num_classes': 10, 'num_channel': 1,
            #           'out1': 20, 'out2': 50, 'fc1': 8450, 'fc2': 500,
            #           'k_size': 2,
            #           'c_stride': 1,
            #           'p_stride': 2,
            #           }

            kwargs = {'num_classes': 10, 'num_channel': 1,
                      'out1': t_channel, 'out2': t_channel, 'fc1': t_channel * CONS_FLATTEN_MULTIPLIER,
                      'fc2': t_channel * FULLY_LAYER_MULTIPLIER,
                      'k_size': 3,
                      'c_stride': 1,
                      'p_stride': 2,
                      }

        trainset = torchvision.datasets.MNIST(root='../data', train=True,
                                              download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=0)
        testset = torchvision.datasets.MNIST(root='../data', train=False,
                                             download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=0)

    if dataset == 'EMG':
        num_classes = 6
        num_channels = 8
        width = 20
        Mode = '2D'
        if model_name == 'LSTM':
            kwargs = {'num_layers': 2, 'hidden_size': 128, }
            Mode = 'S+C'

        if model_name == 'CNN':
            kwargs = {'num_classes': 6,
                      'num_channel': 8,
                      'out1': t_channel,
                      'out2': t_channel,
                      'k_size': (1, 3),  # 3->576, 7->288
                      'c_stride': (1, 1),
                      'p_stride': (1, 2),
                      'fc1': 192,
                      'fc2': 300}

        if model_name == 'TCN':
            Mode = 'C+S'
            kwargs = {
                'input_size': num_channels, 'output_size': num_classes, 'num_channels': [8, 36, 72, ], 'dropout': 0.05,
                'fc2': 300, 'kernel_size': 3,
            }

        path_to_x_train = '/Users/zber/Program_dev/Finger_demo/Data/W20_overlap_X_train.npy'
        path_to_y_train = '/Users/zber/Program_dev/Finger_demo/Data/W20_overlap_y_train.npy'
        path_to_x_test = '/Users/zber/Program_dev/Finger_demo/Data/W20_overlap_X_test.npy'
        path_to_y_test = '/Users/zber/Program_dev/Finger_demo/Data/W20_overlap_y_test.npy'

        trainset = data_loader.Data(x=path_to_x_train, y=path_to_y_train, num_channels=num_channels, width=width,
                                    Mode=Mode)
        testset = data_loader.Data(x=path_to_x_test, y=path_to_y_test, num_channels=num_channels, width=width,
                                   Mode=Mode)
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=shuffle)

    ###  load model and training
    # model
    if model_name == 'mobileNet':
        model = mobileNetv1.Net()
    if model_name == 'LSTM':
        model = LSTM.RNN(input_size=num_channels, num_classes=num_classes, **kwargs)
    if model_name == 'CNN':
        # model = test2.Net()
        model = CNN.Net(**kwargs)
        model.apply(weights_init)  # Customer the model weights
    if model_name == 'TCN':
        model = TCN.TCN(**kwargs)

    # initialize critierion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = open(path_to_log, 'a')

    # start training
    start = time.time()

    test_loss = []
    test_acc = []
    batch_loss = []
    for e in range(epoch):
        train_loss = train(trainloader, model, criterion, optimizer, epoch=(e + 1), to_log=path_to_log)
        batch_loss.append(train_loss)
        # validate(testloader, model, criterion)
        loss, acc = test(model, test_loader=testloader, criterion=criterion, to_log=path_to_log)
        path_to_model_epoch = os.path.join(res_dir, ('model_epoch_{}.ckpt'.format(e)))
        saveModel(model=model, path_to_model=path_to_model_epoch, mode='entire')
        test_loss.append(loss)
        test_acc.append(acc)

    # log.write('====================Validation====================\n')
    # validate(testloader, model, criterion)
    total_time = time.time() - start
    # save model
    saveModel(model=model, path_to_model=path_to_model)
    log = open(path_to_log, 'a')
    # the size of parameters
    log.write('The number of parameters in the model is : {}\n'.format(count_parameters(model)))
    print('The total training time is {}\n'.format(total_time))
    log.write('Loss:{}\n'.format(test_loss))
    log.write('Accuracy:{}\n'.format(test_acc))
    log.write('The total training time is {}\n'.format(total_time))
    for index, item in enumerate(batch_loss):
        log.write('\nC_{}_Epoch_{}_Batch_{} = {};\n'.format(t_channel, index + 1, batch_size, item))
    log.close()
