from prune_filter.finetune import PrunningFineTuner_VGG16
from model.CNN import LeNet5
from main import train_test
import torch
import torch.nn as nn
from utils import write_log, dir_path
from MNIST_train_valide_splite import get_train_valid_loader
import time

if __name__ == "__main__":
    param = {
        'batch_size': 100,
        'learning_rate': 0.0005,
        'epoch': 20,
        'split_percentage': 0.66
    }

    # dataloader added
    #_, test_loader = generate_data_loader(batch_size=param['batch_size'], dataset='MNIST')

    (train_loader, valid_loader) = get_train_valid_loader(data_dir='../../data', batch_size=param['batch_size'],
                                                          valid_size=param['split_percentage'], random_seed=45)
    (test_loader, test2_loader) = get_train_valid_loader(data_dir='../../data', batch_size=param['batch_size'],
                                                         valid_size=param['split_percentage'], random_seed=45, is_train=False)

    # path created
    path = dir_path('Prune_LeNet5')

    # pretrained model loaded
    model = LeNet5()
    train_test(model, train_loader, test_loader, param['learning_rate'], param['epoch'], dic_path=path,
               str_channel='LeNet5')

    # model.load_state_dict(
    #     torch.load('/Users/zber/ProgramDev/exp_pyTorch/results/Prune_LeNet520200207-195559/model.ckpt'))

    # criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=param['learning_rate'])

    print("--- Pretrained network loaded ---")
    write_log("--- Pretrained network test ---\n", path['path_to_log'])

    # test accuracy after loading
    # test(model, test_loader, criterion, path['path_to_log'])

    # pruning
    start = time.time()
    fine_tuner = PrunningFineTuner_VGG16(train_loader, valid_loader, test_loader, test2_loader, model, optimizer, path)

    fine_tuner.prune()
    total_time = time.time() - start
    write_log('The total time is : {}\n'.format(total_time), path['path_to_log'])
