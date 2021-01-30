from prune_filter.finetune import PrunningFineTuner_VGG16
from main import train_test
import torch
import torch.nn as nn
from utils import write_log, dir_path
from MNIST_train_valide_splite import get_train_valid_CIFAR_loader
from model.VGG import VGG
import time

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    param = {
        'batch_size': 100,
        'learning_rate': 0.01,
        'epoch': 20,
        'split_percentage': 0.66
    }

    # dataloader added
    # _, test_loader = generate_data_loader(batch_size=param['batch_size'], dataset='MNIST')

    (train_loader, valid_loader) = get_train_valid_CIFAR_loader(data_dir='../../data', batch_size=param['batch_size'],
                                                                valid_size=param['split_percentage'], random_seed=45)

    (test_loader, test2_loader) = get_train_valid_CIFAR_loader(data_dir='../../data', batch_size=param['batch_size'],
                                                               valid_size=param['split_percentage'], random_seed=45,
                                                               is_train=False)
    # path created
    path = dir_path('Prune_Vgg')

    # load model
    model = VGG('VGG16')
    model = model.to(device)
    # train model
    train_test(model, train_loader, test_loader, param['learning_rate'], param['epoch'], dic_path=path,
               str_channel='VGG16')

    # load pretrained model
    # model.load_state_dict(
    #     torch.load(
    #         'C:\\Users\\Zber\\Documents\\Dev_program\\FastGrownTest\\results\\CIFAR10_AlexNet_20200209-022945\\model_AlexNet.ckpt'))

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
