from torch.utils.data import DataLoader, Dataset
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

from sklearn.model_selection import train_test_split


#######################


class Data(Dataset):

    def __init__(self, x, y, num_channels=8, width=20, Mode='2D'):
        self.x = np.load(x)
        self.y = np.load(y)
        # normalize
        self.Mode = Mode
        if Mode == '2D':
            self.x = self.x.transpose((0, 2, 1))
            self.x = self.x.reshape((-1, num_channels, 1, width))
            self.x = self.x.astype(np.float32)
            self.x = (self.x - np.mean(self.x)) / np.std(self.x)
        if Mode == 'S+C':
            self.x = self.x.astype(np.float32)
            self.x = (self.x - np.mean(self.x)) / np.std(self.x)
        if Mode == 'C+S':
            self.x = self.x.transpose((0, 2, 1))
            self.x = self.x.astype(np.float32)
            self.x = (self.x - np.mean(self.x)) / np.std(self.x)
        self.y = self.y
        self.y = self.y.astype(np.long)
        self.y = self.y.reshape((-1))
        self.y = self.y - 1

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        # load image as ndarray type (Height * Width * Channels)
        # be carefull for converting dtype to np.uint8 [Unsigned integer (0 to 255)]
        # in this example, i don't use ToTensor() method of torchvision.transforms
        # so you can convert numpy ndarray shape to tensor in PyTorch (H, W, C) --> (C, H, W)
        if self.Mode == '2D':
            x = self.x[index, :, :, :]
        if self.Mode == 'S+C' or self.Mode == 'C+S':
            x = self.x[index, :, :]
        y = self.y[index]
        return x, y


def save_numpy_to_csv(data, file_path):
    np.savetxt(file_path, data, delimiter=',', fmt='%1.10e')


class DataSet(Dataset):
    def __init__(self, path_to_x, path_to_y, train=True):
        self.x = np.load(path_to_x)
        self.y = np.load(path_to_y)

        # if train:
        #     for i, seed in enumerate([1, 3, 56, 123, 444, 4, 555, 435]):
        #         X_train, X_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.1, shuffle=True, random_state=seed, stratify=self.y)
        #         a = X_test
        #         b = y_test
        #         b = b + 1
        #         d = np.hstack((a.reshape(-1, 900), b.reshape(-1, 1)))
        #         save_numpy_to_csv(d, f"/Users/zber/Desktop/train/train_{i}.csv")

        self.x = self.x.astype(np.float32)
        self.y = self.y.astype(np.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        return x, y


class DataSetNP(Dataset):
    def __init__(self, x_np, y_np):
        self.x = x_np
        self.y = y_np
        self.x = self.x.astype(np.float32)
        self.y = self.y.astype(np.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        return x, y


class Config:

    def __init__(self, path_x=None, path_y=None, input_width=20, input_height=1, channel=8, num_classes=6,
                 batch_size=16, num_epochs=20, learning_rate=0.001, shuffle=True):
        self.input_width = input_width
        self.input_height = input_height
        self.channel = channel
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.path_to_x = path_x
        self.path_to_y = path_y
        self.num_classes = num_classes
        self.dataset = Data(self.path_to_x, self.path_to_y, num_channels=channel, width=input_width)
        self.shuffle = shuffle

    def data_loader(self):
        train_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        return train_loader


def generate_fewshot_npy(is_main=True):
    npy_file = "/Users/zber/ProgramDev/exp_pyTorch/data/MNIST_npy/mnist"

    n_classes = 10
    n_pclass = 5000
    n_channel = 1
    width = 28
    hight = 28

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    trainset = torchvision.datasets.MNIST(root=root_dir, train=True,
                                          download=True, transform=transform)

    testset = torchvision.datasets.MNIST(root=root_dir, train=False,
                                         download=True, transform=transform)
    train_data = trainset.train_data.numpy()
    train_label = trainset.train_labels.numpy()

    test_data = testset.test_data.numpy()
    test_label = testset.test_label.numpy()

    data_npy = np.zeros((n_classes, n_pclass, n_channel, hight, width))

    num_per_class = [0 for _ in range(10)]

    for p_image, label in zip(train_data, train_label):
        index_in_label = num_per_class[label]
        if index_in_label == n_pclass:
            continue
        data_npy[label, index_in_label] = np.expand_dims(p_image, axis=0)
        num_per_class[label] += 1

    # save npy file
    np.save(npy_file, data_npy)


def generate_data_loader(batch_size, dataset='MNIST', is_main=True):
    if dataset == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        if is_main:
            root_dir = '../data'
        else:
            root_dir = '../../data'
        trainset = torchvision.datasets.MNIST(root=root_dir, train=True,
                                              download=True, transform=transform)

        testset = torchvision.datasets.MNIST(root=root_dir, train=False,
                                             download=True, transform=transform)

        # selecting classes
        # train_idx = torch.tensor([True if i in [0, 1, 2, 3, 4] else False for i in trainset.targets])
        # trainset.data = trainset.data[train_idx]
        # trainset.targets = trainset.targets[train_idx]
        #
        # test_idx = torch.tensor([True if i in [0, 1, 2, 3, 4] else False for i in testset.targets])
        # testset.data = testset.data[test_idx]
        # testset.targets = testset.targets[test_idx]

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=0)

        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=0)

    elif dataset == 'CIFAR10':
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        if is_main:
            root_dir = '../data'
        else:
            root_dir = '../../data'

        trainset = torchvision.datasets.CIFAR10(root=root_dir, train=True,
                                                download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=0)
        testset = torchvision.datasets.CIFAR10(root=root_dir, train=False,
                                               download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=0)
    elif dataset == 'EMG':

        # path_to_x_train = "/Users/zber/ProgramDev/data_process_jupyter/EMG/data_gen/npy_data/train_X.npy"
        # path_to_y_train = "/Users/zber/ProgramDev/data_process_jupyter/EMG/data_gen/npy_data/train_y.npy"
        # path_to_x_test = "/Users/zber/ProgramDev/data_process_jupyter/EMG/data_gen/npy_data/test_X.npy"
        # path_to_y_test = "/Users/zber/ProgramDev/data_process_jupyter/EMG/data_gen/npy_data/test_y.npy"

        # path_to_x_train = "/Users/zber/ProgramDev/data_process_jupyter/EMG/data_gen/npy_data_w100/train_X.npy"
        # path_to_y_train = "/Users/zber/ProgramDev/data_process_jupyter/EMG/data_gen/npy_data_w100/train_y.npy"
        # path_to_x_test = "/Users/zber/ProgramDev/data_process_jupyter/EMG/data_gen/npy_data_w100/test_X.npy"
        # path_to_y_test = "/Users/zber/ProgramDev/data_process_jupyter/EMG/data_gen/npy_data_w100/test_y.npy"

        path_to_x_train = "/Users/zber/ProgramDev/data_process_jupyter/EMG/data_gen/npy_data_part/train_X.npy"
        path_to_y_train = "/Users/zber/ProgramDev/data_process_jupyter/EMG/data_gen/npy_data_part/train_y.npy"
        path_to_x_test = "/Users/zber/ProgramDev/data_process_jupyter/EMG/data_gen/npy_data_part/test_X.npy"
        path_to_y_test = "/Users/zber/ProgramDev/data_process_jupyter/EMG/data_gen/npy_data_part/test_y.npy"

        trainset = DataSet(path_to_x_train, path_to_y_train)
        testset = DataSet(path_to_x_test, path_to_y_test)
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)

    elif dataset == 'Har':

        # path_to_x_train = '/Users/zber/ProgramDev/data_process_jupyter/Har/data_gen/data_npy/train_X.npy'
        # path_to_y_train = '/Users/zber/ProgramDev/data_process_jupyter/Har/data_gen/data_npy/train_y.npy'
        # path_to_x_test = '/Users/zber/ProgramDev/data_process_jupyter/Har/data_gen/data_npy/test_X.npy'
        # path_to_y_test = '/Users/zber/ProgramDev/data_process_jupyter/Har/data_gen/data_npy/test_y.npy'

        path_to_x_train = '/Users/zber/ProgramDev/data_process_jupyter/Har/data_gen/data_npy_shuffle/train_X.npy'
        path_to_y_train = '/Users/zber/ProgramDev/data_process_jupyter/Har/data_gen/data_npy_shuffle/train_y.npy'
        path_to_x_test = '/Users/zber/ProgramDev/data_process_jupyter/Har/data_gen/data_npy_shuffle/test_X.npy'
        path_to_y_test = '/Users/zber/ProgramDev/data_process_jupyter/Har/data_gen/data_npy_shuffle/test_y.npy'

        # path_to_x_train = '/Users/zber/ProgramDev/data_process_jupyter/Har/data_gen/data_npy_part/train_X.npy'
        # path_to_y_train = '/Users/zber/ProgramDev/data_process_jupyter/Har/data_gen/data_npy_part/train_y.npy'
        # path_to_x_test = '/Users/zber/ProgramDev/data_process_jupyter/Har/data_gen/data_npy_part/test_X.npy'
        # path_to_y_test = '/Users/zber/ProgramDev/data_process_jupyter/Har/data_gen/data_npy_part/test_y.npy'

        trainset = DataSet(path_to_x_train, path_to_y_train)
        testset = DataSet(path_to_x_test, path_to_y_test)
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)

    elif dataset == 'myHealth':
        path_to_x_test = "/Users/zber/ProgramDev/data_process_jupyter/MyHealth/gen_data/data_npy/test_X.npy"
        path_to_y_test = "/Users/zber/ProgramDev/data_process_jupyter/MyHealth/gen_data/data_npy/test_y.npy"
        path_to_x_train = "/Users/zber/ProgramDev/data_process_jupyter/MyHealth/gen_data/data_npy/train_X.npy"
        path_to_y_train = "/Users/zber/ProgramDev/data_process_jupyter/MyHealth/gen_data/data_npy/train_y.npy"

        trainset = DataSet(path_to_x_train, path_to_y_train)
        testset = DataSet(path_to_x_test, path_to_y_test)
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)

    elif dataset == 'FinDroid':
        path_to_x_test = "/Users/zber/ProgramDev/data_process_jupyter/Finger/data_gen/test_X.npy"
        path_to_y_test = "/Users/zber/ProgramDev/data_process_jupyter/Finger/data_gen/test_y.npy"

        # path_to_x_test = "/Users/zber/ProgramDev/data_process_jupyter/Finger/data_8test/x8.npy"
        # path_to_y_test = "/Users/zber/ProgramDev/data_process_jupyter/Finger/data_8test/y8.npy"

        path_to_x_train = "/Users/zber/ProgramDev/data_process_jupyter/Finger/data_gen/train_X.npy"
        path_to_y_train = "/Users/zber/ProgramDev/data_process_jupyter/Finger/data_gen/train_y.npy"

        trainset = DataSet(path_to_x_train, path_to_y_train)
        testset = DataSet(path_to_x_test, path_to_y_test, train=False)
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)

    elif dataset == 'Fin_Sitting':
        path_to_x_train = "/Users/zber/ProgramDev/data_process_jupyter/Finger/sitting/x_train.npy"
        path_to_y_train = "/Users/zber/ProgramDev/data_process_jupyter/Finger/sitting/y_train.npy"

        path_to_x_test = "/Users/zber/ProgramDev/data_process_jupyter/Finger/sitting/x_test.npy"
        path_to_y_test = "/Users/zber/ProgramDev/data_process_jupyter/Finger/sitting/y_test.npy"

        trainset = DataSet(path_to_x_train, path_to_y_train)
        testset = DataSet(path_to_x_test, path_to_y_test, train=False)
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)

    elif dataset == 'Fin_SitRun':
        path_to_x_train = "/Users/zber/ProgramDev/data_process_jupyter/Finger/running/x_train.npy"
        path_to_y_train = "/Users/zber/ProgramDev/data_process_jupyter/Finger/running/y_train.npy"

        path_to_x_test = "/Users/zber/ProgramDev/data_process_jupyter/Finger/running/x_test.npy"
        path_to_y_test = "/Users/zber/ProgramDev/data_process_jupyter/Finger/running/y_test.npy"

        trainset = DataSet(path_to_x_train, path_to_y_train)
        testset = DataSet(path_to_x_test, path_to_y_test, train=False)
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)

    return trainloader, testloader


if __name__ == "__main__":
    generate_fewshot_npy()
