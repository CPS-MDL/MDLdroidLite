import numpy as np  # linear algebra
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
sequence_length = 20
input_size = 8
hidden_size = 128
num_layers = 2
num_classes = 8
batch_size = 100
num_epochs = 20
learning_rate = 0.001


# Recurrent neural network (many-to-one)
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out



if __name__ == '__main__':
    # Dataset settings
    path_to_x = '/Users/zber/ProgramDev/Finger_demo/Data/W20_overlap_x.npy'
    path_to_y = '/Users/zber/ProgramDev/Finger_demo/Data/W20_overlap_y.npy'


    class DatasetEMG(Dataset):

        def __init__(self, x, y, transform=None):
            self.x = np.load(x)
            self.y = np.load(y)
            self.x = self.x.transpose((0, 2, 1))
            self.x = self.x.reshape((-1, 8, 20))
            self.x = self.x.astype(np.float32)
            self.y = self.y.astype(np.long)
            self.transform = transform

        def __len__(self):
            return len(self.x)

        def __getitem__(self, index):
            # load image as ndarray type (Height * Width * Channels)
            # be carefull for converting dtype to np.uint8 [Unsigned integer (0 to 255)]
            # in this example, i don't use ToTensor() method of torchvision.transforms
            # so you can convert numpy ndarray shape to tensor in PyTorch (H, W, C) --> (C, H, W)
            x = self.x[index, :, :]
            y = self.y[index]

            return x, y


    # EMG dataset
    train_dataset = DatasetEMG(path_to_x, path_to_y, transform=None)

    # Data loader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


    # Recurrent neural network (many-to-one)
    class RNN(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, num_classes):
            super(RNN, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, num_classes)

        def forward(self, x):
            # Set initial hidden and cell states
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

            # Forward propagate LSTM
            out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

            # Decode the hidden state of the last time step
            out = self.fc(out[:, -1, :])
            return out


    model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)


    def get_n_params(model):
        pp=0
        for p in list(model.parameters()):
            nn=1
            for s in list(p.size()):
                nn = nn*s
            pp += nn
        return pp

    num = get_n_params(model)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.reshape(-1, sequence_length, input_size).to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

    # Test the model
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in train_loader:
            images = images.reshape(-1, sequence_length, input_size).to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

    # Save the model checkpoint
    torch.save(model.state_dict(), 'model.ckpt')