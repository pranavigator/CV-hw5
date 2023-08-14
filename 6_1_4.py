import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim
import scipy
import torchvision
import torchvision.transforms as transforms
import cv2

from os.path import join
from copy import copy

np.random.seed(0)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size = 5, stride = 1, padding = 2) 
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size = 2)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=16, kernel_size = 5, stride = 1, padding = 2)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size = 2)

        self.fc1 = nn.Linear(16*16**2, 128)
        self.fc2 = nn.Linear(128, 36)

        self.softmax = nn.Softmax(1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):

        x = x.view(-1, 3, 32, 32)
        # print(x.shape)

        out = self.conv1(x)
        out = self.relu1(out)
        out = self.maxpool1(out)
        # print("out", out.shape)

        out = self.conv2(out)
        out = self.relu2(out)

        out = out.view(-1, 16*16**2)
        # print(out.shape)
        out = self.fc1(out)
        out = self.sigmoid(out)
        out = self.fc2(out)
        out = self.softmax(out)

        return out

def get_random_batches(x,y,batch_size):
    batches = []

    num_batches = x.shape[0]//batch_size

    batch_indices = np.random.choice(x.shape[0], size = (num_batches, batch_size), replace = False)

    for i in range(num_batches):
        row = batch_indices[i,:]
        batch_x = x[row]
        batch_y = y[row]
        batches.append((batch_x, batch_y))

    return batches

def main():

    #Followed tutorial from Pytorch website: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

    transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 4
    data_dir = "C:/Users/prana/Desktop/Carnegie Mellon/Computer Vision/16720_S23_hw5/sun_data/"

    train_files = open(join(data_dir, "train_files.txt")).read().splitlines()
    train_labels = np.loadtxt(join(data_dir, "train_labels.txt"), np.int32)

    test_files = open(join(data_dir, "test_files.txt")).read().splitlines()
    test_labels = np.loadtxt(join(data_dir, "test_labels.txt"), np.int32)

    train_data = np.zeros((len(train_files), 32, 32, 3))
    test_data = np.zeros((len(test_files), 32, 32, 3))

    for i, file_name in enumerate(train_files):
        img = cv2.imread(data_dir + file_name)
        img = cv2.resize(img, (32, 32))
        train_data[i, :, :, :] = img

    for i, file_name in enumerate(test_files):
        img = cv2.imread(data_dir + file_name)
        try:
            img = cv2.resize(img, (32, 32))
        except:
            print(file_name)
        test_data[i, :, :, :] = img

    train_data = np.swapaxes(train_data, 1, 3)
    test_data = np.swapaxes(test_data, 1, 3)

    train_dataset = TensorDataset(torch.from_numpy(train_data).float(), torch.from_numpy(train_labels).long())
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle = True)

    test_dataset = TensorDataset(torch.from_numpy(test_data).float(), torch.from_numpy(test_labels).long())
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle = True)

    num_epochs = 15
    # batch_size = 10
    learning_rate = 1e-4

    net = CNN()
    optim = torch.optim.Adam(net.parameters(), lr = learning_rate)
    criterion = nn.CrossEntropyLoss()

    acc_list = []
    loss_list = []

    for i in range(num_epochs):
        loss_tot = 0
        avg_acc = 0
        total_batch_len = 0
        match_count = 0

        for j, data in enumerate(train_dataloader):

            # if (j > 1000):
            #     break
            
            x_batch, y_batch = data

            # zero the parameter gradients
            optim.zero_grad()

            # forward + backward + optimize
            outputs = net(x_batch)
            loss = criterion(outputs, y_batch)
            loss_tot += loss.sum()
            _, predicted = torch.max(outputs.data, 1)
            total_batch_len += y_batch.size(0)
            match_count += (predicted == y_batch).sum().item()
            loss.backward()
            optim.step()

            # print statistics
            loss_tot += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{i + 1}, {j + 1:5d}] loss: {loss_tot / 2000:.3f}')
                loss_tot = 0.0

            # if (j % 100 == 0):
            #     print("Image run:", j)
        
        avg_acc = match_count/total_batch_len

        print("\nEpoch:", i)
        print("Average Accuracy for epoch:", avg_acc)

        print("Total Loss:", loss_tot.detach().numpy())

        acc_list.append(avg_acc)
        loss_list.append(loss_tot.detach().numpy())

    epochs = np.arange(num_epochs)

    plt.plot(epochs, loss_list, label="training")
    plt.xlabel("Epoch")
    plt.ylabel("Average loss")
    plt.xlim(0, len(loss_list)-1)
    plt.legend()
    plt.grid()
    plt.show()

    plt.plot(epochs, acc_list, label="training")
    plt.xlabel("Epoch")
    plt.ylabel("Average Accuracy")
    plt.xlim(0, len(acc_list)-1)
    plt.ylim(0, None)
    plt.legend()
    plt.grid()
    plt.show()
    
    #Testing model on test dataset
    acc_sum = 0
    count = 0
    with torch.no_grad():
        for test in test_dataloader:
            x_test, y_test = test
            # print(y_test)
            outputs = net(x_test)
            y_pred = np.argmax(outputs.detach().numpy(), axis=1)
    
            y_test = y_test.detach().numpy()
            match_count = np.sum(y_pred == y_test)

            acc = match_count/len(y_test)

            acc_sum += acc
            count += 1
    
    avg_acc = acc_sum/count

    print("Accuracy on test dataset:", avg_acc)
if __name__==main():
    main()