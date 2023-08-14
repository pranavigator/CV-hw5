import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim
import scipy

np.random.seed(0)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size = 5, stride = 1, padding = 2) 
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size = 2)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size = 5, stride = 1, padding = 2)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size = 2)

        self.fc1 = nn.Linear(32*8**2, 36)

        self.softmax = nn.Softmax(1)
        # self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):

        x = x.view(-1, 1, 32, 32)
        # print(x.shape)

        out = self.conv1(x)
        # print("out", out.shape)
        out = self.relu1(out)
        # print("out", out.shape)
        out = self.maxpool1(out)
        # print("out", out.shape)

        out = self.conv2(out)
        # print("out", out.shape)
        out = self.relu2(out)
        # print("out", out.shape)
        out = self.maxpool2(out)
        # print("out", out.shape)

        out = out.view(-1, 32*8**2)
        out = self.fc1(out)
        out = self.softmax(out)

        return out

#Code from nn.py
def compute_loss_torch(y, probs):

    ##########################
    ##### your code here #####
    ##########################
    # loss = 0.0
    loss = - torch.sum(torch.multiply(y, torch.log(probs)))

    return loss

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
    train_data = scipy.io.loadmat('../data/nist36_train.mat')

    train_x, train_y = train_data['train_data'], train_data['train_labels'] 

    num_epochs = 5
    batch_size = 10
    learning_rate = 1e-3

    net = CNN()
    optim = torch.optim.Adam(net.parameters(), lr = learning_rate)
    # criterion = nn.CrossEntropyLoss()

    dataset = TensorDataset(torch.from_numpy(train_x).float(), torch.from_numpy(train_y).float())
    dataloader = DataLoader(dataset, batch_size, shuffle = True)

    acc_list = []
    loss_list = []

    for i in range(num_epochs):
        loss_tot = 0
        avg_acc = 0
        acc_sum = 0
        count = 0

        for _, (x_batch, y_batch) in enumerate(dataloader):

            probs = net.forward(x_batch)

            loss = compute_loss_torch(y_batch, probs)

            y_pred = np.argmax(probs.detach().numpy(), axis=1)
    
            label = np.argmax(y_batch.detach().numpy(), axis = 1)

            match_count = np.sum(y_pred == label)

            acc = match_count/len(label)

            acc_sum += acc
            count += 1
            
            optim.zero_grad()
                        
            loss.backward()

            optim.step()
            
            loss_tot += loss.detach().numpy()
        
        avg_acc = acc_sum/count

        print("\nEpoch:", i)
        print("Average Accuracy for epoch:", avg_acc)

        print("Total Loss:", loss_tot)

        acc_list.append(avg_acc)
        loss_list.append(loss_tot)

    epochs = np.arange(5)

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
        
if __name__==main():
    main()