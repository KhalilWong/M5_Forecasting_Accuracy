import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import Init
import numpy as np
import matplotlib.pyplot as mpl
import time

################################################################################
class Net(nn.Module):
    #
    def __init__(self, N_Item = 1, Type = 28):
        #
        super().__init__()
        self.conv1 = nn.Conv2d(1, 3, 3, padding = 1)
        self.conv2 = nn.Conv2d(3, 8, 3, padding = 1)
        #
        self.fc1 = nn.Linear(15520 * N_Item, 4096 * N_Item)           #248,448; 16,384##15520
        self.fc2 = nn.Linear(4096 * N_Item, 1024 * N_Item)                  #16,384; 1,024
        self.fc3 = nn.Linear(1024 * N_Item, Type * N_Item)

    def forward(self, x):
        # 2x2 Max pooling
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # 除去批处理维度的其他所有维度
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

################################################################################
class RMSSE(nn.Module):
    #
    def __init__(self):
        super().__init__()

    def forward(self, device, y_a, y_f, n, h = 28, N_Batch = 1):
        Sum_loss = torch.tensor([0.0], dtype = torch.float32, device = device)
        for i in range(N_Batch):
            loss = torch.sum(torch.pow((y_a[0, (i + 1) * n + i * h:(i + 1) * (n + h)] - y_f[0, i * h:(i + 1) * h]), 2)) / h
            loss /= torch.sum(torch.pow((y_a[0, i * (n + h) + 1:(i + 1) * n + i * h] - y_a[0, i * (n + h):(i + 1) * n + i * h - 1]), 2)) / (n - 1)
            loss = torch.sqrt(loss)
            Sum_loss += loss
        Sum_loss /= N_Batch
        return(Sum_loss)

################################################################################
def StrListToTensor(List):
    N = len(List)
    d = len(List[0])
    array0 = np.zeros((N, d))
    for i in range(N):
        for j in range(d):
            array0[i, j] = float(List[i][j])
    tensor0 = torch.from_numpy(array0).float()
    #tensor0 = torch.Tensor(tensor0, dtype = torch.double)
    return(tensor0)
################################################################################
def main():
    #
    Names = ['Evaluation', 'Validation']
    N_day = [1941, 1913]
    N_Train_day = [1913, 1885]
    N_Batch = 2
    #
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = Net(N_Item = N_Batch)
    net.to(device)
    criterion = RMSSE()
    optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum = 0.9)
    running_loss = 0.0
    Error = []
    start_time = time.time()
    for epoch in range(20):
        for i in range(77):
            data_list = Init.Read_CSV('./Train_Evaluation/'+ Names[0] + '_Part'+ str(i) +'.csv')
            data = StrListToTensor(data_list)
            N, d = data.shape
            #
            for j in range(int(N / N_day[0] / N_Batch)):
                inputs = data[j * N_day[0] * N_Batch: (j + 1) * N_day[0] * N_Batch, :d - 1]
                inputs = inputs.view(1, 1, -1, 16)
                #labels_Batch = []
                #for k in range(N_Batch):
                #    labels_Batch.append(data[j * N_day[0] * N_Batch + k * N_day[0] + N_Train_day[0]: (j + 1) * N_day[0] * N_Batch - (N_Batch - 1 - k) * N_day[0], -1])
                #labels = torch.stack(labels_Batch, dim = 0)
                #print(labels.shape)
                labels = data[j * N_day[0] * N_Batch: (j + 1) * N_day[0] * N_Batch, -1]
                labels = labels.view(1, -1)
                inputs_d = inputs.to(device)
                labels_d = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = net(inputs_d)
                loss = criterion(device, labels_d, outputs, N_Train_day[0], 28, N_Batch)
                loss.backward()
                optimizer.step()
                #
                running_loss += loss.item()
                if j % int(100 / N_Batch) == int(100 / N_Batch) - 1:
                    print('[Epoch:%d, File:%d, Batch:%d] loss: %f' % (epoch, i, j + 1, running_loss / int(100 / N_Batch)))
                    Error.append(running_loss / int(100 / N_Batch))
                    running_loss = 0.0
    print('Finished Training')
    end_time = time.time()
    print('Total Time: %dd %dh %dm %ds.' % (int(int(end_time - start_time) / 86400), int(int(end_time - start_time) % 86400 / 3600), int(int(end_time - start_time) % 3600 / 60), int(end_time - start_time) % 60))
    PATH = './final_net.pth'
    torch.save(net.state_dict(), PATH)
    mpl.plot(Error)
    mpl.savefig('Error.png', dpi = 600)
    mpl.close()

################################################################################
if __name__ == '__main__':
    main()
