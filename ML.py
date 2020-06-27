import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import Init
import numpy as np

################################################################################
class Net(nn.Module):
    #
    def __init__(self, N_Item = 1, Type = 28):
        #
        super().__init__()
        self.conv1 = nn.Conv2d(1, 3, 3, padding = 1)
        self.conv2 = nn.Conv2d(3, 8, 3, padding = 1)
        #
        self.fc1 = nn.Linear(15520 * N_Item, 2048 * N_Item)                     #248,448; 16,384
        self.fc2 = nn.Linear(2048 * N_Item, 1024 * N_Item)                      #16,384; 1,024
        self.fc3 = nn.Linear(1024 * N_Item, Type)

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

    def forward(self, y_a, y_f, n, h = 28):
        loss = torch.sum(torch.pow((y_a[0, -h:] - y_f), 2)) / h
        loss /= torch.sum(torch.pow((y_a[0, 1:-h] - y_a[0, :-h-1]), 2)) / (n - 1)
        loss = torch.sqrt(loss)
        return(loss)

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
    #
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = Net()
    net.to(device)
    criterion = RMSSE()
    optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum = 0.9)
    running_loss = 0.0
    for i in range(77):
        data_list = Init.Read_CSV('./Train_Evaluation/'+ Names[0] + '_Part'+ str(i) +'.csv')
        data = StrListToTensor(data_list)
        N, d = data.shape
        #
        for j in range(int(N / N_day[0])):
            inputs = data[j * N_day[0]: (j + 1) * N_day[0], :d - 1]
            inputs = inputs.view(1, 1, -1, 16)
            labels = data[j * N_day[0]: (j + 1) * N_day[0], -1]
            labels = labels.view(1, -1)
            inputs_d = inputs.to(device)
            labels_d = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs_d)
            loss = criterion(labels_d, outputs, N_Train_day[0])
            loss.backward()
            optimizer.step()
            #
            running_loss += loss.item()
            if j % 100 == 99:
                print('[%d, %d] loss: %f' % (i, j + 1, running_loss / 100))
                running_loss = 0.0
    print('Finished Training')

################################################################################
if __name__ == '__main__':
    main()
