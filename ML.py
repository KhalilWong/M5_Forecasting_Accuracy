import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import Init

################################################################################
class Net(nn.Module):
    #
    def __init__(self, Type = 1941):
        #
        super().__init__()
        self.conv1 = nn.Conv1d(3, 6, 1000)
        self.conv2 = nn.Conv1d(6, 9, 1000)
        self.conv3 = nn.Conv1d(9, 12, 1000)
        self.conv4 = nn.Conv1d(12, 15, 1000)
        #
        self.fc1 = nn.Linear(79695, 4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, Type)
        #
        #self.conv1 = nn.Conv1d(1, 3, 5)

    def forward(self, x):
        # 2x2 Max pooling
        x = F.max_pool1d(F.relu(self.conv1(x)), 2)
        x = F.max_pool1d(F.relu(self.conv2(x)), 2)
        x = F.max_pool1d(F.relu(self.conv3(x)), 2)
        x = F.max_pool1d(F.relu(self.conv4(x)), 2)
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

    def forward(self, y_a, y_f):
        x = F.max_pool1d(F.relu(self.conv1(x)), 2)
        x = F.max_pool1d(F.relu(self.conv2(x)), 2)
        x = F.max_pool1d(F.relu(self.conv3(x)), 2)
        x = F.max_pool1d(F.relu(self.conv4(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

################################################################################
def main():
    #
    Names = ['Evaluation', 'Validation']
    N_day = [1941, 1969]
    N_Train_day = [1913, 1941]
    #
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum = 0.9)
    running_loss = 0.0
    for i in range(77):
        for j in range(2):
            data_list = Init.Read_CSV(Names[j] + '_Part'+ str(i) +'.csv')
            data = np.array(data_list)
            N, d = data.shape
            #
            inputs = data[:, :d - 1]
            labels = data[:, -1]
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            #
            running_loss += loss.item()
