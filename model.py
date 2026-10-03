import torch
import torch.nn as nn
import torch.nn.functional as F

class FC(nn.Module):
    
    def __init__(self, in_dim, out_dim, num_hidden_layers, layer_size):
        super().__init__()

        self.num_layers = num_hidden_layers * 2 + 3 # *2 accounts for ReLU layers, +3 is input layer, input relu layer, output layer

        self.in_dim = in_dim
        self.out_dim = out_dim        

        self.layer_size = layer_size

        self.layer_list = nn.ModuleList()

        self.layer_list.append(nn.Linear(self.in_dim, self.layer_size))
        self.num_hidden_layers = num_hidden_layers

        for i in range(1,self.num_hidden_layers):
            self.layer_list.append(nn.Linear(self.layer_size, self.layer_size))
            

        self.layer_list.append(nn.Linear(self.layer_size, self.out_dim))
        
    def forward(self, x):

        x = x.view(-1, self.in_dim)

        for i in range(self.num_hidden_layers):
            x = F.relu(self.layer_list[i](x))

        return self.layer_list[self.num_hidden_layers](x)

class CNN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv1_1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2)    

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_input_size = 256
        self.dropout = nn.Dropout(0.5)

        self.fc = nn.Linear(self.fc_input_size, self.out_dim)

    def forward(self, x):
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = F.relu(self.bn1_2(self.conv1_2(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = F.relu(self.bn2_2(self.conv2_2(x)))
        x = self.pool2(x)

        x = F.relu(self.bn3_1(self.conv3_1(x)))
        x = F.relu(self.bn3_2(self.conv3_2(x)))
        x = self.pool3(x)

        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class CNN_small(nn.Module):
    
    def __init__(self, in_dim, out_dim):
        super().__init__()
        #image size
        self.in_dim = in_dim
        self.out_dim = out_dim        

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        #assuming input images are 32x32
        self.fc_inputs = 64 * 8 * 8

        self.fc1 = nn.Linear(self.fc_inputs, 512)
        self.fc2 = nn.Linear(512, self.out_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        #layer 1
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        #layer 2
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))

        #flatten
        x = x.view(x.size(0), -1)

        #layer 3
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        #layer 4
        x = self.fc2(x)
        return x