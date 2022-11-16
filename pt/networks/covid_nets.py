import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 3)

        self.flat = nn.Flatten()
        
        self.fc1 = nn.Linear(1152, 128)
        self.fc2 = nn.Linear(128, 4)
        
        self.pool = nn.MaxPool2d(2, 2)
        
    def forward(self, x):
        # Ex. Input (b, 3, 50, 50)
        x = self.pool(F.relu(self.conv1(x))) # (b, 32, 23, 23)
        x = self.pool(F.relu(self.conv2(x))) # (b, 64, 9, 9)
        x = self.pool(F.relu(self.conv3(x))) # (b, 128, 3, 3)
        x = self.flat(x) # (b, 1152, )
        x = self.fc1(x) # (b, 128, )
        x = self.fc2(x) # (b, 4, )
        return x