import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv4 = nn.Conv2d(128, 256, 5)
        
        self.fc1 = nn.Linear(256, 4)
        
        self.pool = nn.MaxPool2d(2, 2)
        
    def forward(self, x):
        # Ex. Input (3, 224, 224)
        x = self.pool(F.relu(self.conv1(x))) # (32, 110, 100)
        x = self.pool(F.relu(self.conv2(x))) # (64, 53, 53)
        x = self.pool(F.relu(self.conv3(x))) # (128, 25, 25)
        x = self.pool(F.relu(self.conv4(x))) # (256, 10, 10)
        bs, _, _, _ = x.shape
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1) # (256, )
        x = self.fc1(x) # (4, )
        return x