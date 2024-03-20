import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import board

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__() #input = 8 x 8 x 9, output = 4096
        
        self.conv1 = nn.Conv2d(9, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 2 * 2, 2048)  # Adjust based on image size
        self.fc2 = nn.Linear(2048, 4096)  # 10 classes for CIFAR-10
       
    def forward(self, state):
        batch_size = None
        if(len(state.shape) == 1):
            state = state[:576]
        else:
            state = state[:, :576]
            batch_size = state.shape[0]
        
        x = state.view(-1, 9, 8, 8)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 128 * 2 * 2)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        if batch_size is not None:
            return x.view(batch_size, 4096)
        else:
            return x[0]