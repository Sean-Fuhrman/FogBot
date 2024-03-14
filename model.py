import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import board

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__() #input = 578, output = 4,672
        self.flatten = nn.Flatten()
        self.activation = nn.ReLU()
        self.dense1 = nn.Linear(8*8*9 + 2, 1024)
        self.dense2 = nn.Linear(1024, 2048)
        self.dense3 = nn.Linear(2048, 2048)
        self.output = nn.Linear(2048, 8 * 8 * 73)
       
    def forward(self, state):
        state = self.activation(self.dense1(state))
        state = self.activation(self.dense2(state))
        state = self.activation(self.dense3(state))
        return self.output(state)