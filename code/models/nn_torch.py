import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class NN(nn.Module):
    def __init__(self, input_size):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 12)
        self.dr1 = nn.Dropout(p=0.3)
        #self.fc2 = nn.Linear(12, 6)
        #self.dr2 = nn.Dropout(p=0.3)
        self.out = nn.Linear(12, 1)

    def forward(self, x):
        x = self.dr1(F.relu(self.fc1(x)))
        #x = self.dr2(F.relu(self.fc2(x)))
        x = F.sigmoid(self.out(x))
        return x