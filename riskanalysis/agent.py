import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import env 




class Policy(nn.Module):
    def __init__(self, state_size, action_size):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        actions = self.softmax(self.fc3(x))
        return actions
    
class Value(nn.Module):
    def __init__(self, state_size):
        super(Value, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value
    
def trainppo(policy, value, optimizer, trajectories, gamma = 0.99, epsilon = 0.2):

    
