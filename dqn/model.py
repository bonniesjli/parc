import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Q Network"""

    def __init__(self, state_size, action_size, hidden_size):
        '''Initialize parameters and build model.
        :param state_size (int): Dimension of each state
        :param action_size (int): Dimension of each action
        :param hidden_size (int): size of hidden layers
        '''
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
