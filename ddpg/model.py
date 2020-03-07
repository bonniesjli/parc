import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class Actor(nn.Module):
    '''Simple Policy Network'''
    def __init__(self, state_dim, action_dim, hidden_dim, action_space):
    	super(Actor, self).__init__()

    	self.l1 = nn.Linear(state_dim, hidden_dim)
    	self.l2 = nn.Linear(hidden_dim, hidden_dim)
    	self.l3 = nn.Linear(hidden_dim, action_dim)

    	self.max_action = float(action_space.high[0])


    def forward(self, state):
    	a = F.relu(self.l1(state))
    	a = F.relu(self.l2(a))
    	return self.max_action * torch.tanh(self.l3(a))

class Critic(nn.Module):
    '''Critic Network'''
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim + action_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        q = F.relu(self.l1(state))
        q = F.relu(self.l2(torch.cat([q, action], 1)))
        return self.l3(q)

    def forward_one(self, state, action):
        q = F.relu(self.l1(state))
        q = F.relu(self.l2(torch.cat([q, action], 0)))
        return self.l3(q)
