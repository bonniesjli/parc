import numpy as np
import random
import copy
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

from model import *
from utils.buffer import ReplayBuffer
from utils.controller import EpsilonController
from utils.update import soft_update, hard_update

class DQN():
    def __init__(self, state_size, action_size, action_space, args):
        self.device = torch.device("cuda" if args.cuda else "cpu")
        self.buffer = ReplayBuffer(args.buffer_size, args.batch_size, self.device)

        self.action_size = action_size
        self.gamma = args.gamma
        self.tau = args.tau

        self.eps = EpsilonController(e_decays = args.eps_decays, e_min = args.eps_min)

        self.q_local = QNetwork(state_size, action_size, args.hidden_size).to(self.device)
        self.q_optimizer = optim.Adam(self.q_local.parameters(), lr=args.lr)
        self.q_target = copy.deepcopy(self.q_local)


    def act(self, state, eval = False):
        '''return action given state
        :param state (np.ndarray): state
        :param eval (bool): whether if we are evaluating policy. set to True in utils.traj.py
        :return action (np.ndarray): action with episilon noise if not eval
        '''
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            action_values = self.q_local(state)

        if not eval:
            # Epsilon-greedy action selection
            if random.random() > self.eps.val():
                action = np.argmax(action_values.cpu().data.numpy())
            else:
                action = random.choice(np.arange(self.action_size))
        else:
            action = np.argmax(action_values.cpu().data.numpy())
        return action

    def val(self, state, action):
        '''return the estimated Q value of state action pair'''
        state = torch.FloatTensor(state).to(self.device)
        # action = torch.LongTensor(action).to(self.device)
        q_value = self.q_local(state)[action]
        return q_value.item()


    def step(self, state, action, reward, next_state, mask):
        '''step on transition'''
        # transition: (state, action, reward, next_state, mask)
        self.buffer.add(state, action, reward, next_state, mask)
        self.eps.update()

    def update(self):
        '''sample batch of experience tuple and update q network'''

        # Sample replay buffer
        batch = self.buffer.sample(discrete = True)
        states, actions, rewards, next_states, not_done = batch

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.q_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        Q_targets = rewards + (self.gamma * Q_targets_next * not_done)

        # Get expected Q values from local model
        Q_expected = self.q_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.q_optimizer.zero_grad()
        loss.backward()
        self.q_optimizer.step()

        # ------------------- update target Q network ------------------- #
        soft_update(self.q_local, self.q_target, self.tau)

        return loss
