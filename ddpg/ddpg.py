import numpy as np
import copy
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

from model import *
from utils.buffer import ReplayBuffer
from utils.update import soft_update, hard_update

class DDPG():
    def __init__(self, state_size, action_size, action_space, args,
                 policy_noise=0.2,
		         noise_clip=0.5):
        self.device = torch.device("cuda" if args.cuda else "cpu")
        self.buffer = ReplayBuffer(args.buffer_size, args.batch_size, self.device)

        self.action_size = action_size
        self.gamma = args.gamma
        self.tau = args.tau
        self.start_steps = args.start_steps

        self.total_it = 0
        self.max_action = float(action_space.high[0])
        self.policy_noise = policy_noise * self.max_action # Target policy smoothing is scaled wrt the action scale
        self.noise_clip = noise_clip * self.max_action
        self.expl_noise = args.expl_noise

        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()

        self.policy = Actor(state_size, action_space.shape[0], args.hidden_size, action_space).to(self.device)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=args.lr)
        self.policy_target = copy.deepcopy(self.policy)

        self.critic_local = Critic(state_size, action_space.shape[0], args.hidden_size).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=args.lr)
        self.critic_target = copy.deepcopy(self.critic_local)


    def act(self, state, eval = False):
        '''return action given state
        :param state (np.ndarray): state
        :param eval (bool): whether if we are evaluating policy. set to True in utils.traj.py
        :return action (np.ndarray): action with Gaussian noise if not eval
        '''
        state = torch.FloatTensor(state).to(self.device)
        if not eval:
            action = (self.policy(state).detach().cpu().numpy()
                      + np.random.normal(0, self.max_action * self.expl_noise, size=self.action_size)).clip(-self.max_action, self.max_action)
        else:
            action = self.policy(state).detach().cpu().numpy()
        return action

    def val(self, state, action):
        '''return the estimated Q value of state action pair'''
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        v = self.critic_local.forward_one(state, action)
        return v.item()


    def step(self, state, action, reward, next_state, mask):
        '''step on transition, called after every transition'''
        self.buffer.add(state, action, reward, next_state, mask)

    def update(self):
        '''update critic and policy'''
        self.total_it += 1

        # Sample replay buffer
        batch = self.buffer.sample()
        state, action, reward, next_state, not_done = batch

        # Compute the target Q value
        target_Q = self.critic_target(next_state, self.policy_target(next_state))
        target_Q = reward + (not_done * self.gamma * target_Q).detach()

        # Get current Q estimate
        current_Q = self.critic_local(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss
        actor_loss = -self.critic_local(state, self.policy(state)).mean()

        # Optimize the actor
        self.policy_optimizer.zero_grad()
        actor_loss.backward()
        self.policy_optimizer.step()

        # Update the frozen target models
        soft_update(self.critic_local, self.critic_target, self.tau)
        soft_update(self.policy, self.policy_target, self.tau)

        return critic_loss.item(), actor_loss.item()
