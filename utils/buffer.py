import torch
import numpy as np
import random
from collections import namedtuple, deque


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, device):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.device = device

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self, batch_size = None, discrete = False):
        """Randomly sample a batch of experiences from memory."""
        if batch_size is None:
            batch_size = self.batch_size
        experiences = random.sample(self.memory, k=batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        if discrete:
            actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        else:
            actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones)

    # def sample_recent(self, batch_size = None):
    #     """Randomly sample from recent experiences"""
    #     if batch_size is None:
    #         batch_size = self.batch_size
    #     experiences = random.sample(self.memory[-5000:], k=batch_size)
    #
    #     states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
    #     actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
    #     rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
    #     next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
    #     dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
    #
    #     return (states, actions, rewards, next_states, dones)


    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

    def enough_sample(self):
        """If num samples >= batch size"""
        if len(self.memory) >= self.batch_size:
            return True
        else:
            return False

    def mean_state(self):
        """Return the mean state of all states in buffer"""
        return np.mean([e.state for e in self.memory if e is not None], axis = 0)

class Rollout:
    '''
    Collect rollout data, then yield them in random batches
    '''
    def __init__(self, device):
        self.device = device

        self.all_states = []
        self.all_actions = []
        self.all_rewards = []
        self.all_next_states = []
        self.all_dones = []

    def add(self, states, actions, rewards, next_states, gamma):
        '''add transitions'''
        for s, a, r, n_s, g in zip(states, actions, rewards, next_states, gamma):
            self.all_states.append(s)
            self.all_actions.append(a)
            self.all_rewards.append(r)
            self.all_next_states.append(n_s)
            self.all_dones.append(g)

    def sample(self, batch_size):
        '''Shuffle the data collected then sample in batch_size'''
        self.all_states = np.asarray(self.all_states)
        self.all_actions = np.asarray(self.all_actions)
        self.all_rewards = np.asarray(self.all_rewards)
        self.all_next_states = np.asarray(self.all_next_states)
        self.all_dones = np.asarray(self.all_dones)
        # indices = np.arange(len(self.all_states))
        # indices = np.random.permutation(indices)
        l = int(len(self.all_states))
        indices = list(range(l))
        random.shuffle(indices)
        # batches = indices[:len(indices) // batch_size * batch_size].reshape(-1, batch_size)
        # batches = indices.reshape(len(indices) // batch_size, batch_size)
        for j in range(0, len(indices), batch_size):
            batch_idx = indices[j:j + batch_size]
            obs = torch.from_numpy(self.all_states[batch_idx]).float().to(self.device)
            a = torch.from_numpy(self.all_actions[batch_idx]).float().to(self.device)
            r = torch.from_numpy(self.all_rewards[batch_idx]).float().to(self.device)
            nobs = torch.from_numpy(self.all_next_states[batch_idx]).float().to(self.device)
            d = torch.from_numpy(self.all_dones[batch_idx]).float().to(self.device)
            yield (obs, a, r, nobs, d)
        r = len(indices) % batch_size
        if r:
            batch_idx = indices[-r:]
            obs = torch.from_numpy(self.all_states[batch_idx]).float().to(self.device)
            a = torch.from_numpy(self.all_actions[batch_idx]).float().to(self.device)
            r = torch.from_numpy(self.all_rewards[batch_idx]).float().to(self.device)
            nobs = torch.from_numpy(self.all_next_states[batch_idx]).float().to(self.device)
            d = torch.from_numpy(self.all_dones[batch_idx]).float().to(self.device)
            yield (obs, a, r, nobs, d)

if __name__ == "__main__":
    pass
