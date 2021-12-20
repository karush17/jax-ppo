import os
import collections
import numpy as np
import gym
import h5py

from typing import Tuple, Union
from tqdm import tqdm

Batch = collections.namedtuple(
    'Batch',
    ['observations', 'actions', 'rewards', 'masks', 'next_observations', 'logprobs'])

class Database(object):
    def __init__(self, env, discount):
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.discount = discount
        self.curr_cost = 0

    def normalize(self, arr):
        if isinstance(arr, list):
            arr = np.asarray(arr)
        arr = (arr - np.mean(arr)) / (np.std(arr) + 1e-7)
        return arr
    
    def compute_monte_carlo(self, arr, dones):
        new_arr = []
        disc_val = 0
        for val, is_terminal in zip(reversed(arr), reversed(dones)):
            if is_terminal:
                disc_val = 0
            disc_val = val + (self.discount*disc_val)
            new_arr.insert(0, disc_val)
        new_arr = self.normalize(new_arr)
        return new_arr

    def sample(self):
        new_rewards = self.normalize(self.compute_monte_carlo(self.rewards, self.masks))
        return Batch(observations=self.observations,
                     actions=self.actions,
                     masks=self.masks,
                     rewards=new_rewards,
                     next_observations=self.next_observations,
                     logprobs=self.logprobs.squeeze(-1))

    def clear(self):
        self.observations = np.zeros((0, self.state_dim))
        self.next_observations = np.zeros((0, self.state_dim))
        self.actions = np.zeros((0, self.action_dim))
        self.masks = np.zeros((0, 1))
        self.rewards = np.zeros((0, 1))
        self.logprobs = np.zeros((0, 1))
    
    def push(self, state, action, reward, done, next_state, logprob):
        self.observations = np.vstack([self.observations, state])
        self.next_observations = np.vstack([self.next_observations, next_state])
        self.actions = np.vstack([self.actions, action])
        self.masks = np.vstack([self.masks, done])
        self.rewards = np.vstack([self.rewards, reward])
        self.logprobs = np.vstack([self.logprobs, logprob])

