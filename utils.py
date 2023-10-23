"""Implements database utilities."""

from typing import Any

import collections
import numpy as np

Batch = collections.namedtuple(
    'Batch',
    ['observations', 'actions', 'rewards', 'masks', 'next_observations', 'logprobs'])

class Database(object):
    """Implements the database object.

    Attributes:
        state_dim: observation dimensions.
        action_dim: action dimensions.
        discount: discount factor for agent.
        curr_cost: current cost incurred by the agent.
        observations: states of the agent.
        actions: actions of the agent.
        rewards: rewards of the agent.
        next_obervations: next states of the agent.
        masks: termination flags of the agent.
        logprobs: log probabilities of actions of the agent.
    """
    def __init__(self, env: Any, discount: float):
        """Initializes the databaase object."""
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.discount = discount
        self.curr_cost = 0

    def normalize(self, arr: np.ndarray) -> np.ndarray:
        """Normalizes the input array to 0 mean and unit variance."""
        if isinstance(arr, list):
            arr = np.asarray(arr)
        arr = (arr - np.mean(arr)) / (np.std(arr) + 1e-7)
        return arr

    def compute_monte_carlo(self, arr, dones):
        """Computes true monte carlo returns."""
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
        """Samples a batch of transitions."""
        new_rewards = self.normalize(self.compute_monte_carlo(self.rewards,
                                                              self.masks))
        return Batch(observations=self.observations,
                     actions=self.actions,
                     masks=self.masks,
                     rewards=new_rewards,
                     next_observations=self.next_observations,
                     logprobs=self.logprobs.squeeze(-1))

    def clear(self):
        """Clears the buffer."""
        self.observations = np.zeros((0, self.state_dim))
        self.next_observations = np.zeros((0, self.state_dim))
        self.actions = np.zeros((0, self.action_dim))
        self.masks = np.zeros((0, 1))
        self.rewards = np.zeros((0, 1))
        self.logprobs = np.zeros((0, 1))

    def push(self, state, action, reward, done, next_state, logprob):
        """Pushes samples into the replay buffer."""
        self.observations = np.vstack([self.observations, state])
        self.next_observations = np.vstack([self.next_observations, next_state])
        self.actions = np.vstack([self.actions, action])
        self.masks = np.vstack([self.masks, done])
        self.rewards = np.vstack([self.rewards, reward])
        self.logprobs = np.vstack([self.logprobs, logprob])
