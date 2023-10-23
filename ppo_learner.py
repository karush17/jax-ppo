"""Implements the PPO agent."""

from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax

from tensorflow_probability.substrates import jax as tfp
from actor import NormalTanhPolicy, Estimator
from updates import update_actor, update_rew, _evaluate_actions
from utils import Batch
from common import InfoDict, Model

tfd = tfp.distributions
tfb = tfp.bijectors

class Learner(object):
    """Implements the learner module for PPO.
    
    Attributes:
        state_dim: number of dimensions in observations.
        action_dim: number of dimensions in action.
        iter: number of iterations.
        scalars: scalar values for storage.
        actor: policy network.
        actor_old: old actor weights.
        rew_est: value network estimator.
        rng: jax random key.
        step: current training step.
    """
    def __init__(self,
                 seed: int,
                 observations: jnp.ndarray,
                 actions = jnp.ndarray,
                 actor_lr: float = 3e-4,
                 rew_lr: float = 1e-3,
                 clip: float = 0.2,
                 epochs: float = 50,
                 hidden_dims: Sequence[int] = (256, 256)):
        """Initializes the learner object."""

        observations = jnp.array(observations)
        actions = jnp.array(actions)
        self.state_dim = observations.shape[-1]
        self.action_dim = actions.shape[-1]
        self.iters = epochs
        self.scalars = {
            'clip': clip
        }

        rng = jax.random.PRNGKey(seed)
        rng, actor_old_key, actor_key, rew_key = jax.random.split(rng, 4)
        actor_def = NormalTanhPolicy(hidden_dims, self.action_dim)
        actor_old = Model.create(actor_def,
                             inputs=[actor_old_key, actor_old_key, observations],
                             tx=optax.adam(learning_rate=actor_lr))
        actor = Model.create(actor_def,
                             inputs=[actor_key, actor_key, observations],
                             tx=optax.adam(learning_rate=actor_lr))
        rew_def = Estimator(hidden_dims, 1)
        rew_est = Model.create(rew_def,
                             inputs=[rew_key, observations],
                             tx=optax.adam(learning_rate=rew_lr))

        self.actor = actor
        self.actor_old = actor_old
        self.rew_est = rew_est
        self.rng = rng
        self.step = 1

    def evaluate_actions(self, observations: np.ndarray) -> jnp.ndarray:
        """Samples an action given the state."""
        rng, actions, logprobs = _evaluate_actions(self.rng,
                                                   self.actor_old.apply_fn,
                                                   self.actor_old.params,
                                                   observations)
        self.rng = rng
        actions = np.asarray(actions)
        logprobs = np.asarray(logprobs)
        return actions, logprobs

    def update(self, batch: Batch) -> InfoDict:
        """Updates the parameters of the PPO learner."""
        self.step += 1
        for _ in range(self.iters):
            self.rng, _ = jax.random.split(self.rng)
            self.rng, self.actor, self.actor_old, _ = update_actor(self.rng,
                                                                   self.actor,
                                                                   self.actor_old,
                                                                   self.rew_est,
                                                                   self.scalars,
                                                                   batch)            
            self.rng, self.rew_est, _ = update_rew(self.rng, self.rew_est, batch)
        self.actor_old = self.actor_old.replace(params=self.actor.params)
