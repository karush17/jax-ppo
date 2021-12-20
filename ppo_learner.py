import jax
import jax.numpy as jnp
import numpy as np
import optax
import functools

from typing import Optional, Sequence, Tuple
from tensorflow_probability.substrates import jax as tfp
from actor import NormalTanhPolicy, Lagrange, Estimator
from updates import update_actor, update_rew, _evaluate_actions
from utils import Batch
from common import InfoDict, Model, PRNGKey

tfd = tfp.distributions
tfb = tfp.bijectors

class Learner(object):
    def __init__(self,
                 seed: int,
                 observations: jnp.ndarray,
                 actions = jnp.ndarray,
                 actor_lr: float = 3e-4,
                 rew_lr: float = 1e-3,
                 clip: float = 0.2,
                 epochs: float = 50,
                 hidden_dims: Sequence[int] = (256, 256)):

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
        rng, actions, logprobs = _evaluate_actions(self.rng, self.actor_old.apply_fn, self.actor_old.params,\
                                                                 observations)
        self.rng = rng
        actions = np.asarray(actions)
        logprobs = np.asarray(logprobs)
        return actions, logprobs

    def update(self, batch: Batch) -> InfoDict:
        self.step += 1

        for upd in range(self.iters):
            self.rng, key = jax.random.split(self.rng)
            self.rng, self.actor, self.actor_old, actor_info = update_actor(self.rng, self.actor, \
                                                self.actor_old, self.rew_est, self.scalars, batch)
            
            self.rng, self.rew_est, rew_info = update_rew(self.rng, self.rew_est, batch)
        
        self.actor_old = self.actor_old.replace(params=self.actor.params)
