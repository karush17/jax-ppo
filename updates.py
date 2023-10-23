"""Implements graidnet updates and loss functions."""

from typing import Any, Callable, Tuple

import functools
import numpy as np
import jax
import jax.numpy as jnp

from tensorflow_probability.substrates import jax as tfp
from utils import Batch
from common import InfoDict, Model, Params, PRNGKey

tfd = tfp.distributions
tfb = tfp.bijectors

@functools.partial(jax.jit, static_argnames=())
def update_actor(key: PRNGKey, actor: Model, actor_old: Model, rew_est: Model, \
            scalars: dict, batch: Batch) -> Tuple[Model, InfoDict]:
    """Updates the actor using surrogate loss function."""
    rng, key = jax.random.split(key)
    advs = jnp.squeeze(batch.rewards - rew_est(batch.observations), -1)
    clip = scalars['clip']

    def loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        _, logprobs = actor.apply_fn({'params': actor_params},
                                           key, batch.observations,
                                           actions = batch.actions)
        ratios = jnp.exp(logprobs - batch.logprobs)
        surr1 = ratios*advs
        surr2 = jnp.clip(ratios, 1-clip, 1+clip)*advs
        actor_loss = -jnp.minimum(surr1, surr2).mean()
        return actor_loss, {'actor_loss': actor_loss}

    actor, info = actor.apply_gradient(loss_fn)
    return rng, actor, actor_old, info


@functools.partial(jax.jit, static_argnames=())
def update_rew(key: PRNGKey, rew_est: Model, batch: Batch):
    """Updates the value network."""
    rng, key = jax.random.split(key)

    def loss_fn(est_params: Params):
        values = rew_est.apply_fn({'params': est_params}, batch.observations)
        loss = 0.5*jnp.square(batch.rewards - values).mean()
        return loss, {'rew_loss': loss}

    new_rew, info = rew_est.apply_gradient(loss_fn)
    return rng, new_rew, info


@functools.partial(jax.jit, static_argnames=('actor_apply_fn'))
def _evaluate_actions(
        rng: PRNGKey,
        actor_apply_fn: Callable[..., Any],
        actor_params: Params,
        observations: np.ndarray) -> Tuple[PRNGKey, jnp.ndarray]:
    """Evaluates actions for a given state."""
    rng, key = jax.random.split(rng)
    actions, logprobs = actor_apply_fn({'params': actor_params}, key, observations)
    return rng, actions, logprobs
