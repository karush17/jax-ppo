import functools
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from typing import Any, Callable, Optional, Sequence, Tuple
from tensorflow_probability.substrates import jax as tfp
from utils import Batch
from common import MLP, default_init, InfoDict, Model, Params, PRNGKey

tfd = tfp.distributions
tfb = tfp.bijectors

LOG_STD_MIN = -10.0
LOG_STD_MAX = 2.0


class Estimator(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 temperature: float = 1.0,
                 training: bool = False) -> jnp.ndarray:
        outputs = MLP(self.hidden_dims,
                      activate_final=True,
                      dropout_rate=self.dropout_rate)(observations,
                                                      training=training)

        outs = nn.Dense(self.action_dim,
                           kernel_init=default_init())(outputs)
        return outs


class NormalTanhPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    state_dependent_std: bool = True
    dropout_rate: Optional[float] = None
    log_std_scale: float = 1.0
    log_std_min: Optional[float] = None
    log_std_max: Optional[float] = None
    tanh_squash_distribution: bool = False

    @nn.compact
    def __call__(self,
                 key: PRNGKey,
                 observations: jnp.ndarray,
                 actions = None,
                 training: bool = False) -> tfd.Distribution:
        outputs = MLP(self.hidden_dims,
                      activate_final=True,
                      dropout_rate=self.dropout_rate)(observations,
                                                      training=training)
        means = nn.Dense(self.action_dim, kernel_init=default_init())(outputs)

        if self.state_dependent_std:
            log_stds = nn.Dense(self.action_dim,
                                kernel_init=default_init(
                                    self.log_std_scale))(outputs)
        else:
            log_stds = self.param('log_stds', nn.initializers.zeros,
                                  (self.action_dim, ))

        log_std_min = self.log_std_min or LOG_STD_MIN
        log_std_max = self.log_std_max or LOG_STD_MAX
        log_stds = jnp.clip(log_stds, log_std_min, log_std_max)

        if not self.tanh_squash_distribution:
            means = nn.tanh(means)

        base_dist = tfd.Normal(loc=means, scale=jnp.exp(log_stds)) #MultivariateNormalDiag
        if self.tanh_squash_distribution:
            dist = tfd.TransformedDistribution(distribution=base_dist,
                                               bijector=tfb.Tanh())
        else:
            dist = base_dist
        if actions is None:
            actions = dist.sample(seed=key)
        logprobs = dist.log_prob(actions).sum(axis=-1)
        return actions, logprobs


class Lagrange(nn.Module):
    initial_lagrange: float = 1.0

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_lag = self.param('log_lag',
                              init_fn=lambda key: jnp.full(
                                  (), self.initial_lagrange))
        return nn.softplus(log_lag), log_lag
