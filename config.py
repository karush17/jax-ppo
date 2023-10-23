"""Implements agent configurations."""

import ml_collections


def get_config():
    """Stores agent hyperparameter configurations."""
    config = ml_collections.ConfigDict()
    config.rew_lr = 1e-3
    config.actor_lr = 3e-4
    config.clip = 0.2
    config.epochs = 80
    config.hidden_dims = (256, 256)
    return config
