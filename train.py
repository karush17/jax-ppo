"""Implements the training protocol."""

from typing import Tuple

import numpy as np
import tqdm
import gym

from absl import app, flags
from ml_collections import config_flags
from utils import Database
from ppo_learner import Learner
from logger import Logger

FLAGS = flags.FLAGS
flags.DEFINE_string('env_name', 'Safexp-CarGoal1-v0', 'Environment name.')
flags.DEFINE_string('save_dir', './logs', 'Logging dir.')
flags.DEFINE_integer('seed', 10, 'Random seed.')
flags.DEFINE_float('discount', 0.99, 'Discount factor.')
flags.DEFINE_integer('epoch_steps', 4000, 'Train epoch after n steps.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('max_steps', int(1e6), 'Number of training steps.')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
config_flags.DEFINE_config_file(
    'config',
    'config.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)


def make_env_and_dataset(env_name: str, seed: int) -> Tuple[gym.Env, Database]:
    """Creates an environment and the dataset."""
    env = gym.make(env_name)
    env.seed(seed)
    data = Database(env, FLAGS.discount)
    data.clear()
    return env, data

def main(_):
    """Main function for training."""
    env, dataset = make_env_and_dataset(FLAGS.env_name, FLAGS.seed)
    logger = Logger(FLAGS.save_dir, FLAGS.env_name, FLAGS.seed, FLAGS.max_steps)

    kwargs = dict(FLAGS.config)
    learner = Learner(FLAGS.seed, env.observation_space.sample()[np.newaxis],
                           env.action_space.sample()[np.newaxis], **kwargs)

    for i in tqdm.tqdm(range(1, int(FLAGS.max_steps/1000) + 1),
                       smoothing=0.1,
                       disable=not FLAGS.tqdm):
        observation, done = env.reset(), False
        total_reward = 0
        while not done:
            action, logprob = learner.evaluate_actions(observation)
            next_state, reward, done, _ = env.step(action)
            dataset.push(observation, action, reward, done, next_state, logprob)
            observation = next_state
            total_reward += reward

        if i % (FLAGS.epoch_steps/1000) == 0:
            batch = dataset.sample()
            learner.update(batch)
            dataset.clear()

        logger.log([total_reward])
    dataset.clear()


if __name__ == '__main__':
    app.run(main)
