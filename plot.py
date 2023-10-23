"""Plots averaged returns of the agent."""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc

rc('font',**{'family':'serif','sans-serif':['Helvetica']}, weight='bold',
   size=20)
rc('text', usetex=True)

il_name = './logs/'

env_list = ['Safexp-CarGoal1-v0']
alpha = 0.2
avg = 50
fig = plt.figure(figsize=(4,4))
count = 1

def moving_average(a: np.ndarray, n: int = 20,diff: bool = False) -> np.ndarray:
    """Implements moving average over an array."""
    if diff==True:
      for i in range(len(a)):
        a[i] = np.mean(a[i].cpu().numpy(),0)
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

for env_name in env_list:

    il_reward = []
    il_rew_max = []
    il_rew_min = []
    len_dir = len(os.listdir(il_name + env_name + '/'))
    for k in os.listdir(il_name + env_name + '/'):
      with open(il_name + env_name + '/' + k, 'r') as f:
        il_dict = json.load(f)
        il_ret_arr =moving_average(np.array(il_dict['returns']))
        if not il_reward:
          il_reward = il_rew_max = il_rew_min = il_ret_arr
        else:
          il_reward = il_reward + il_ret_arr
          il_rew_max = np.maximum(il_rew_max, il_ret_arr)
          il_rew_min = np.minimum(il_rew_min, il_ret_arr)
    il_reward = il_reward / len_dir

    plt.title(env_name, fontsize=20)
    plt.plot(np.arange(0,len(il_reward),1),il_reward, color='darkorange',
             linewidth=4, label='PPO')
    plt.fill_between(np.arange(0, len(il_reward), 1), il_rew_min,
                     il_rew_max, facecolor='darkorange',
                     alpha=alpha, linewidth=0, antialiased=True)
    plt.ylabel('Returns')
    plt.xlabel('Steps'+ r'($\times 10^{4}$)')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

plt.tight_layout()
plt.savefig('plot.png', dpi=800, bbox_inches='tight')
