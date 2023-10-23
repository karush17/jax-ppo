## PPO Lagrangian in JAX

This repository implements PPO in JAX. Implementation is tested on the safety-gym benchmark.

## Usage

Install dependencies using the following-

```
pip install -r requirements.txt
```

Install safety-gym (after installing mujoco-py) using the following-

```
git clone https://github.com/openai/safety-gym.git
cd safety-gym
pip install -e .
```

Train the PPO agent using the following-

```
python train.py --env=Safexp-CarGoal1-v0
```

Results will be stored in the `logs` folder. To create a plot run the following-

```
python plot.py
```


## Citation

In case you find the code helpful then please cite the following-

```
@misc{ppolag,
  author = {Suri, Karush},
  title = {{PPO Lagrangian in JAX.}},
  url = {https://github.com/karush17/jax-ppo},
  year = {2021}
}
```