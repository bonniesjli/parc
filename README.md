# Parc - A Learning Agents Framework

A exceptionally clean codebase for reinforcement learning using pytorch, designed for simplicity and efficiency.
<p float="left">
  <img src="/img/ant.gif" width="250" />
  <img src="/img/pushblock.gif" width="250" />
  <img src="/img/dqn.gif" width="250" />
</p>

### Algorithms
* DQN: Deep Q Learning
* DDPG: Deep Deterministic Policy Gradient
* TD3: Twin Delayed Deep Deterministic Policy Gradient
* [ ] PPO: Proximal Policy Optimization
* [ ] SAC: Soft Actor Critic

### To Run
```
usage: run.py  [--env ENV_NAME] [--eval EVAL]
               [--gamma G] [--tau G] [--lr G]
               [--seed N]
               [--num_steps N] [--start_steps N] [--hidden_size N]
               [--batch_size N] [--buffer_size N]
               [--update_freq N]
               [--dir DIRECTORY]
               [--cuda]
```
#### Getting Started:
See `dqn/getting_started.ipynb` for a quick tutorial on how to interface agents with environments.
#### Quick Examples:
```
python dqn/run.py --env CartPole-v1 --eps_decays 5000 --num_steps 10000
python ddpg/run.py --env Reacher-v2 --start_steps 500 --num_steps 10000
python td3/run.py --env Reacher-v2 --start_steps 500 --num_steps 10000
```
See `exp_scripts.sh` for more sample commands. <br>
See `exp.sh` for running on cloud or cluster.

### Installation

More dependencies to be added

```
git clone https://github.com/bonniesjli/parc.git
cd parc
python setup.py develop
```

### Custom Environment

Learning Agents are modularized to be easily used in custom environments. <br>
Learning loop would loop similar to this:
```
(argparse hyperparams)
env = CustomEnvironment()
agent = DQN/DDPG/TD3(state_size, action_size, action_space, args)
for _ in num_episodes:
  while not done:
    action = agent.act(state)
    next_state, reward, done = env.step(action)
    agent.step(state, action, reward, next_state, 1 - done)
    agent.update()
```
