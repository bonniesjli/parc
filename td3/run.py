import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
from tensorboardX import SummaryWriter
import mujoco_py

from td3 import TD3
from utils.loggin import Logger
from utils.traj import eval_policy, eval_trajs

parser = argparse.ArgumentParser(description='Parc')
parser.add_argument('--env', default="HalfCheetah-v2",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument("--expl_noise", type=float, default=0.1, metavar='G',
                    help='std of Gaussian exploration noise')
parser.add_argument('--seed', type=int, default=10, metavar='N',
                    help='random seed (default: 10)')
parser.add_argument('--num_steps', type=int, default=100000, metavar='N',
                    help='maximum number of steps (default: 100000)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 40000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='network size (net_size) for Actor and Model (default: 256)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='batch size (default: 128)')
parser.add_argument('--buffer_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--update_freq', type=int, default=1, metavar='N',
                    help='number of simulations steps per update (default: 1)')
parser.add_argument('--dir', default="runs",
                    help='loggin directory to create folder containing tensorboard and loggin files')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
args = parser.parse_args()

env = gym.make(args.env)

env.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.deterministic=True

agent = TD3(env.observation_space.shape[0], env.action_space.shape[0], env.action_space, args)
LOG_DIR = '{}/{}_TD3_{}'.format(args.dir, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env)
writer = SummaryWriter(logdir=LOG_DIR)
LOG = Logger(LOG_DIR)
LOG.create("q_values")
LOG.create("estimated_r")
LOG.create("test_reward")
LOG.create("train_reward")

total_numsteps = 0
for i_episode in itertools.count(1):

    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()

    while not done:
        if total_numsteps < args.start_steps:
            action = env.action_space.sample()  # Sample random action
        else:
            action = agent.act(state)  # Sample action from policy

        next_state, reward, done, _ = env.step(action) # Step
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward
        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = 1 if episode_steps == env._max_episode_steps else float(not done)

        agent.step(state, action, reward, next_state, mask)
        if total_numsteps >= args.start_steps and total_numsteps % args.update_freq == 0:
            agent.update()

        state = next_state

    if total_numsteps > args.num_steps:
        break

    writer.add_scalar('reward/train_per_stp', episode_reward, total_numsteps)
    LOG.log("train_reward", episode_reward, total_numsteps)
    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))

    # ---------- Evaluation loop every 10 episodes ---------- #
    # loggin Q vs estimated return here (see utils.traj for trajs function)
    if i_episode % 10 == 0 and args.eval == True:
        test_episodes = 5
        num_points = 10
        avg_reward, mean_q, mean_return = eval_trajs(env, agent, args.gamma, test_episodes, num_points)
        print ("mean q value: ", mean_q)
        print ("mean return: ", mean_return)
        print("Test Episodes: {}, Avg. Reward: {}".format(test_episodes, round(avg_reward, 2)))
        writer.add_scalar('reward/test_per_stp', avg_reward, total_numsteps)
        writer.add_scalar('return/q_values', mean_q, total_numsteps)
        writer.add_scalar('return/estimated_r', mean_return, total_numsteps)
        LOG.log("test_reward", avg_reward, total_numsteps)
        LOG.log("q_values", mean_q, total_numsteps)
        LOG.log("estimated_r", mean_return, total_numsteps)

env.close()
LOG.close()
writer.close()
