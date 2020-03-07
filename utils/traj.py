import numpy as np

def generate_intervals(n, l):
    '''
    generate l number of index at equal interval in n timesteps
    :param n: number of timestep
    :param l: number of index
    :return itv: list of index
    '''
    a = n // l
    itv = [a * i for i in range(l)]
    return itv

def eval_policy(env, agent, gamma, num_eval):
    '''
    evaluate the policy over num_eval trajectories

    :param env: environment
    :param agent: agent
    :param gamma: args.gamma discount factor
    :param num_eval: number of evaluation trajectories

    :return avg_reward: average reward achieved over num_eval trajectories
    '''
    avg_reward = 0
    for num_traj in range(num_eval):
        episode_reward = 0
        state = env.reset()
        done = False
        while not done:
            action = agent.act(state, eval = True)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state
        avg_reward += episode_reward

    avg_reward /= num_eval
    return avg_reward

def eval_trajs(env, agent, gamma, num_eval, num_point):
    '''
    collect trajectories; compute q values and estimated returns of points in the trajectories
    used to examine predicted Q value and actual return (overestimation of Q)

    :param env: environment
    :param agent: agent
    :param gamma: args.gamma discount factor
    :param num_eval: number of evaluation trajectories
    :param num_point: number of evaluation points for q values and estimated return in each trajectory

    :return avg_reward: average reward achieved on the evaluation trajectories
    :return np.mean(q_values): mean of all the q values collected at points in trajs
    :return np.mean(returns): mean of all the returns of points in trajs
    '''
    all_states = []
    all_actions = []
    all_rewards = []
    avg_reward = 0
    n_trajs_to_save = num_eval
    while len(all_states) < n_trajs_to_save:
        this_traj_states, this_traj_actions, this_traj_rewards = [], [], []
        episode_reward = 0
        state = env.reset()
        done = False
        while not done:
            action = agent.act(state, eval = True)
            next_state, reward, done, _ = env.step(action)
            this_traj_states.append(state)
            this_traj_actions.append(action)
            this_traj_rewards.append(reward)
            episode_reward += reward
            state = next_state

        this_traj_states = np.array(this_traj_states)
        this_traj_actions = np.array(this_traj_actions)
        this_traj_rewards = np.array(this_traj_rewards)
        all_states.append(this_traj_states)
        all_actions.append(this_traj_actions)
        all_rewards.append(this_traj_rewards)
        avg_reward += episode_reward

    avg_reward /= num_eval

    # --------------- PROCESSING TRAJECTORIES --------------- #
    q_values = []
    returns = []
    for i_traj in range(len(all_states)):
        this_traj_states = all_states[i_traj]
        this_traj_actions = all_actions[i_traj]
        this_traj_rewards = all_rewards[i_traj]
        if len(this_traj_states) != len(this_traj_rewards):
            raise ValueError("state length does not match reward length in trajectory :(")
            break
        itvs = generate_intervals(len(this_traj_states), num_point)
        for itv in itvs:
            q = agent.val(this_traj_states[itv], this_traj_actions[itv])
            rews = this_traj_rewards[itv:]
            discounts = [gamma**i for i in range(len(rews))]
            R = sum([a*b for a,b in zip(rews, discounts)])

            q_values.append(q)
            returns.append(R)

    if len(q_values) != (num_eval * num_point):
        raise ValueError("number of q values does not look right :(")
    if len(returns) != (num_eval * num_point):
        raise ValueError("number of returns does not look right :(")

    return avg_reward, np.mean(q_values), np.mean(returns)


if __name__ == "__main__":
    print (generate_intervals(72, 10))
