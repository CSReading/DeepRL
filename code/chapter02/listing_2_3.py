import gym
import numpy as np

def iterate_value_function(v_inp, gamma, env):
    ret = np.zeros(env.nS)
    for sid in range(env.nS):
        temp_v = np.zeros(env.nA)
        for action in range(env.nA):
            for (prob, dst_state, reward, is_final) in env.P[sid][action]:
                temp_v[action] += prob*(reward + gamma*v_inp[dst_state]*(not is_final))
        ret[sid] = max(temp_v)
    return ret

def build_greedy_policy(v_inp, gamma, env):
    new_policy = np.zeros(env.nS)
    for state_id in range(env.nS):
        profits = np.zeros(env.nA)
        for action in range(env.nA):
            for (prob, dst_state, reward, is_final) in env.P[state_id][action]:
                profits[action] += prob*(reward + gamma*v_inp[dst_state])  # v[dst_state] は v_inp のタイポ?
        new_policy[state_id] = np.argmax(profits)
    return new_policy

env = gym.make('Taxi-v3')
gamma = 0.9
cum_reward = 0
n_rounds = 500
env.reset()
for t_rounds in range(n_rounds):
    # init env and value function
    observation = env.reset()
    v = np.zeros(env.nS)

    # solve MDP
    for _ in range(100):
        v_old = v.copy()
        v = iterate_value_function(v, gamma, env)
        if np.all(v == v_old):
            break
    policy = build_greedy_policy(v, gamma, env).astype(int)  # np.int だと warning が出るので int に直した

    # apply policy
    for t in range(1000):
        action = policy[observation]
        observation, reward, done, info = env.step(action)
        cum_reward += reward
        if done:
            break
    if t_rounds % 50 == 0 and t_rounds > 0:
        print(cum_reward * 1.0 / (t_rounds + 1))
env.close()
