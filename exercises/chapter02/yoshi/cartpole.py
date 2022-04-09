import gym
import numpy as np
from math import sin, cos

gravity = 9.8
masscart = 1.0
masspole = 0.1
total_mass = masspole + masscart
length = 0.5  # actually half the pole's length
polemass_length = masspole * length
force_mag = 10.0
tau = 0.02  # seconds between state updates
kinematics_integrator = "euler"

def iterate_value_function(v_inp, gamma):
    ret = np.zeros(shape=(4, 4, 4, 4))
    for w in range(4):
        for x in range(4):
            for y in range(4):
                for z in range(4):
                    temp_v = np.zeros(2)
                    for action in [0, 1]:
                        next_discrete_state = next_state([w, x, y, z], action, pos_space, vel_space, angle_space, angular_vel_space)
                        reward: int
                        if (next_discrete_state[2] != 0):
                            reward = 1
                        else:
                            reward = 0
                        temp_v[action] += reward + gamma * v_inp[next_discrete_state[0], next_discrete_state[1],next_discrete_state[2],next_discrete_state[3]]
                    ret[w, x, y, z] = max(temp_v)
    return ret


def build_greedy_policy(v_inp, gamma):
    num_pos, num_vel, num_angle, num_angular_vel = 4,4,4,4
    new_policy = np.zeros(shape=(num_pos, num_vel, num_angle, num_angular_vel), dtype=int)
    for w in range(num_pos):
        for x in range(num_vel):
            for y in range(num_angle):
                for z in range(num_angular_vel):
                    profits = np.zeros(2)
                    for action in [0, 1]:
                        next_discrete_state = next_state([w, x, y, z], action, pos_space, vel_space, angle_space, angular_vel_space)
                        reward: int
                        if (next_discrete_state[2] != 0 and next_discrete_state[2] != num_angle-1):
                            reward = 1
                        else:
                            reward = 0
                        profits[action] += reward + gamma*v_inp[next_discrete_state[0], next_discrete_state[1],next_discrete_state[2],next_discrete_state[3]]
                    new_policy[w, x, y, z] = np.argmax(profits)
    return new_policy

def state(cont_state, pos_space, vel_space, angle_space, angular_vel_space):
    space_list = [pos_space, vel_space, angle_space, angular_vel_space]
    discrete_state = [0, 0, 0, 0]
    for i in range(4):
        discrete_state[i] = np.argmin(np.abs(space_list[i] - cont_state[i]))
    return discrete_state

def next_state(digitized_state, action, pos_space, vel_space, angle_space, angular_vel_space):
    x = pos_space[digitized_state[0]]
    x_dot = vel_space[digitized_state[1]]
    theta = angle_space[digitized_state[2]]
    theta_dot = angular_vel_space[digitized_state[3]]

    force = force_mag if action == 1 else -force_mag
    tmp = (force + polemass_length * theta_dot**2 * sin(theta)) / total_mass
    thetaacc = (gravity * sin(theta) - cos(theta) * tmp) / (length * (4.0 / 3.0 - masspole * cos(theta)**2 / total_mass))
    xacc = tmp - polemass_length * thetaacc * cos(theta) / total_mass

    x = x + tau * x_dot
    x_dot = x_dot + tau * xacc
    theta = theta + tau * theta_dot
    theta_dot = theta_dot + tau * thetaacc

    return state([x, x_dot, theta, theta_dot], pos_space, vel_space, angle_space, angular_vel_space)




env = gym.make('CartPole-v1') 
gamma = 0.9
cum_reward = 0
n_rounds = 500 
env.reset()

for episode in range(n_rounds):
    # 環境の初期化
    observation = env.reset()
    pos_space = np.linspace(-4.8, 4.8, 4)
    vel_space = np.linspace(-3.0, 3.0, 4)
    angle_space = np.linspace(-0.2, 0.2, 4)
    angular_vel_space = np.linspace(-2.0, 2.0, 4)

    v = np.zeros(shape=(4, 4, 4, 4))

    dicrete_state = state(observation, pos_space,vel_space, angle_space, angular_vel_space)

    episode_reward = 0

    # CartPoleの描画
 

    for _ in range(100): 
        v_old = v.copy()
        v = iterate_value_function(v, gamma) 
        if np.all(v == v_old):
            break
        policy = build_greedy_policy(v, gamma).astype(np.int)
    
    # apply policy
    for t in range(1000):
        env.render()
        discrete_state = state(observation, pos_space, vel_space, angle_space, angular_vel_space)
        action = policy[discrete_state[0], discrete_state[1], discrete_state[2], discrete_state[3]]
        observation, reward, done, info = env.step(action)
        cum_reward += reward
        if done:
            break

    print(f"episdoe: {episode} {cum_reward * 1.0 / (episode + 1)}")             
env.close()