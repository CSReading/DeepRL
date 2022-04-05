# SARSA for OpenAI Gym Taxi environment
import random

import gym
import numpy as np

# Environment Setup
env = gym.make("Taxi-v3")
env.reset()
env.render()

# Q[state,action] table implementation
Q = np.zeros([env.observation_space.n, env.action_space.n])
gamma = 0.7  # discount factor
alpha = 0.2  # learning rate
epsilon = 0.1  # epsilon greedy

for episode in range(1000):
    done = False
    total_reward = 0
    current_state = env.reset()
    if random.uniform(0, 1) < epsilon:
        current_action = env.action_space.sample()  # Explore state space
    else:
        current_action = np.argmax(Q[current_state])  # Exploit learned values
    while not done:
        next_state, reward, done, info = env.step(current_action)  # invoke Gym
        if random.uniform(0, 1) < epsilon:
            next_action = env.action_space.sample()  # Explore state space
        else:
            next_action = np.argmax(Q[next_state])  # Exploit learned values
        sarsa_value = Q[next_state, next_action]
        old_value = Q[current_state, current_action]

        new_value = old_value + alpha * \
            (reward + gamma * sarsa_value - old_value)

        Q[current_state, current_action] = new_value
        total_reward += reward
        current_state = next_state
        current_action = next_action
    if episode % 100 == 0:
        print(f"Episode {episode} Total Reward: {total_reward}")
