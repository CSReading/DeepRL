# Q learning for OpenAI Gym Taxi environment
import gym
import numpy as np
import random

# Environment Setup
env = gym.make("Taxi-v3")
env.reset()
env.render()

# Q[state, action] table implementation
Q = np.zeros([env.observation_space.n, env.action_space.n])
gamma = 0.7  # discount factor
alpha = 0.2  # learning rate
epsilon = 0.1  # epsilon greedy

for episode in range(1000):
    done = False
    total_reward = 0
    state = env.reset()
    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore state space
        else:
            action = np.argmax(Q[state])  # Exploit learned values
        next_state, reward, done, info = env.step(action)  # invoke Gym
        next_max = np.max(Q[next_state])
        old_value = Q[state, action]

        new_value = old_value + alpha * (reward + gamma * next_max - old_value)

        Q[state, action] = new_value
        total_reward += reward
        state = next_state

    if episode % 100 == 0:
        print(f"Episode {episode} Total Reward: {total_reward}")
