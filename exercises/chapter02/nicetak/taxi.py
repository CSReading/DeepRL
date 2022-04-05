import random

import gym
import numpy as np


class Taxi:
    def __init__(self, gamma=0.7, alpha=0.2, epsilon=0.1, n_episode=1000, method="Q-learning"):

        # Environment Setup
        self.env = gym.make("Taxi-v3")
        self.env.reset()

        # Parameters
        self.gamma = gamma  # discount factor
        self.alpha = alpha  # learning rate
        self.epsilon = epsilon  # epsilon greedy

        self.n_episode = n_episode

        # Initialize Q
        self.Q = np.zeros(
            [self.env.observation_space.n, self.env.action_space.n])

        # Method "Q-learing" or "SARSA"
        self.method = method

        # Logging
        self.log_Q = []
        self.log_reward = []

    def train(self, verbose=False):

        for episode in range(self.n_episode):
            if self.method == "Q-learning":
                self.Q_one_episode()
            else:
                raise ValueError("self.method should be 'Q-learning'")

            if (episode % 100 == 0) and verbose:
                print(f"Episode {episode} Total Reward: {self.log_reward[-1]}")

    def Q_one_episode(self):
        done = False
        total_reward = 0
        state = self.env.reset()

        while not done:
            if random.uniform(0, 1) < self.epsilon:
                action = self.env.action_space.sample()  # Explore state space
            else:
                action = np.argmax(self.Q[state])  # Exploit learned values
            next_state, reward, done, info = self.env.step(
                action)  # invoke Gym

            next_max = np.max(self.Q[next_state])
            old_value = self.Q[state, action]
            new_value = old_value + self.alpha * \
                (reward + self.gamma * next_max - old_value)
            self.Q[state, action] = new_value
            total_reward += reward
            state = next_state

        self.log_Q.append(self.Q)
        self.log_reward.append(total_reward)
