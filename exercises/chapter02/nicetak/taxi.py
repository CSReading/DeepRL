import random

import gym
import numpy as np


class Taxi:
    def __init__(self):

        # Environment Setup
        self.env = gym.make("Taxi-v3")
        self.env.reset()

        # Initialize Q
        self.Q = np.zeros(
            [self.env.observation_space.n, self.env.action_space.n])

        # Logging
        self.log_Q = []
        self.log_reward = []

    def train(self, method="Q-learning", verbose=False, gamma=0.7, alpha=0.2, epsilon=0.1, n_episode=1000):

        # Parameters
        self.gamma = gamma  # discount factor
        self.alpha = alpha  # learning rate
        self.epsilon = epsilon  # epsilon greedy

        self.n_episode = n_episode

        for episode in range(self.n_episode):
            if method == "Q-learning":
                self.Q_one_episode()
            elif method == "SARSA":
                self.SARSA_one_episode()
            else:
                raise ValueError("self.method should be 'Q-learning'")

            if (episode % (self.n_episode // 10) == 0) and verbose:
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

    def SARSA_one_episode(self):
        done = False
        total_reward = 0
        current_state = self.env.reset()
        if random.uniform(0, 1) < self.epsilon:
            current_action = self.env.action_space.sample()  # Explore state space
        else:
            # Exploit learned values
            current_action = np.argmax(self.Q[current_state])
        while not done:
            next_state, reward, done, info = self.env.step(
                current_action)  # invoke Gym
            if random.uniform(0, 1) < self.epsilon:
                next_action = self.env.action_space.sample()  # Explore state space
            else:
                # Exploit learned values
                next_action = np.argmax(self.Q[next_state])
            sarsa_value = self.Q[next_state, next_action]
            old_value = self.Q[current_state, current_action]

            new_value = old_value + self.alpha * \
                (reward + self.gamma * sarsa_value - old_value)

            self.Q[current_state, current_action] = new_value
            total_reward += reward
            current_state = next_state
            current_action = next_action

        self.log_Q.append(self.Q)
        self.log_reward.append(total_reward)
