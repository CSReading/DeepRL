import random

import gym
import numpy as np


class Cartpole:
    def __init__(self, n_digitized=6):

        # Environment Setup
        self.env = gym.make("CartPole-v1")
        self.env.reset()

        self.n_digitized = n_digitized

        # Initialize
        self.n_state = self.n_digitized ** self.env.observation_space.shape[0]
        self.n_action = self.env.action_space.n
        self.Q = np.zeros([self.n_state, self.n_action])
        self.V = np.zeros(self.n_state)

        # Bins
        self.bins0 = self.bins(-2.4, 2.4, self.n_digitized)
        self.bins1 = self.bins(-3.0, 3.0, self.n_digitized)
        self.bins2 = self.bins(-0.5, 0.5, self.n_digitized)
        self.bins3 = self.bins(-2.0, 2.0, self.n_digitized)

        # Logging
        self.log_Q = []
        self.log_reward = []

    def bins(self, clip_min, clip_max, num):
        return np.linspace(clip_min, clip_max, num + 1)[1:-1]

    def digitize_state(self, observation):

        cart_pos, cart_v, pole_angle, pole_v = observation

        digitized = [
            np.digitize(cart_pos, bins=self.bins0),
            np.digitize(cart_v, bins=self.bins1),
            np.digitize(pole_angle, bins=self.bins2),
            np.digitize(pole_v, bins=self.bins3)
        ]
        return sum([x * (self.n_digitized ** i) for i, x in enumerate(digitized)])

    def train(self, method="Value", verbose=False, gamma=0.7, alpha=0.2, epsilon=0.1, n_episode=1000):

        # Parameters
        self.gamma = gamma  # discount factor
        self.alpha = alpha  # learning rate
        self.epsilon = epsilon  # epsilon greedy
        self.n_episode = n_episode

        for episode in range(1, self.n_episode + 1):
            if method == "Q-learning":
                self.Q_one_episode()
            elif method == "SARSA":
                self.SARSA_one_episode()
            elif method == "Value":
                self.value_one_episode()
            else:
                raise ValueError("self.method should be 'Q-learning'")

            if (episode % (self.n_episode // 10) == 0) and verbose:
                if method == "Value":
                    print(f"Episode {episode}")
                else:
                    print(
                        f"Episode {episode} Total Reward: {self.log_reward[-1]}")

    def Q_one_episode(self):
        done = False
        total_reward = 0
        state = self.env.reset()
        state = self.digitize_state(state)

        while not done:
            if random.uniform(0, 1) < self.epsilon:
                action = self.env.action_space.sample()  # Explore state space
            else:
                action = np.argmax(self.Q[state])  # Exploit learned values
            next_state, reward, done, info = self.env.step(
                action)  # invoke Gym
            next_state = self.digitize_state(next_state)

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
        current_state = self.digitize_state(current_state)
        if random.uniform(0, 1) < self.epsilon:
            current_action = self.env.action_space.sample()  # Explore state space
        else:
            # Exploit learned values
            current_action = np.argmax(self.Q[current_state])
        while not done:
            next_state, reward, done, info = self.env.step(
                current_action)  # invoke Gym
            next_state = self.digitize_state(next_state)

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

    def value_one_episode(self):

        for s0 in self.bins0:
            for s1 in self.bins1:
                for s2 in self.bins2:
                    for s3 in self.bins3:
                        v_temp = np.zeros(self.n_action)
                        for action in range(self.n_action):
                            self.env.reset()
                            self.env.state = (s0, s1, s2, s3)
                            i_state = self.digitize_state(self.env.state)
                            next_state, reward, done, info = self.env.step(
                                action)
                            next_state = self.digitize_state(next_state)
                            v_temp[action] = reward + \
                                self.gamma * self.V[next_state]
                        action = np.argmax(v_temp)
                        self.V[i_state] = np.max(v_temp)

    def play(self, n_episode=100):

        state = self.env.reset()
        state = self.digitize_state(state)
        reward = 0
        done = False
        while not done:
            self.env.render()
            action = np.argmax(self.Q[state])
            state, reward, done, info = self.env.step(action)
            state = self.digitize_state(state)
        self.env.close()


