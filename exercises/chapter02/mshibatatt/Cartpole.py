# listing_2_5: Q learning for Taxi example
import gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
import random
import sys

class CartPole:
    def __init__(self, alpha = 0.2, gamma = 0.7, epsilon = 0.1,
                method = "Q-learning", state_split_num = 4,
                quiet = False, plot_Q = False) -> None:
        # Environment Setup
        self.env = gym.make("CartPole-v1")
        self.env.reset()
        # set parameters
        self.gamma = gamma# discount factor
        self.alpha = alpha # learning rate
        self.epsilon = epsilon # epsilon greedy
        # "Q" or "SARSA"
        self.method = method 
        self.state_split_num = state_split_num
        # print setting
        self.quiet = quiet
        self.plot_Q = plot_Q

    def reset_learning(self):
        # Q[state, action] table implementation
        state_n = self.state_split_num ** self.env.observation_space.shape[0]
        self.Q = np.zeros([state_n, self.env.action_space.n])
        self.learning_log = np.zeros(0)

    def discretize_state(self, state):
        output = 0
        for i in range(self.env.observation_space.shape[0]):
            output *= self.state_split_num
            low = self.env.observation_space.low[i]
            high = self.env.observation_space.high[i]
            # to avoid overflow
            state_idx = state[i] // (high / self.state_split_num - low / self.state_split_num)  - low // (high / self.state_split_num - low / self.state_split_num)  
            if state_idx == self.state_split_num:
                state_idx -= 1
            output += int(state_idx)

        return output

    def train(self, N = 1000, keep_table = False):
        if not keep_table:
            self.reset_learning()
        
        # learning
        for episode in range(N):
            if self.method == "Q-learning":
                self.Q_one_episode()
            elif self.method == "SARSA":
                self.SARSA_one_episode()
            else:
                raise ValueError("self.method should be 'Q-learning' or 'SARSA'")

            if episode * 10 % N == 0:
                if not self.quiet:
                    print("Episode {} Total Reward:{}". format(episode, self.learning_log[-1]))
                if self.plot_Q:
                    fig, ax = self.plot_policy()
                    plt.show()

    def Q_one_episode(self):
        done = False
        total_reward = 0
        state = self.env.reset()
        state = self.discretize_state(state)
        while not done:
            if random.uniform(0, 1) < self.epsilon:
                action = self.env.action_space.sample() # Explore state space
            else:
                action = np.argmax(self.Q[state]) # Exploit learned values
            next_state, reward, done, info = self.env.step(action) # invoke Gym
            next_state = self.discretize_state(next_state)

            next_max = np.max(self.Q[next_state])
            old_value = self.Q[state , action]
            new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
            self.Q[state, action] = new_value
            total_reward += reward
            state = next_state
        self.learning_log = np.append(self.learning_log, total_reward)

    def SARSA_one_episode(self):
        done = False
        total_reward = 0
        current_state = self.env.reset()
        current_state = self.discretize_state(current_state)
        if random.uniform(0, 1) < self.epsilon:
            current_action = self.env.action_space.sample() # Explore state space
        else:
            current_action = np.argmax(self.Q[current_state]) # Exploit leaned value

        while not done:
            next_state, reward, done, info = self.env.step(current_action) # envoke gym
            next_state = self.discretize_state(next_state)

            if random.uniform(0, 1) < self.epsilon:
                next_action = self.env.action_space.sample()
            else:
                next_action = np.argmax(self.Q[next_state])
            
            sarsa_value = self.Q[next_state, next_action]
            old_value = self.Q[current_state, current_action]

            new_value = old_value + self.alpha * (reward + self.gamma * sarsa_value - old_value)
            self.Q[current_state, current_action] = new_value
            total_reward += reward
            current_state = next_state
            current_action = next_action
        self.learning_log = np.append(self.learning_log, total_reward)

    def play(self, ep = 100):
        # listing_2_6
        total_reward = 0
        for _ in range(ep):
            state = self.env.reset()
            state = self.discretize_state(state)
            done = False
            while not done:
                action = np.argmax(self.Q[state])
                state, reward, done, info = self.env.step(action)
                state = self.discretize_state(state)
                total_reward += reward

        print(f"Results after {ep} episodes:")
        print(f"Average reward per episode {total_reward / ep}")
    
    def show(self):
        state = self.env.reset()
        state = self.discretize_state(state)
        reward = 0
        done = False
        while not done:
            self.env.render()
            action = np.argmax(self.Q[state])
            state, reward, done, info = self.env.step(action)
            state = self.discretize_state(state)
        self.env.close()

