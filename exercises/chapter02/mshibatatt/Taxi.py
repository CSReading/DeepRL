# listing_2_5: Q learning for Taxi example
import gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
import random

class Taxi:
    def __init__(self, alpha = 0.2, gamma = 0.7, epsilon = 0.1,
                 method = "Q-learning", quiet = False, plot_Q = False) -> None:
        # Environment Setup
        self.env = gym.make("Taxi-v3")
        self.env.reset()
        self.env.reset()
        # set parameters
        self.gamma = gamma# discount factor
        self.alpha = alpha # learning rate
        self.epsilon = epsilon # epsilon greedy
        # "Q" or "SARSA"
        self.method = method 
        # print setting
        self.quiet = quiet
        self.plot_Q = plot_Q

    def reset_learning(self):
        # Q[state, action] table implementation
        self.Q = np.zeros([self.env.observation_space.n, self.env.action_space.n])
        self.learning_log = np.zeros(0)

    def __str__(self):
        return self.env.render(mode="ansi")

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
        while not done:
            if random.uniform(0, 1) < self.epsilon:
                action = self.env.action_space.sample() # Explore state space
            else:
                action = np.argmax(self.Q[state]) # Exploit learned values
            next_state, reward, done, info = self.env.step(action) # invoke Gym

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
        if random.uniform(0, 1) < self.epsilon:
            current_action = self.env.action_space.sample() # Explore state space
        else:
            current_action = np.argmax(self.Q[current_state]) # Exploit leaned value

        while not done:
            next_state, reward, done, info = self.env.step(current_action) # envoke gym
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
        total_epochs, total_penalties, total_reward = 0, 0, 0
        for _ in range(ep):
            state = self.env.reset()
            epochs, penalties, reward = 0, 0, 0
            done = False
            while not done:
                action = np.argmax(self.Q[state])
                state, reward, done, info = self.env.step(action)
                if reward == -10:
                    penalties += 1
                epochs += 1
            total_penalties += penalties
            total_epochs += epochs
            total_reward += reward

        print(f"Results after {ep} episodes:")
        print(f"Average reward per episode {total_reward / ep}")
        print(f"Average timesteps per episode {total_epochs / ep}")
        print(f"Average penalites per episode {total_penalties / ep}")

    def plot_policy(self):
        fig, ax = plt.subplots(5, 4)
        cbar_ax = fig.add_axes([.91, .3, .03, .4])
        plot_data = np.zeros((5, 4, 5, 5))
        for s in range(500):
            taxi_row, taxi_col, pass_loc, dest_idx = self.env.decode(s)
            plot_data[pass_loc][dest_idx][taxi_row][taxi_col] = np.argmax(self.Q[s])
        for i in range(5):
            for j in range(4):
                sns.heatmap(plot_data[i][j], ax= ax[i][j], vmin = 0, vmax = 5,
                            cbar = (i+j==0), cbar_ax = cbar_ax if (i+j==0) else None, animated = True)
                ax[i][j].axis("off")
        return fig, ax
