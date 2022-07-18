# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
# solve same task as medium.py
from multiprocessing import pool
import numpy as np
import argparse
import math
import random
from collections import namedtuple, deque
import torch
from torch import nn, optim
import gym
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import DQN

def base_line(env, lr, ep):
    model = DQN(
        'MlpPolicy',
        env,
        learning_rate = lr)
    model.learn(total_timesteps = ep)
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f'mean reward: {mean_reward}, s.d. of reward: {std_reward}')
    return model
    
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    
    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.state_space = env.observation_space.shape[0]
        self.action_space = env.action_space.n
        self.l1 = nn.Linear(self.state_space, 128, bias=False)
        self.l2 = nn.Linear(128, self.action_space, bias=False)
        
        self.gamma = gamma
        # Overall reward and loss history
        self.reward_history = []
        self.loss_history = []

    def forward(self, x):  
        model = torch.nn.Sequential(
            self.l1,
            nn.Dropout(p=0.6),
            nn.ReLU(),
            self.l2
        )
        return model(x)


# Select action
def select_action(state):
    #Select an action (0 or 1) by running policy model
    state = torch.from_numpy(state).type(torch.FloatTensor)

    # skip tempreture parameter
    if random.random() > EPSILON:
        with torch.no_grad():
            # take argmax of policy_net output
            action = policy_net(state).argmax() 
            action = action.item()
    else:
        action = random.randrange(policy_net.action_space)
    
    return action

# update network
def update():
    if len(memory) < BATCH_SIZE:
        return
    
    transitions = memory.sample(BATCH_SIZE)
    # Transpose batch
    batch = Transition(*zip(*transitions))

    # Conpute a mask of non-final states
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    # concatenate batch elements
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    
    # Compute Q(s_t, a) by estimating and picking taken action
    Q_estimated = policy_net(state_batch)
    state_action_values = Q_estimated.gather(1, torch.cat((action_batch, action_batch)).reshape((-1, 2)))[:, 0]

    # Compute V(s_t+1) for all next states
    next_state_values = torch.zeros(BATCH_SIZE)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    # Calculate expected Q values
    expected_state_action_values = (next_state_values * policy_net.gamma) + reward_batch

    # Calculate Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values)
    
    # Update network weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    

def main(episodes):
    running_reward = float('inf')

    for episode in range(episodes):
        state = env.reset() # Reset environment and record the starting state
        done = False   
        reward_episode = []
        for time in range(1000):
            action = select_action(state)
            next_state, reward, done, _ = env.step(action) # Save reward
            reward_episode.append(reward)
            memory.push(
                torch.tensor(state.reshape((1, -1))),
                torch.tensor([action]),
                torch.tensor(next_state.reshape((1, -1))),
                torch.tensor([reward])
            )
            state = next_state
            update()
            if done:
                break

        # Used to determine when the environment is solved.
        episode_reward = sum(reward_episode)
        running_reward = running_reward * 0.99 + episode_reward * 0.01 if running_reward < float('inf') else episode_reward 
        
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if episode % 100 == 0:
            print('Episode {}\tLast length: {:5d}\tRunning reward: {:.2f}'.format(episode, time, running_reward))
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and the last episode runs to {} time steps!".format(running_reward, time))
            break

def show(base_line_flag):
    state = env.reset() # Reset environment and record the starting state
    done = False
    if base_line_flag:
        for time in range(1000):
            env.render()
            action, _ = model.predict(state)
            state, reward, done, _ = env.step(action) 
            if done:
                break
    else:       
        for time in range(1000):
            env.render()
            action = select_action(state)
            state, reward, done, _ = env.step(action) 
            if done:
                break
    env.close()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t", "--title",
        help = "gym game name",
        default = "CartPole-v1", 
        choices=["CartPole-v1", "MountainCar-v0"]
    )
    parser.add_argument(
        "-e", "--episode",
        type=int,
        default=10000,
        help="episode",
    )
    parser.add_argument(
        "-lr", "--learning_rate",
        type=float,
        default=0.01,
        help="learning rate",
    )
    parser.add_argument(
        "-base", "--baseline",
        action='store_true',
        help="if use stable_baseline or not",
    )

    args = parser.parse_args()
    title = args.title
    # title = "MountainCar-v0"
    env = gym.make(title)

    #Hyperparameters
    learning_rate = args.learning_rate
    gamma = 0.99
    episodes = args.episode
    EPSILON = 0.05
    BATCH_SIZE = 128
    TARGET_UPDATE = 10

    if args.baseline:
        model = base_line(env, learning_rate, episodes)
    else:
        policy_net = DQN()
        target_net = DQN()
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()
        optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
        memory = ReplayMemory(10_000)
        # Running the Model
        main(episodes)
    
    show(args.baseline)