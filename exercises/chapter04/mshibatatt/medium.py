# https://tims457.medium.com/policy-gradient-reinforcement-learning-in-pytorch-df1383ea0baf
import numpy as np
import argparse
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.distributions import Categorical
import gym

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.state_space = env.observation_space.shape[0]
        if type(env.action_space) == gym.spaces.discrete.Discrete:
            self.action_space = env.action_space.n
        else:
            self.action_space = env.action_space.shape[0]
        
        self.l1 = nn.Linear(self.state_space, 128, bias=False)
        self.l2 = nn.Linear(128, self.action_space, bias=False)
        
        self.gamma = gamma
        
        # Episode policy and reward history 
        self.policy_history = Variable(torch.Tensor()) 
        self.reward_episode = []
        # Overall reward and loss history
        self.reward_history = []
        self.loss_history = []

    def forward(self, x):  
        if type(env.action_space) == gym.spaces.discrete.Discrete:
            model = torch.nn.Sequential(
                self.l1,
                nn.Dropout(p=0.6),
                nn.ReLU(),
                self.l2,
                nn.Softmax(dim=-1)
            )
        else:
            model = torch.nn.Sequential(
                self.l1,
                nn.Dropout(p=0.6),
                nn.ReLU(),
                self.l2,
                nn.Tanh()
            )
        return model(x)


# Select action
def select_action(state):
    #Select an action (0 or 1) by running policy model and choosing based on the probabilities in state
    state = torch.from_numpy(state).type(torch.FloatTensor)
    state = policy(Variable(state))
    if type(env.action_space) == gym.spaces.discrete.Discrete:
        c = Categorical(state)
        action = c.sample()
        # Add log probability of our chosen action to our history    
        if policy.policy_history.dim() != 0:
            policy.policy_history = torch.cat([policy.policy_history, c.log_prob(action).reshape(1)])
        else:
            policy.policy_history = (c.log_prob(action)).reshape(1)
        action = action.item()
    else: 
        action = state
        if policy.policy_history.dim() != 0:
            policy.policy_history = torch.cat([policy.policy_history, action.reshape(1)])
        else:
            policy.policy_history = action.reshape(1)
        action = action.detach().numpy()

    return action

# update policy
def update_policy():
    R = 0
    rewards = []
    
    # Discount future rewards back to the present using gamma
    for r in policy.reward_episode[::-1]:
        R = r + policy.gamma * R
        rewards.insert(0,R)
        
    # Scale rewards
    rewards = torch.FloatTensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
    
    # Calculate loss
    loss = (torch.sum(torch.mul(policy.policy_history, Variable(rewards)).mul(-1), -1))
    
    # Update network weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    #Save and intialize episode history counters
    policy.loss_history.append(loss.item())
    policy.reward_history.append(np.sum(policy.reward_episode))
    policy.policy_history = Variable(torch.Tensor())
    policy.reward_episode= []

def main(episodes):
    running_reward = float('inf')
    for episode in range(episodes):
        state = env.reset() # Reset environment and record the starting state
        done = False   
        # temp_reward = 0
        for time in range(1000):
            action = select_action(state)
            # Step through environment using chosen action
            state, reward, done, _ = env.step(action) # Save reward
            # reward of mountain car is too strict
            # if title.startswith("MountainCar"):
            #     temp_reward = max(temp_reward, state[0]+0.5)
            #     reward += temp_reward
            policy.reward_episode.append(reward)
            if done:
                break

        # Used to determine when the environment is solved.
        episode_reward = sum(policy.reward_episode)
        running_reward = running_reward * 0.99 + episode_reward * 0.01 if running_reward < float('inf') else episode_reward 
        
        update_policy()

        if episode % 100 == 0:
            print('Episode {}\tLast length: {:5d}\tRunning reward: {:.2f}'.format(episode, time, running_reward))
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and the last episode runs to {} time steps!".format(running_reward, time))
            break

def show():
    state = env.reset() # Reset environment and record the starting state
    done = False       
    for time in range(1000):
        env.render()
        action = select_action(state)
        state, reward, done, _ = env.step(action) 
        policy.reward_episode.append(reward)
        if done:
            break
    env.close()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t", "--title",
        help = "gym game name",
        default = "MountainCarContinuous-v0",# "CartPole-v1", 
        choices=["CartPole-v1", "MountainCar-v0", "MountainCarContinuous-v0"]
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

    args = parser.parse_args()
    title = args.title
    # title = "MountainCar-v0"
    env = gym.make(title)

    #Hyperparameters
    learning_rate = args.learning_rate
    gamma = 0.99

    policy = Policy()
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    # Running the Model
    episodes = args.episode
    main(episodes)
    show()