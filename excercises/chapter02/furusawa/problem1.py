import gym
import numpy as np

"""
Problem 1: Q-Learning

Implement Q-learning for Taxi, including the procedure to derive the best policy for the Q-table.
"""

class Taxi:
  def __init__(self):
    # Gym環境のセットアップ
    self.env = gym.make("Taxi-v3")
    self.env.reset()

    # Q[state, action]の実装
    self.Q = np.zeros([self.env.observation_space.n, self.env.action_space.n])
  
  def train(self, method = 'Q-learning', gamma = 0.7, alpha = 0.2, epsilon = 0.1, num_episodes = 1000):

    # パラメータの設定
    self.gamma = gamma # 割引率
    self.alpha = alpha # 学習率
    self.epsilon = epsilon # 探索割合

    # エピソードの数
    self.num_episodes = num_episodes

    if (method == 'Q-learning'):
      self.Q_learning()
    elif (method == 'SARSA'):
      self.SARSA()
    else:
      raise ValueError("The method must be either `Q-learning` or `SARSA`")
  
  def Q_learning(self):
    # loopで学習
    for episode in range(self.num_episodes):
      done = False
      total_reward = 0
      state = self.env.reset()

      while not done:
        # 行動の選択
        if (np.random.rand() < self.epsilon):
          # epsilonの確率で、ランダムで行動を選ぶ
          action = self.env.action_space.sample()
        else:
          # それ以外はQ値を最大にするような行動を選ぶ
          action = np.argmax(self.Q[state])
        
        # 行動の結果
        # next_state : 遷移先のstate
        # reward : 行動の結果得られた報酬
        # done : ゲームが終わりならdone=Trueになる
        next_state, reward, done, _ = self.env.step(action)

        # Q値の更新を行う
        # 予想していたQ値と実際のQ値からQ値の学習を行う
        next_max = np.max(self.Q[next_state])
        old_value = self.Q[state, action] # 予想していたvalue
        next_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value) # 更新されたvalue
        self.Q[state, action] = next_value

        total_reward += reward
        state = next_state
      
      if (episode % 100 == 0):
        print(f"Episode {episode} Total Reward : {total_reward}")
  
  def SARSA(self):
    for episode in range(self.num_episodes):
      done = False
      total_reward = 0

      # 一番最初のstateとactionを計算
      current_state = self.env.reset()
      if np.random.rand() < self.epsilon:
        current_action = self.env.action_space.sample()
      else:
        current_action = np.argmax(self.Q[current_state])

      while not done:
        # 次のstateを計算
        next_state, reward, done, _ = self.env.step(current_action)

        # 行動の選択
        # SARSAではpolicyがactionを決めるので、遷移先でどのようなactionを取るのかを計算する必要がある
        # そのため、SARSAの実装ではnext_stateの実行後にnext_actionを計算している
        if (np.random.rand() < self.epsilon):
          next_action = self.env.action_space.sample()
        else:
          next_action = np.argmax(self.Q[next_state])
        
        old_value = self.Q[current_state, current_action] # 予想していたvalue
        next_value = old_value + self.alpha * (reward + self.gamma * self.Q[next_state, next_action] - old_value) # 更新されたvalue
        self.Q[current_state, current_action] = next_value

        total_reward += reward

        current_state = next_state
        current_action = next_action
      
      if (episode % 100 == 0):
        print(f"Episode {episode} Total Reward : {total_reward}")


if __name__ == "__main__":
  taxi = Taxi()
  taxi.train()
