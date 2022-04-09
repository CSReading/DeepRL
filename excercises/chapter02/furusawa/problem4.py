import gym
import numpy as np

"""
Problem 4 : Cartpole

Run Cartpole with the greedy policy computed by value iteration.
"""

# メインの学習を行う
def main(
  num_rounds=10, gamma=0.9, threshhold=0.1,
  num_pos=4,
  min_vel=-2.0, max_vel=2.0, num_vel=4,
  num_angle=4,
  min_angular_vel=-2.0, max_angular_vel=2.0, num_angular_vel=4
  ):

  # Gym環境のセットアップ
  env = gym.make('CartPole-v1')
  env.reset()

  cum_reward = 0

  for t_round in range(num_rounds):
    observation = env.reset()
    
    # stateは4つ
    # しかしそれぞれ連続区間になっている為、離散化する必要がある
    position_space = np.linspace(-4.8, 4.8, num_pos)
    velocity_space = np.linspace(min_vel, max_vel, num_vel) # 本来は (-inf, inf) の範囲だが、プログラムでは上限下限を設定する
    angle_space = np.linspace(-0.2, 0.2, num_angle)
    angular_velocity_space = np.linspace(min_angular_vel, max_angular_vel, num_angular_vel)

    V = np.zeros(shape=(num_pos, num_vel, num_angle, num_angular_vel))

    # ベルマン方程式を解く
    while True:
      prev_V = V.copy()
      V = iterate_value_function(V, env, gamma)
      if (np.max(V - prev_V) < threshhold):
        break
    
    # 算出されたVからpolicyをつくる
    policy = build_greedy_policy()

    # 計算されたpolicyを使ってepisilon-greedy methodを用いる
    for t in range(1000):
      action = policy[observation]
      observation, reward, done, _ = env.step(action)
      cum_reward += reward
      if done:
        break

# value iterationのコアとなる関数
# これをiterateしていきVを計算する
def iterate_value_function(V, env, gamma):
  inner_V = np.zeros(shape=V.shape)

  # V(state)を更新していく
  for pos in range(V.shape[0]):
    for vel in range(V.shape[1]):
      for ang in range(V.shape[2]):
        for ang_vel in range(V.shape[3]):
          temp_v = np.zeros(env.action_space.n)
          for action in range(env.action_space.n):
            env.reset()
            env.state = (pos, vel, ang, ang_vel)
            next_state, reward, done, _ = env.step(action)
            temp_v[action] = reward + gamma * inner_V[next_state]
            
          # value iterationではmaxのものを取る
          inner_V[pos, val, ang, ang_vel] = max(temp_v)
  
  return inner_V

# Vを所与としてgreedyなpolicyを計算する
def build_greedy_policy(V, env, gamma):
  new_policy = np.zeros(shape=V.shape)

  for pos in range(V.shape[0]):
    for vel in range(V.shape[1]):
      for ang in range(V.shape[2]):
        for ang_vel in range(V.shape[3]):
          profits = np.zeros(env.action_space.n)
          for action in range(env.action_space.n):
            env.reset()
            env.state = (pos, vel, ang, ang_vel)
            next_state, reward, done, _ = env.step(action)
            profits[action] = reward + gamma * V[next_state]
            
          # greedyなのでmaxのものを取る
          new_policy[pos, val, ang, ang_vel] = np.argmax(profits)

  return new_policy

if __name__ == "__main__":
  main()
