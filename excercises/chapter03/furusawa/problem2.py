from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.dqn import CnnPolicy
from stable_baselines3.common.evaluation import evaluate_policy

# same as problem1
def main(alpha=0.0001, buffer_size=1000000, epsilon=0.1, train_freq=4, time_steps=25000, isRender=False):
  env = make_atari_env('BreakoutNoFrameskip-v4')

  # モデルの作成
  model = DQN(
    CnnPolicy,
    env,
    learning_rate=alpha,
    buffer_size=buffer_size,
    exploration_fraction=epsilon,
    train_freq=train_freq,
    verbose=0)

  # モデルの学習
  model.learn(total_timesteps=time_steps)

  # モデルの評価
  mean_reward, std_reward = evaluate_policy(model, env)
  print(f'mean reward: {mean_reward}, s.d. of reward: {std_reward}')

  if isRender:
    obs = env.reset()
    while True:
      action, _states = model.predict(obs)
      oobs, rewards, dones, info = env.step(action)
      env.render()

if __name__ == "__main__":
  print("Benchmark")
  main()
  print("Change Learning Rate")
  main(alpha=0.01)
  main(alpha=0.00001)
  print("Change Exploration Rate")
  main(epsilon=0.5)
  main(epsilon=0.01)
  print("Change Buffer Size")
  main(buffer_size=10000000)
  main(buffer_size=10000)

"""
Benchmark
mean reward: 1.9, s.d. of reward: 0.3

Change Learning Rate
mean reward: 2.3, s.d. of reward: 2.368543856465402
mean reward: 1.5, s.d. of reward: 2.29128784747792

Change Exploration Rate
mean reward: 2.2, s.d. of reward: 0.6
mean reward: 1.9, s.d. of reward: 3.562302626111375

Change Buffer Size
mean reward: 0.1, s.d. of reward: 0.30000000000000004
mean reward: 1.2, s.d. of reward: 1.9899748742132397
"""