from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.dqn import CnnPolicy
from stable_baselines3.common.evaluation import evaluate_policy

"""
Default Parameters

- alpha(learning rate): 0.0001
- buffer size: 1,000,000
- epsilon(exploration rate): 0.1
- training frequency: Update the model every 4 steps
- network architecture: CNN
"""

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
  main()