import gym

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
env = gym.make('CartPole-v1')
model = PPO('MlpPolicy', env, verbose =1)
model.learn(total_timesteps =10000)
obs = env.reset()
for i in range (1000) :
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()