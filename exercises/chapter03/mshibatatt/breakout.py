from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3 import DQN
env = make_atari_env('BreakoutNoFrameskip-v4')
model = DQN('CnnPolicy', env, verbose =1)
model.learn(total_timesteps = 25000)
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render(mode = 'human')