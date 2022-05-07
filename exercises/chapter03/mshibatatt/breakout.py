import argparse
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import DQN

def main(lr, ex, rb, fr, t, seed, hide_result, name, save, model):
    env = make_atari_env('BreakoutNoFrameskip-v4')
    if model is None:
        model = DQN(
            'CnnPolicy',
            env,
            tensorboard_log='./tensorboard',
            learning_rate = lr,
            exploration_fraction=ex,
            buffer_size=rb,
            train_freq=fr,
            seed=seed)
        model.learn(total_timesteps = t,  tb_log_name=name)
        if save:
            model.save("./" + name)
    else:
        model = DQN.load(model, env=env)

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f'mean reward: {mean_reward}, s.d. of reward: {std_reward}')
    obs = env.reset()

    if not hide_result:
        while True:
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            env.render(mode = 'human')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "-lr", "--lr",
        type=float,
        default=0.0001,
        help="learning rate",
    )
    parser.add_argument(
        "-ex", "--ex",
        type=float,
        default=0.1,
        help="exploration rate",
    )
    parser.add_argument(
        "-rb", "--rb",
        type=int,
        default=1000000,
        help="replay buffer size",
    )
    parser.add_argument(
        "-fr", "--fr",
        type=int,
        default=4,
        help="how frequently the target network updated",
    )
    parser.add_argument(
        "-t", "--t",
        type=int,
        default=25000,
        help="total timesteps",
    )
    parser.add_argument(
        "-seed", "--seed",
        type=int,
        default=None,
        help="seed number default does not set",
    )
    parser.add_argument(
        "-hide", "--hide",
        action='store_true',
        help="show train result as rendered form",
    )
    parser.add_argument(
        "-name", "--name",
        type=str,
        default=None,
        help="model name for tensorboard",
    )
    parser.add_argument(
        "-save", "--save",
        action='store_true',
        help="flag if save trained model",
    )
    parser.add_argument(
        "-model", "--model",
        type=str,
        default=None,
        help="model object path to load",
    )
    args = parser.parse_args()
    if args.name is None:
        args.name = 'DQN'
    
    main(args.lr, args.ex, args.rb, args.fr, args.t,
         args.seed, args.hide, args.name, args.save, args.model)