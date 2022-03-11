import gym
import gym_search
from gym_search.wrappers import ObserveVisible, ObserveVisited, ResizeImage
from stable_baselines3 import PPO
from stable_baselines3.ppo import MultiInputPolicy
from stable_baselines3.common.env_util import make_vec_env

env = make_vec_env("SearchGaussian-v0", n_envs=8, wrapper_class=ResizeImage)

model = PPO(MultiInputPolicy, env, verbose=2, tensorboard_log="logs", ent_coef=0.01)

model.learn(total_timesteps=10000000, tb_log_name="sb3")