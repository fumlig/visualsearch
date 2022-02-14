import gym
from importlib_metadata import entry_points

from envs import ptz, cov, toy


gym.register(
    id="PanTiltZoom-v0",
    entry_point=ptz.PTZEnv,
    max_episode_steps=2000,
)

gym.register(
    id="Coverage-v0",
    entry_point=cov.CovEnv,
    max_episode_steps=5000,
)

gym.register(
    id="Toy-v0",
    entry_point=toy.ToyEnv,
)