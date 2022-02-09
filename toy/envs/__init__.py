import gym

from envs import ptz, cov


gym.register(
    id="PanTiltZoom-v0",
    entry_point=ptz.PTZEnv,
    max_episode_steps=2000,
)

gym.register(
    id="Coverage-v0",
    entry_point=cov.CovEnv,
    max_episode_steps=8000,
)
