import gym

from gym_ptz.envs.ptz import PTZEnv

gym.register(
    id="PanTiltZoom-v0",
    entry_point=PTZEnv,
    max_episode_steps=1000,
    kwargs={}
)