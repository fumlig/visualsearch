import gym

from gym_search.envs.search import SearchEnv
from gym_search.terrain import gaussian_terrain


gym.register(
    id="Search-v0",
    entry_point=SearchEnv,
    max_episode_steps=1000,
    kwargs={}
)

gym.register(
    id="SearchBumpy-v0",
    entry_point=SearchEnv,
    max_episode_steps=1000,
    kwargs=dict(
        num_targets=25,
        terrain_func=lambda s, r: gaussian_terrain(s, r, 40, sigma=5, n=25),
    )
)
