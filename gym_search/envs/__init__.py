import gym

from gym_search.envs.search import SearchEnv
from gym_search.terrain import gaussian_terrain, uniform_terrain


gym.register(
    id="Search-v0",
    entry_point=SearchEnv,
    max_episode_steps=1000,
)

gym.register(
    id="SearchUniform-v0",
    entry_point=SearchEnv,
    max_episode_steps=1000,
    kwargs=dict(
        terrain_func=uniform_terrain,
    )
)

gym.register(
    id="SearchDense-v0",
    entry_point=SearchEnv,
    max_episode_steps=1000,
    kwargs=dict(
        num_targets=25,
        terrain_func=lambda s, r: gaussian_terrain(s, r, 40, sigma=5, n=25),
    )
)

gym.register(
    id="SearchSparse-v0",
    entry_point=SearchEnv,
    max_episode_steps=1000,
    kwargs=dict(
        num_targets=3,
        terrain_func=lambda s, r: gaussian_terrain(s, r, 40, sigma=5, n=10),
    )
)