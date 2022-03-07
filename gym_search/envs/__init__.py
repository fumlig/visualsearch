import gym

from gym_search.envs.search import SearchEnv
from gym_search.terrain import gaussian_terrain, realistic_terrain


gym.register(
    id="Search-v0",
    entry_point=SearchEnv,
    kwargs=dict(
        world_shape=(1024, 1024),
        view_shape=(64, 64),
        step_size=64,
        terrain_func=realistic_terrain,
        max_levels=1024
    )
)

gym.register(
    id="SearchDense-v0",
    entry_point=SearchEnv,
    kwargs=dict(
        world_shape=(32, 32), 
        view_shape=(8, 8), 
        step_size=1,
        terrain_func=lambda shape, random: gaussian_terrain(shape, random, 32, sigma=4, num_kernels=8, num_targets=8),
    )
)

gym.register(
    id="SearchSparse-v0",
    entry_point=SearchEnv,
    kwargs=dict(
        world_shape=(32, 32), 
        view_shape=(8, 8), 
        step_size=1,
        terrain_func=lambda shape, random: gaussian_terrain(shape, random, 128, sigma=16, num_kernels=1, num_targets=1),
    )
)
