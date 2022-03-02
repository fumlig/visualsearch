import gym

from gym_search.envs.search import SearchEnv
from gym_search.terrain import basic_terrain, realistic_terrain


gym.register(
    id="Search-v0",
    entry_point=SearchEnv,
    kwargs=dict(
        world_shape=(1024, 1024),
        view_shape=(64, 64),
        step_size=64,
        terrain_func=realistic_terrain
    )
)

gym.register(
    id="SearchDense-v0",
    entry_point=SearchEnv,
    kwargs=dict(
        world_shape=(64, 64), 
        view_shape=(4, 4), 
        step_size=1,
        terrain_func=lambda shape, random: basic_terrain(shape, random, 40, sigma=5, num_kernels=25, num_targets=10),
    )
)

gym.register(
    id="SearchSparse-v0",
    entry_point=SearchEnv,
    kwargs=dict(
        world_shape=(64, 64), 
        view_shape=(4, 4), 
        step_size=1,
        terrain_func=lambda shape, random: basic_terrain(shape, random, 40, sigma=5, num_kernels=10, num_targets=3),
    )
)
