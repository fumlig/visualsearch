import gym

from gym_search.envs.search import SearchEnv, PrettySearchEnv
from gym_search.terrain import gaussian_terrain, uniform_terrain


gym.register(
    id="Search-v0",
    entry_point=PrettySearchEnv,
)

gym.register(
    id="SearchUniform-v0",
    entry_point=SearchEnv,
    kwargs=dict(
        terrain_func=uniform_terrain,
    )
)

gym.register(
    id="SearchDense-v0",
    entry_point=SearchEnv,
    kwargs=dict(
        world_shape=(40, 40), 
        view_shape=(5, 5), 
        num_targets=25,
        terrain_func=lambda s, r: gaussian_terrain(s, r, 40, sigma=5, n=25),
    )
)

gym.register(
    id="SearchSparse-v0",
    entry_point=SearchEnv,
    kwargs=dict(
        world_shape=(40, 40), 
        view_shape=(5, 5), 
        num_targets=3,
        terrain_func=lambda s, r: gaussian_terrain(s, r, 40, sigma=5, n=10),
    )
)
