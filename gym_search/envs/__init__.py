import gym

from gym_search.datasets import AirbusAircraftDataset, AirbusOilDataset
from gym_search.envs.search import SearchEnv
from gym_search.envs.generators import GaussianGenerator, TerrainGenerator, DatasetGenerator

"""
is the agent rewarded for finishing quickly?
- not as of now, it is only punished for visiting a square twice...
"""


gym.register(
    id="Search-v0",
    entry_point=SearchEnv,
    kwargs=dict(
        generator=TerrainGenerator((1024, 1024), max_terrains=1024),
        view_shape=(64, 64),
        step_size=64,
    )
)

gym.register(
    id="SearchDense-v0",
    entry_point=SearchEnv,
    kwargs=dict(
        generator=GaussianGenerator((32, 32), num_targets=8, num_kernels=8, size=32, sigma=4),
        view_shape=(8, 8), 
        step_size=1,
    )
)

gym.register(
    id="SearchDense-v1",
    entry_point=SearchEnv,
    kwargs=dict(
        generator=GaussianGenerator((32, 32), num_targets=8, num_kernels=8, size=32, sigma=4),
        view_shape=(4, 4), 
        step_size=4,
        max_steps=250,
    )
)

gym.register(
    id="SearchSparse-v0",
    entry_point=SearchEnv,
    kwargs=dict(
        generator=GaussianGenerator((32, 32), num_targets=1, num_kernels=1, size=128, sigma=16),
        view_shape=(8, 8), 
        step_size=1,
    )
)

"""
gym.register(
    id="SearchAirbusAircraft-v0",
    entry_point=SearchEnv,
    kwargs=dict(
        generator=DatasetGenerator(AirbusAircraftDataset()),
        view_shape=(128, 128),
        step_size=128
    )
)

gym.register(
    id="SearchAirbusOil-v0",
    entry_point=SearchEnv,
    kwargs=dict(
        generator=DatasetGenerator(AirbusOilDataset()),
        view_shape=(128, 128),
        step_size=128
    )
)
"""