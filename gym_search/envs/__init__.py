import os
import gym

from gym_search.datasets import AirbusAircraftDataset, AirbusOilDataset
from gym_search.envs.search import SearchEnv
from gym_search.envs.voxel import VoxelEnv
from gym_search.generators import GaussianGenerator, TerrainGenerator, DatasetGenerator

"""
is the agent rewarded for finishing quickly?
- not as of now, it is only punished for visiting a square twice...
"""


gym.register(
    id="Search-v0",
    entry_point=SearchEnv,
    kwargs=dict(
        generator=GaussianGenerator((4, 4), 1, 1, 1, 1),
        view_shape=(1, 1),
        step_size=1
    )
)

gym.register(
    id="Voxel-v0",
    entry_point=VoxelEnv
)

gym.register(
    id="SearchGaussian-v0",
    entry_point=SearchEnv,
    kwargs=dict(
        generator=GaussianGenerator((256, 256), 3, 4, 3, 128, sigma=24),
        view_shape=(16, 16),
        step_size=16,
    )
)

gym.register(
    id="SearchTerrain-v0",
    entry_point=SearchEnv,
    kwargs=dict(
        generator=TerrainGenerator((1024, 1024), 3, max_terrains=1024),
        view_shape=(64, 64),
        step_size=64,
    )
)


if os.path.exists("data/airbus-aircraft"):
    gym.register(
        id="SearchAirbusAircraft-v0",
        entry_point=SearchEnv,
        kwargs=dict(
            generator=DatasetGenerator(AirbusAircraftDataset("data/airbus-aircraft")),
            view_shape=(128, 128),
            step_size=128
        )
    )

if os.path.exists("data/airbus-oil"):
    gym.register(
        id="SearchAirbusOil-v0",
        entry_point=SearchEnv,
        kwargs=dict(
            generator=DatasetGenerator(AirbusOilDataset("data/airbus-oil")),
            view_shape=(128, 128),
            step_size=128
        )
    )