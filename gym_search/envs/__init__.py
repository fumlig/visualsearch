import os
import gym

from gym_search.envs.search import SearchEnv
from gym_search.envs.camera import CameraEnv
from gym_search.generators import GaussianGenerator, TerrainGenerator, DatasetGenerator
from gym_search.datasets import XViewDataset

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
    id="Camera-v0",
    entry_point=CameraEnv
)

gym.register(
    id="Gaussian-v0",
    entry_point=SearchEnv,
    kwargs=dict(
        generator=GaussianGenerator((256, 256), 3, 4, 3, 128, sigma=24),
        view_shape=(16, 16),
        step_size=16,
    )
)

gym.register(
    id="Gaussian-v1",
    entry_point=SearchEnv,
    kwargs=dict(
        generator=GaussianGenerator((1024, 1024), 3, 8, 3, 512, sigma=96),
        view_shape=(64, 64),
        step_size=64,
    )
)

gym.register(
    id="Terrain-v0",
    entry_point=SearchEnv,
    kwargs=dict(
        generator=TerrainGenerator((1024, 1024), 3, 100, max_terrains=1024),
        view_shape=(64, 64),
        step_size=64,
    )
)

gym.register(
    id="Terrain-v1",
    entry_point=SearchEnv,
    kwargs=dict(
        generator=TerrainGenerator((512, 512), 3, 100, max_terrains=1024),
        view_shape=(64, 64),
        step_size=64,
    )
)

gym.register(
    id="Terrain-v2",
    entry_point=SearchEnv,
    kwargs=dict(
        generator=TerrainGenerator((1024, 1024), 3, 100, max_terrains=1024),
        view_shape=(64, 64),
        step_size=64,
        max_steps=2500
    )
)


#if os.path.exists("data/xview"):
#    gym.register(
#        id="XView-v0",
#        entry_point=SearchEnv,
#        kwargs=dict(
#            generator=DatasetGenerator(XViewDataset("data/xview")),
#            view_shape=(128, 128),
#            step_size=128
#        )
#    )