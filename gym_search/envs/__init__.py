import os
import gym
from gym_search.envs.dataset import DatasetEnv


from gym_search.envs.gaussian import GaussianEnv
from gym_search.envs.terrain import TerrainEnv
#from gym_search.envs.camera import CameraEnv
from gym_search.datasets import XViewDataset

gym.register(
    id="Gaussian-v0",
    entry_point=GaussianEnv,
    kwargs=dict()
)

gym.register(
    id="Terrain-v0",
    entry_point=TerrainEnv,
    kwargs=dict()
)

#gym.register(
#    id="Camera-v0",
#    entry_point=CameraEnv
#)


if os.path.exists("data/xview"):
    gym.register(
        id="XView-v0",
        entry_point=DatasetEnv,
        kwargs=dict(
            dataset=XViewDataset("data/xview"),
            shape=(16, 16),
            view=(128, 128),
        )
    )