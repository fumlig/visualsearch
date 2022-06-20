import gym

from gym_search.envs.search import Action
from gym_search.envs.gaussian import GaussianEnv
from gym_search.envs.terrain import TerrainEnv
from gym_search.envs.camera import CameraEnv


gym.register(
    id="Gaussian-v0",
    entry_point=GaussianEnv,
)

gym.register(
    id="Terrain-v0",
    entry_point=TerrainEnv,
)

gym.register(
    id="Camera-v0",
    entry_point=CameraEnv,
)
