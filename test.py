import torch as th

import gym
import gym_search

env = gym.make("SearchSparse-v0")
env = gym.wrappers.FlattenObservation(env)


print(env.observation_space)