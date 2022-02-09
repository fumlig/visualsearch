#!/usr/bin/env python3

import gym
from gym.utils.play import play

import envs

env = gym.make("Coverage-v0", width=100, height=100, radius=10)

play(env, zoom=8)
