#!/usr/bin/env python3

import gym
from gym.utils.play import play

import envs

env = gym.make("Toy-v0")

play(env, zoom=8)
