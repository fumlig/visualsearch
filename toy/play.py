#!/usr/bin/env python3

import gym
from gym.utils.play import play

import envs

play(gym.make("Coverage-v0", width=25, height=10), zoom=32)
