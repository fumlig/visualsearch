#!/usr/bin/env python3

import gym
from gym.utils.play import play

import envs

play(gym.make("Coverage-v0", width=100, height=100), zoom=8)
