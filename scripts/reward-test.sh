#!/usr/bin/env bash

./run.sh map/reward/terrain/both Terrain-v0 "{reward_explore: true, reward_closer: true}" ppo params/our.yaml map "{}"
./run.sh map/reward/terrain/none Terrain-v0 "{}" ppo params/our.yaml map "{}"

./run.sh lstm/reward/terrain/both Terrain-v0 "{reward_explore: true, reward_closer: true}" ppo params/our.yaml lstm "{}"
./run.sh lstm/reward/terrain/none Terrain-v0 "{}" ppo params/our.yaml lstm "{}"

./run.sh map/reward/gaussian/both Gaussian-v0 "{reward_explore: true, reward_closer: true}" ppo params/our.yaml map "{}"
./run.sh lstm/reward/gaussian/both Gaussian-v0 "{reward_explore: true, reward_closer: true}" ppo params/our.yaml lstm "{}"

./run.sh map/reward/gaussian/none Gaussian-v0 "{}" ppo params/our.yaml map "{}"
./run.sh lstm/reward/gaussian/none Gaussian-v0 "{}" ppo params/our.yaml lstm "{}"
