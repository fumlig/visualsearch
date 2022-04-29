#!/usr/bin/env bash

./run.sh map/shape/s/r1 Gaussian-v0 "{shape: [10,10]}" ppo params/our.yaml map "{}"
./run.sh map/shape/s/r2 Gaussian-v0 "{shape: [10,10], reward_explore: true}" ppo params/our.yaml map "{}"
./run.sh map/shape/s/r3 Gaussian-v0 "{shape: [10,10], reward_closer: true}" ppo params/our.yaml map "{}"

./run.sh map/shape/m/r1 Gaussian-v0 "{shape: [15,15]}" ppo params/our.yaml map "{}"
./run.sh map/shape/m/r2 Gaussian-v0 "{shape: [15,15], reward_explore: true}" ppo params/our.yaml map "{}"
./run.sh map/shape/m/r3 Gaussian-v0 "{shape: [15,15], reward_closer: true}" ppo params/our.yaml map "{}"

./run.sh map/shape/l/r1 Gaussian-v0 "{shape: [20,20]}" ppo params/our.yaml map "{}"
./run.sh map/shape/l/r2 Gaussian-v0 "{shape: [20,20], reward_explore: true}" ppo params/our.yaml map "{}"
./run.sh map/shape/l/r3 Gaussian-v0 "{shape: [20,20], reward_closer: true}" ppo params/our.yaml map "{}"
