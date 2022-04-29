#!/usr/bin/env bash

./run.sh lstm/shape/s/r1 Gaussian-v0 "{shape: [10,10]}" ppo params/our.yaml lstm "{}"
./run.sh lstm/shape/s/r2 Gaussian-v0 "{shape: [10,10], reward_explore: true}" ppo params/our.yaml lstm "{}"
./run.sh lstm/shape/s/r3 Gaussian-v0 "{shape: [10,10], reward_closer: true}" ppo params/our.yaml lstm "{}"

./run.sh lstm/shape/m/r1 Gaussian-v0 "{shape: [15,15]}" ppo params/our.yaml lstm "{}"
./run.sh lstm/shape/m/r2 Gaussian-v0 "{shape: [15,15], reward_explore: true}" ppo params/our.yaml lstm "{}"
./run.sh lstm/shape/m/r3 Gaussian-v0 "{shape: [15,15], reward_closer: true}" ppo params/our.yaml lstm "{}"

./run.sh lstm/shape/l/r1 Gaussian-v0 "{shape: [20,20]}" ppo params/our.yaml lstm "{}"
./run.sh lstm/shape/l/r2 Gaussian-v0 "{shape: [20,20], reward_explore: true}" ppo params/our.yaml lstm "{}"
./run.sh lstm/shape/l/r3 Gaussian-v0 "{shape: [20,20], reward_closer: true}" ppo params/our.yaml lstm "{}"
