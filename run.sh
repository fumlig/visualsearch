#!/usr/bin/env bash

for agent in "image" "recurrent" "baseline" "map"
do
    for environment in "Gaussian-v0" "Terrain-v0" "Camera-v0"
    do
        for seed in 1 2 3
        do
            python3 train.py $environment ppo $agent --seed=$seed
        done
    done
done