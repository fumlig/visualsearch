#!/usr/bin/env bash

# experiment 1
# todo

for shape in "[8,8]" "[16,16]" "[24,24]" "[32,32]"
do
    for agent in "baseline" "map"
    do
        for seed in 1 2 3
            do
                python3 train.py "Gaussian-v0" ppo $agent --seed=$seed --num-envs=128 --env-kwargs="{shape: $shape}" --alg-kwargs=params/our.yaml
            done
        done
    done
done

# experiment 2
# todo

# experiment 3
# todo