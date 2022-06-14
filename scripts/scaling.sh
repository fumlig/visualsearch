#!/usr/bin/env bash

# search space size experiment

search_space_sizes=10 15 20
environment="gaussian"
agents="lstm" "map"
algorithm="ppo"
seeds=0 1 2

num_timesteps=25000000
num_envs=64
num_checkpoints=250
alg_hparams=params/procgen.yaml

for shape_side in $search_space_sizes
do
    for agent in $agents
    do
        for seed in $seeds
        do
            # train agent
            python3 train.py $environment $agent $algorithm \
                --name="$environment/$agent/$seed" \
                --seed=$seed \
                --num-timesteps=$train_timesteps \
                --num-envs=$num_envs \
                --num-checkpoints=$num_checkpoints \
                --env-kwargs="{shape: [$side,$side]}" \
                --alg-kwargs=$alg_hparams
        done
    done
done
