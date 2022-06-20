#!/usr/bin/env bash

# search generalization experiment

experiment="generalization"
training_set_sizes=10000 5000 1000 5000
environment="terrain"
agents="lstm" "map"
algorithm="ppo"
seeds=0 1 2

num_timesteps=25000000
num_envs=64
num_checkpoints=250
alg_hparams=params/procgen.yaml

for num_samples in $training_set_sizes
do
    for agent in $agents
    do
        for seed in $seeds
        do
            # train agent
            python3 train.py $environment $agent $algorithm \
                --name="$experiment/$agent/$num_samples/$seed" \
                --seed=$seed \
                --num-timesteps=$num_timesteps \
                --num-envs=$num_envs \
                --num-checkpoints=$num_checkpoints \
                --env-kwargs="{num_samples: $num_samples}" \
                --alg-kwargs=$alg_hparams
        
            # test agent (checkpoints on separate set)
            python3 test.py $environment \
                --models=models/$experiment/$agent/$num_samples/$seed/ckpt/*.pt \
                --name="$experiment/$agent/$num_samples/$seed"
                --seed=0 \
                --hidden \
                --env-kwargs="{first_sample: $num_samples}"
        done
    done
done
