#!/usr/bin/env bash

# search performance experiment

environments="gaussian" "terrain" "camera"
agents="lstm" "map"
baselines="random" "greedy" "exhaustive"
algorithm="ppo"
seeds=0 1 2

train_timesteps=25000000
test_episodes=100
num_envs=64
num_checkpoints=250
alg_hparams=params/procgen.yaml

for environment in $environments
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
                --alg-kwargs=$alg_hparams
        done

        # test agent
        python3 test.py $environment \
            --models=models/$environment/$agent/*.pt \
            --name=gaussian/$agent \
            --episodes=$test_episodes \
            --seed=0 \
            --hidden
    done

    for baseline in $baselines
    do
        # test baseline
        python3 test.py $environment \
            --agent=$baseline \
            --name=$environment/$baseline \
            --episodes=$test_episodes \
            --seed=0 \
            --runs=3 \
            --hidden 
    done
done
