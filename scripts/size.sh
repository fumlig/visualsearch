#!/usr/bin/env bash

# experiment 1
# todo

for shape in "[10,10]" "[15,15]" "[20,20]"
do
    for agent in "baseline" "map"
    do
        for seed in 1 2 3
            do
                # r
                python3 train.py "Gaussian-v0" ppo $agent \
                    --seed=$seed --num-envs=128 \
                    --env-kwargs="{shape: $shape}" \
                    --alg-kwargs=params/our.yaml
                
                # r'
                python3 train.py "Gaussian-v0" ppo $agent \
                    --seed=$seed --num-envs=128 \
                    --env-kwargs="{shape: $shape, punish_revisit: true}" \
                    --alg-kwargs=params/our.yaml
                
                # r''
                python3 train.py "Gaussian-v0" ppo $agent \
                    --seed=$seed --num-envs=128 \
                    --env-kwargs="{shape: $shape, reward_closer: true}" \
                    --alg-kwargs=params/our.yaml
            done
        done
    done
done

# experiment 2
# todo

# experiment 3
# todo