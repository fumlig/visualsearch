#!/usr/bin/env bash

id="$1"

env_id="$2"
env_params="$3"
alg_id="$4"
alg_params="$5"
agent_id="$6"
agent_params="$7"

num_envs=256
num_timesteps=25000000
ckpt_interval=1000000

for seed in 1 2 3
do
    name="$id-$seed"

    echo "$(date +%T): train $(tput bold)$name$(tput sgr0)"

    python3 train.py "$env_id" "$alg_id" "$agent_id" \
        --name="$name" \
        --seed="$seed" \
        --num-timesteps="$num_timesteps" \
        --num-envs="$num_envs" \
        --ckpt-interval="$ckpt_interval" \
        --env-kwargs="$env_params" \
        --alg-kwargs="$alg_params" \
        --agent-kwargs="$agent_params"

    echo "$(date +%T): test $(tput bold)$name$(tput sgr0)"

    python3 test.py "$env_id" \
        --name="$name" \
        --seed=0 \
        --hidden \
        --model="models/$name.pt" \
        --env-kwargs="$env_params"
done

# ./run.sh gaussian-ppo-lstm-small-r1 Gaussian-v0 "{shape: [10,10]}" ppo params/our.yaml lstm "{}"
# ./run.sh gaussian-ppo-lstm-small-r2 Gaussian-v0 "{shape: [10,10], reward_explore: true}" ppo params/our.yaml lstm "{}"
# ./run.sh gaussian-ppo-lstm-small-r3 Gaussian-v0 "{shape: [10,10], reward_closer: true}" ppo params/our.yaml lstm "{}"

# ./run.sh gaussian-ppo-map-small-r1 Gaussian-v0 "{shape: [10,10]}" ppo params/our.yaml map "{}"
# ./run.sh gaussian-ppo-map-small-r2 Gaussian-v0 "{shape: [10,10], reward_explore: true}" ppo params/our.yaml map "{}"
# ./run.sh gaussian-ppo-map-small-r3 Gaussian-v0 "{shape: [10,10], reward_closer: true}" ppo params/our.yaml map "{}"

# ./run.sh gaussian-ppo-map-m-r1 Gaussian-v0 "{shape: [15,15]}" ppo params/our.yaml map "{}"