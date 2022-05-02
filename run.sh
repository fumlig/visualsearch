#!/usr/bin/env bash

id="$1"

env_id="$2"
env_params="$3"
alg_id="$4"
alg_params="$5"
agent_id="$6"
agent_params="$7"

num_seeds="${NUM_SEEDS:-5}"
num_envs="${NUM_ENVS:-256}"
num_timesteps="${NUM_TIMESTEPS:-25000000}"
ckpt_interval="${CKPT_INTERVAL:-1000000}"


for seed in $(seq 1 $num_seeds)
do
    name="$id/$seed"

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
