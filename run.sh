#!/usr/bin/env bash

function shape
{
    agent=$1
    side=$2
    seed=$3

    python3 train.py Gaussian-v0 $agent ppo \
        --name="shape/$agent/$side/$seed" \
        --seed=$seed \
        --num-timesteps=25000000 \
        --num-envs=64 \
        --num-checkpoints=250 \
        --env-kwargs="{shape: [$side,$side]}" \
        --alg-kwargs=params/procgen.yaml
}

function sample
{
    agent=$1
    sample=$2
    seed=$3

    python3 train.py Terrain-v0 $agent ppo \
        --name="sample/$agent/$sample/$seed" \
        --seed=$seed \
        --num-timesteps=25000000 \
        --num-envs=64 \
        --num-checkpoints=250 \
        --env-kwargs="{num_samples: $sample}" \
        --alg-kwargs=params/procgen.yaml
}


function experiment1
{
    {
        CUDA_VISIBLE_DEVICES=0 shape lstm 10 0 &
        CUDA_VISIBLE_DEVICES=0 shape map 10 0 &
        wait

        CUDA_VISIBLE_DEVICES=0 shape lstm 15 0 &
        CUDA_VISIBLE_DEVICES=0 shape map 15 0 &
        wait
        
        CUDA_VISIBLE_DEVICES=0 shape lstm 20 0 &
        CUDA_VISIBLE_DEVICES=0 shape map 20 0 &
        wait
    } &

    {
        CUDA_VISIBLE_DEVICES=1 shape lstm 10 1 &
        CUDA_VISIBLE_DEVICES=1 shape map 10 1 &
        wait
        
        CUDA_VISIBLE_DEVICES=1 shape lstm 15 1 &
        CUDA_VISIBLE_DEVICES=1 shape map 15 1 &
        wait

        CUDA_VISIBLE_DEVICES=1 shape lstm 20 1 &
        CUDA_VISIBLE_DEVICES=1 shape map 20 1 &
        wait
    } &

    {
        CUDA_VISIBLE_DEVICES=2 shape lstm 10 2 &
        CUDA_VISIBLE_DEVICES=2 shape map 10 2 &
        wait

        CUDA_VISIBLE_DEVICES=2 shape lstm 15 2 &
        CUDA_VISIBLE_DEVICES=2 shape map 15 2 &
        wait

        CUDA_VISIBLE_DEVICES=2 shape lstm 20 2 &
        CUDA_VISIBLE_DEVICES=2 shape map 20 2 &
        wait
    } &

    {
        CUDA_VISIBLE_DEVICES=3 shape lstm 10 3 &
        CUDA_VISIBLE_DEVICES=3 shape map 10 3 &
        wait
        
        CUDA_VISIBLE_DEVICES=3 shape lstm 15 3 &
        CUDA_VISIBLE_DEVICES=3 shape map 15 3 &
        wait
        
        CUDA_VISIBLE_DEVICES=3 shape lstm 20 3 &
        CUDA_VISIBLE_DEVICES=3 shape map 20 3 &
        wait
    } &

    wait
}


function experiment2
{
    {
        CUDA_VISIBLE_DEVICES=0 sample map null 0 &
        CUDA_VISIBLE_DEVICES=0 sample map null 1 &
        CUDA_VISIBLE_DEVICES=0 sample map null 2 &
        wait

        CUDA_VISIBLE_DEVICES=0 sample lstm null 0 &
        CUDA_VISIBLE_DEVICES=0 sample lstm null 1 &
        CUDA_VISIBLE_DEVICES=0 sample lstm null 2 &
        wait
    } &

    {
        CUDA_VISIBLE_DEVICES=1 sample map 1000 0 &
        CUDA_VISIBLE_DEVICES=1 sample map 1000 1 &
        CUDA_VISIBLE_DEVICES=1 sample map 1000 2 &
        wait

        CUDA_VISIBLE_DEVICES=1 sample lstm 1000 0 &
        CUDA_VISIBLE_DEVICES=1 sample lstm 1000 1 &
        CUDA_VISIBLE_DEVICES=1 sample lstm 1000 2 &
        wait
    } &

    {
        CUDA_VISIBLE_DEVICES=2 sample map 10000 0 &
        CUDA_VISIBLE_DEVICES=2 sample map 10000 1 &
        CUDA_VISIBLE_DEVICES=2 sample map 10000 2 &
        wait

        CUDA_VISIBLE_DEVICES=2 sample lstm 10000 0 &
        CUDA_VISIBLE_DEVICES=2 sample lstm 10000 1 &
        CUDA_VISIBLE_DEVICES=2 sample lstm 10000 2 &
        wait
    } &

    {
        CUDA_VISIBLE_DEVICES=3 sample map 100000 0 &
        CUDA_VISIBLE_DEVICES=3 sample map 100000 1 &
        CUDA_VISIBLE_DEVICES=3 sample map 100000 2 &
        wait

        CUDA_VISIBLE_DEVICES=3 sample lstm 100000 0 &
        CUDA_VISIBLE_DEVICES=3 sample lstm 100000 1 &
        CUDA_VISIBLE_DEVICES=3 sample lstm 100000 2 &
        wait
    } &

    wait
}

#experiment1
#experiment2

num_envs=64
alg_kwargs=params/procgen.yaml

seed=0

CUDA_VISIBLE_DEVICES=$seed python3 train.py Camera-v0 lstm ppo \
    --name="camera/lstm/$seed" \
    --seed=0 \
    --num-timesteps=25000000 \
    --num-envs=$num_envs \
    --num-checkpoints=250 \
    --alg-kwargs=params/procgen.yaml &

CUDA_VISIBLE_DEVICES=$seed python3 train.py Camera-v0 map ppo \
    --name="camera/map/$seed" \
    --seed=0 \
    --num-timesteps=25000000 \
    --num-envs=$num_envs \
    --num-checkpoints=250 \
    --alg-kwargs=params/procgen.yaml &

seed=1

CUDA_VISIBLE_DEVICES=$seed python3 train.py Camera-v0 lstm ppo \
    --name="camera/lstm/$seed" \
    --seed=0 \
    --num-timesteps=25000000 \
    --num-envs=$num_envs \
    --num-checkpoints=250 \
    --alg-kwargs=params/procgen.yaml &

CUDA_VISIBLE_DEVICES=$seed python3 train.py Camera-v0 map ppo \
    --name="camera/map/$seed" \
    --seed=0 \
    --num-timesteps=25000000 \
    --num-envs=$num_envs \
    --num-checkpoints=250 \
    --alg-kwargs=params/procgen.yaml &

seed=2

CUDA_VISIBLE_DEVICES=$seed python3 train.py Camera-v0 lstm ppo \
    --name="camera/lstm/$seed" \
    --seed=0 \
    --num-timesteps=25000000 \
    --num-envs=$num_envs \
    --num-checkpoints=250 \
    --alg-kwargs=params/procgen.yaml &

CUDA_VISIBLE_DEVICES=$seed python3 train.py Camera-v0 map ppo \
    --name="camera/map/$seed" \
    --seed=0 \
    --num-timesteps=25000000 \
    --num-envs=$num_envs \
    --num-checkpoints=250 \
    --alg-kwargs=params/procgen.yaml &

wait

# cat results/sample/map/$sample/$seed/test.csv | (sed -u 1q; sort -n -t',' -k1,1) > results/sample/map/$sample/$seed/sort.csv