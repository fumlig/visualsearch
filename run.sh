#!/usr/bin/env bash

function train
{
    name=$1
    env_id=$2
    agent_id=$3
    alg_id=$4
    seed=$5
    device=$6

    env_kwargs=${"{}":-env_kwargs}
    agent_kwargs=${"{}":-agent_kwargs}
    alg_kwargs=${"params/procgen.yaml":-alg_kwargs}

    num_timesteps=25000000
    num_envs=64
    num_checkpoints=250

    CUDA_VISIBLE_DEVICES=$device train.py $environment $agent $algorithm \
        --name="$name" \
        --seed="$seed" \
        --num-timesteps=$num_timesteps \
        --num-checkpoints=$num_checkpoints \
        --env-kwargs=$env_kwargs
}



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


{
    CUDA_VISIBLE_DEVICES=0 python3 test.py Terrain-v0 --hidden --name=sample/map/500/0 --models=models/sample/map/500/0/ckpt/* &   
    CUDA_VISIBLE_DEVICES=0 python3 test.py Terrain-v0 --hidden --name=sample/map/500/1 --models=models/sample/map/500/1/ckpt/* &
    CUDA_VISIBLE_DEVICES=0 python3 test.py Terrain-v0 --hidden --name=sample/map/500/2 --models=models/sample/map/500/2/ckpt/* &
    wait
} &

"'
{
    CUDA_VISIBLE_DEVICES=1 python3 test.py Terrain-v0 --hidden --name=sample/map/5000/0 --models=models/sample/map/5000/0/ckpt/* &   
    CUDA_VISIBLE_DEVICES=1 python3 test.py Terrain-v0 --hidden --name=sample/map/5000/1 --models=models/sample/map/5000/1/ckpt/* &
    CUDA_VISIBLE_DEVICES=1 python3 test.py Terrain-v0 --hidden --name=sample/map/5000/2 --models=models/sample/map/5000/2/ckpt/* &
    wait
} &

{
    CUDA_VISIBLE_DEVICES=2 sample lstm 500 0 &
    CUDA_VISIBLE_DEVICES=2 sample lstm 500 1 &
    CUDA_VISIBLE_DEVICES=2 sample lstm 500 2 &
    wait

    CUDA_VISIBLE_DEVICES=2 python3 test.py Terrain-v0 --hidden --name=sample/lstm/500/0 --models=models/sample/lstm/500/0/ckpt/* &   
    CUDA_VISIBLE_DEVICES=2 python3 test.py Terrain-v0 --hidden --name=sample/lstm/500/1 --models=models/sample/lstm/500/1/ckpt/* &
    CUDA_VISIBLE_DEVICES=2 python3 test.py Terrain-v0 --hidden --name=sample/lstm/500/2 --models=models/sample/lstm/500/2/ckpt/* &
    wait
} &

{
    CUDA_VISIBLE_DEVICES=3 sample lstm 5000 0 &
    CUDA_VISIBLE_DEVICES=3 sample lstm 5000 1 &
    CUDA_VISIBLE_DEVICES=3 sample lstm 5000 2 &
    wait

    CUDA_VISIBLE_DEVICES=3 python3 test.py Terrain-v0 --hidden --name=sample/lstm/5000/0 --models=models/sample/lstm/5000/0/ckpt/* &   
    CUDA_VISIBLE_DEVICES=3 python3 test.py Terrain-v0 --hidden --name=sample/lstm/5000/1 --models=models/sample/lstm/5000/1/ckpt/* &
    CUDA_VISIBLE_DEVICES=3 python3 test.py Terrain-v0 --hidden --name=sample/lstm/5000/2 --models=models/sample/lstm/5000/2/ckpt/* &
    wait
} &
"

wait

# cat results/sample/map/$sample/$seed/test.csv | (sed -u 1q; sort -n -t',' -k1,1) > results/sample/map/$sample/$seed/sort.csv