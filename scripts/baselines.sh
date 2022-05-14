#!/usr/bin/env bash

for baseline in random greedy exhaustive
do
    python3 test.py Gaussian-v0 \
        --agent=$baseline \
        --name=gaussian/$baseline \
        --episodes=100 \
        --runs=3 \
        --seed=0 \
        --hidden &

    python3 test.py Terrain-v0 \
        --agent=$baseline \
        --name=terrain/$baseline \
        --episodes=100 \
        --runs=3 \
        --seed=0 \
        --hidden &

    python3 test.py Camera-v0 \
        --agent=$baseline \
        --name=camera/$baseline \
        --episodes=100 \
        --runs=3 \
        --seed=0 \
        --hidden &
    
    wait
done
