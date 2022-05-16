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

for approach in lstm map
do
    python3 test.py Gaussian-v0 \
        --models=models/gaussian/$approach/*.pt \
        --name=gaussian/$approach \
        --episodes=100 \
        --runs=3 \
        --seed=0 \
        --deterministic \
        --hidden &

    python3 test.py Terrain-v0 \
        --models=models/terrain/$approach/*.pt \
        --name=terrain/$approach \
        --episodes=100 \
        --runs=3 \
        --seed=0 \
        --deterministic \
        --hidden &

    # todo: weird directory
    python3 test.py Camera-v0 \
        --models=models/camera/camera/$approach/*.pt \
        --name=camera/$approach \
        --episodes=100 \
        --runs=3 \
        --seed=0 \
        --deterministic \
        --hidden &

    wait
done
