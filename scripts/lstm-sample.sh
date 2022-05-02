#!/usr/bin/env bash

export NUM_TIMESTEPS=25000000

./run.sh map/sample/inf Terrain-v0 "{}" ppo params/our.yaml lstm "{}"
./run.sh map/sample/100 Terrain-v0 "{train_samples: 100}" ppo params/our.yaml lstm "{}"
./run.sh map/sample/1000 Terrain-v0 "{train_samples: 1000}" ppo params/our.yaml lstm "{}"
./run.sh map/sample/10000 Terrain-v0 "{train_samples: 10000}" ppo params/our.yaml lstm "{}"
./run.sh map/sample/100000 Terrain-v0 "{train_samples: 100000}" ppo params/our.yaml lstm "{}"
