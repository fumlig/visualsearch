#!/usr/bin/env bash

python3 train.py SearchGaussian-v0 ppo image --name=gaussian-image
python3 train.py SearchGaussian-v0 ppo recurrent --name=gaussian-recurrent
python3 train.py SearchGaussian-v0 ppo baseline --name=gaussian-baseline

