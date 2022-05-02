#!/usr/bin/env bash

docker --host=ssh://oslund@bootcamp run --gpus all -it --rm nvcr.io/nvidia/pytorch:22.04-py3 --
