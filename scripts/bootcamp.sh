#!/usr/bin/env bash

TAG="$USER/thesis:latest"
HOST="bootcamp"

docker build --tag=$TAG .
docker save $TAG | ssh -C ${USER}@${HOST} docker load
docker --host=ssh://${USER}@${HOST} run --rm --gpus=all -it $TAG
