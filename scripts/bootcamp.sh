#!/usr/bin/env bash

TAG="$USER/thesis:latest"
HOST="bootcamp"

docker build --tag=$TAG .
docker save $TAG | ssh -C ${USER}@${HOST} docker load
docker --host=ssh://${USER}@${HOST} run --rm --gpus=all --volume=/home/${USER}/logs:/app/logs --volume=/home/${USER}/models:/app/models --volume=/home/${USER}/results:/app/results -it $TAG
