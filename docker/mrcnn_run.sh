#!/bin/bash

CONTAINER_NAME=mrcnn
RUBIK_CONTAINER_IMAGE="rubik_mrcnn:latest" # "578276202366.dkr.ecr.us-east-1.amazonaws.com/rubik-mrcnn"
DATA_MOUNT="-v /home/ubuntu/coco:/home/ubuntu/coco"
docker run --runtime=nvidia --gpus 8 \
    --detach --rm --shm-size=16384m \
    --name $CONTAINER_NAME \
    --security-opt seccomp=unconfined  \
    -it \
    ${DATA_MOUNT} \
    ${RUBIK_CONTAINER_IMAGE} \
    bash
