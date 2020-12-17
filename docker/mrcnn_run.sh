#!/bin/bash

CONTAINER_NAME=rubik_mrcnn
RUBIK_CONTAINER_IMAGE="578276202366.dkr.ecr.us-east-1.amazonaws.com/rubik-mrcnn"
DATA_MOUNT="-v /home/ubuntu/coco:/home/ubuntu/coco"
RUBIK_MOUNT="-v /home/ubuntu/rubik:/rubik"
RESNET_MOUNT="-v /home/ubuntu/.torch/models:/root/.torch/models"
CODE_MOUNT="-v /home/ubuntu/training_results_v0.7:/training_results_v0.7"
docker run --runtime=nvidia --gpus 8 \
    --detach --rm --shm-size=16384m \
    --name $CONTAINER_NAME \
    --security-opt seccomp=unconfined  \
    -it \
    ${DATA_MOUNT} \
    ${CODE_MOUNT} \
    ${RUBIK_MOUNT} \
    ${RESNET_MOUNT} \
    ${RUBIK_CONTAINER_IMAGE} \
    bash
#docker exec ${CONTAINER_NAME} bash -c "/usr/sbin/sshd -D -f /home/efauser/.ssh/sshd_config &"
