#!/bin/bash

PORT=$1
CODE=$2
DATASETS=$3
MODEL=$4

docker run -itd \
           --ipc host \
           --gpus all \
           -e "NVIDIA_DRIVER_CAPABILITIES=all" \
           -e "DISPLAY" \
           -e "QT_X11_NO_MITSHM=1" \
           -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
           -p $PORT:22 \
           -v $CODE:/home/docker_dxslam/catkin_ws/src/dxslam:rw \
           -v $DATASETS:/datasets:ro \
           -v $MODEL:/home/docker_dxslam/model:ro \
           -v /home/${USER}:/home/${USER}:rw \
           --name dxslam \
           nvcr.io/nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

# docker exec --user root \
#             dxslam \
#             /bin/bash -c "/etc/init.d/ssh start"
