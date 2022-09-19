#!/bin/bash

dir_of_repo=${PWD} 
dir_of_dataset=$1

if [ "$(docker ps -aq -f status=exited -f name=netvlad)" ]; then
    docker rm netvlad;
fi

docker run -it --rm -d\
    --gpus all \
    --env="DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --privileged \
    --name netvlad \
    --ipc=host \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v $dir_of_repo/:/home/docker_netvlad:rw \
    -v $dir_of_dataset/:/datasets:rw \
    x64/netvlad_pytorch:latest
