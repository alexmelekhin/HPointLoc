#!/bin/bash

docker exec --user "docker_netvlad" -it netvlad \
    /bin/bash -c "cd /home/docker_netvlad; /bin/bash"
