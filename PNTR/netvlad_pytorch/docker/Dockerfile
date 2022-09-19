FROM nvidia/cuda:11.0-devel-ubuntu20.04
#FROM nvidia/cuda:11.0-devel-centos7

#install required libs and packages
RUN apt-get update  \
	&& apt-get install -y --no-install-recommends build-essential git curl ca-certificates\
	libsparsehash-dev python3-dev libjpeg-dev libpng-dev python3-pip  \
	&& rm -rf /var/lib/apt/lists/*
 

ENV CUDA cu110

# CUDA 11.0 path
ENV PATH=/usr/local/cuda-11.0/bin:$PATH
ENV CUDA_PATH=/usr/local/cuda-11.0
ENV CUDA_HOME=/usr/local/cuda-11.0
ENV LD_LIBRARY_PATH=/usr/local/cuda-11.0/lib64:$LD_LIBRARY_PATH

# Install miniconda to /miniconda
RUN curl -LO http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -p /miniconda -b
RUN rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH=/miniconda/bin:${PATH}
RUN conda update -y conda

ENV FORCE_CUDA="1"
ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"

# Pytorch installation
ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
RUN python3 -m pip install pip --upgrade
RUN python3 -m pip install wheel
RUN pip install torch==1.6 torchvision==0.7.0	

# Installing faiss
RUN conda install -c pytorch faiss-gpu

# Installing scipy
RUN pip install numpy scipy matplotlib ipython jupyter pandas sympy nose

# Installing sklearn
RUN pip install -U scikit-learn

# Installing h5py
RUN pip3 install h5py

# Installing tensorboardX
RUN pip install 'git+https://github.com/lanpa/tensorboardX'
RUN pip install crc32c
RUN pip install soundfile
 

# User Staff
ARG USER=docker_netvlad
ARG UID=1000
ARG GID=1000
# default password
ARG PW=user 

RUN useradd -m ${USER} --uid=${UID} && echo "${USER}:${PW}" | chpasswd && adduser ${USER} sudo

WORKDIR /home/${USER}
RUN cd /home/${USER} && chown -R ${UID}:${GID} ./

USER ${UID}:${GID}
