#FROM ubuntu:18.04
FROM nvidia/cudagl:10.0-devel-ubuntu18.04
MAINTAINER naverlabs "kapture@naverlabs.com"

# setup environment
ENV     LANG C.UTF-8
ENV     LC_ALL C.UTF-8
ENV     DEBIAN_FRONTEND noninteractive
# arguments
ARG     MAKE_OPTIONS="-j8"
ARG     SOURCE_PREFIX="/opt/src"

RUN mkdir -p ${SOURCE_PREFIX}

# Get dependencies
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    git \
    wget\
    curl \
    python3.6 python3-pip \
    pandoc asciidoctor \
    cmake \
    build-essential \
    libboost-all-dev \
    libsuitesparse-dev \
    libfreeimage-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libglew-dev \
    freeglut3-dev \
    libxmu-dev \
    libxi-dev \
    libatlas-base-dev \
    libsuitesparse-dev \
    libcgal-qt5-dev \
    libqt5opengl5-dev \
    qt5-default \
    x11-apps \
    mesa-utils \
  && rm -rf /var/lib/apt/lists/*


########################################################################################################################
# PYTHON-PIP ###########################################################################################################
# make sure pip 3 is >= 20.0 to enable use-feature=2020-resolver
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install --upgrade setuptools wheel twine


########################################################################################################################
# COLMAP ###############################################################################################################
# ├── eigen
# └── ceres

# Eigen 3.2.10
WORKDIR ${SOURCE_PREFIX}
RUN     git clone -b 3.2.10 https://gitlab.com/libeigen/eigen.git eigen
RUN     mkdir -p eigen/build
WORKDIR ${SOURCE_PREFIX}/eigen/build
RUN     cmake \
        -DCMAKE_BUILD_TYPE=Release \
         .. && \
        make ${MAKE_OPTIONS} && make install && make clean

# ceres 1.14.0
WORKDIR ${SOURCE_PREFIX}
RUN     git clone -b 1.14.0 https://github.com/ceres-solver/ceres-solver.git
RUN     mkdir -p ceres-solver/build
WORKDIR ${SOURCE_PREFIX}/ceres-solver/build
RUN     cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_TESTING=OFF \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_BENCHMARKS=OFF \
        ../ && \
        make ${MAKE_OPTIONS} && make install && make clean

# colmap
WORKDIR ${SOURCE_PREFIX}
RUN     git clone -b 3.6-dev.3 https://github.com/colmap/colmap.git
RUN     mkdir -p colmap/build
WORKDIR colmap/build
RUN     cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -DTESTS_ENABLED=OFF \
        .. && \
        make ${MAKE_OPTIONS} && make install && make clean


########################################################################################################################
# install kapture from pip.
RUN      python3 -m pip install kapture

# install kapture-localization
ADD      . ${SOURCE_PREFIX}/kapture-localization
WORKDIR  ${SOURCE_PREFIX}/kapture-localization
RUN      python3 -m pip install "torch==1.4.0" "torchvision==0.5.0" "scikit_learn==0.20.2"
RUN      python3 -m pip install -r requirements.txt --use-feature=2020-resolver
RUN      python3 setup.py install


### FINALIZE ###################################################################
# save space: purge apt-get
RUN     rm -rf /var/lib/apt/lists/*
USER  root
WORKDIR ${SOURCE_PREFIX}/
