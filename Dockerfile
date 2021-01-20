FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime

ENV TZ=America/Chicago

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Apt installs
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    g++ \
    make \
    cmake \
    build-essential \
    wget \
    curl \
    vim \
    nano \
    emacs \
    git \
    python3 \
    python3-pip \
    pkg-config \
    libgl1-mesa-glx \
    libgtk2.0-dev \
    libopencv-dev

# Pip installs
RUN python3 -m pip install --upgrade pip && DEBIAN_FRONTEND=noninteractive python3 -m pip install \
    comet-ml \
    matplotlib \
    scipy \
    h5py \
    opencv-python \
    opencv-contrib-python \ 
    kornia

