FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel

RUN apt-get update && apt-get install -y --no-install-recommends \
        make \
        pkgconf \
        xz-utils \
        gcc \
        g++ \
        cmake \
        xorg-dev \
        libgl1-mesa-dev \
        libglu1-mesa-dev \
        libxrandr-dev \
        libxinerama-dev \
        libxcursor-dev \
        libxi-dev \
        libxxf86vm-dev \
        graphviz \
        ffmpeg \
        && rm -rf /var/lib/apt/lists/*

# Install updated CMake
RUN apt-get update && apt-get install -y wget && \
    wget https://github.com/Kitware/CMake/releases/download/v3.27.7/cmake-3.27.7-linux-x86_64.sh && \
    mkdir /opt/cmake && \
    sh cmake-3.27.7-linux-x86_64.sh --skip-license --prefix=/opt/cmake && \
    ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \ 
    pip install --no-cache-dir -r requirements.txt

# set workdir 
ENV WORKDIR=/root/workspace

WORKDIR $WORKDIR

# Set the default command
CMD ["/bin/bash"]
