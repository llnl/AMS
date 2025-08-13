# Use the base image from AMS project
FROM fluxrm/flux-core:bookworm AS base

LABEL maintainer="Loic Pottier <pottier1@llnl.gov>"

ARG USERNAME=ams-user
ARG USER_UID=1000
ARG USER_GID=1000
ENV USERNAME=${USERNAME}
ENV USER_UID=${USER_UID}
ENV USER_GID=${USER_GID}

USER root

# Install additional development tools
RUN apt-get update && apt-get install -y \
    lsb-release gnupg software-properties-common  \
    git python3 python3-pip mariadb-client \
    curl sudo \
    nlohmann-json3-dev \
    build-essential cmake \
    libopenmpi-dev libmetis-dev libhypre-dev \
    libblas-dev liblapack-dev \
    libhdf5-dev hdf5-tools \
    vim htop curl wget jq \
    python3-pip python3-venv python3-pika \
    python3-isort python3-cffi-backend \
    build-essential \
    cmake \
    ninja-build \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Caliper
RUN wget https://github.com/LLNL/Caliper/archive/refs/tags/v2.13.1.tar.gz -O caliper.tgz && \
    mkdir -p caliper && \
    tar -xzf caliper.tgz --strip-components=1 -C caliper && rm caliper.tgz && \
    cd caliper && mkdir build && cd build && \
    cmake .. && make -j4 && make install

# Install AMQP-CPP
RUN wget https://github.com/CopernicaMarketingSoftware/AMQP-CPP/archive/refs/tags/v4.3.27.tar.gz -O amqp-cpp.tgz && \
    mkdir -p amqp-cpp && \
    tar -xzf amqp-cpp.tgz --strip-components=1 -C amqp-cpp && rm amqp-cpp.tgz && \
    cd amqp-cpp && mkdir build && cd build && \
    cmake -DAMQP-CPP_BUILD_SHARED=ON -DAMQP-CPP_LINUX_TCP=ON .. && make -j4 && make install

ENV LD_LIBRARY_PATH=/usr/lib:/usr/local/lib:/usr/lib/aarch64-linux-gnu/

# Add the group and user that match our ids
RUN groupadd -g ${USER_GID} ${USERNAME} && \
    adduser --disabled-password --uid ${USER_UID} --gid ${USER_GID} --gecos "" ${USERNAME} && \
    echo "${USERNAME} ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers

# Create workspace directory
RUN mkdir -p /workspace && chown -R $USERNAME:$USERNAME /workspace

# Switch to non-root user
USER $USERNAME

# Set working directory
WORKDIR /workspace
