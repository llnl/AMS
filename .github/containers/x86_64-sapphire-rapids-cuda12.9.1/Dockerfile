FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-devel AS base
MAINTAINER Loic Pottier <pottier1@llnl.gov>

# apt-sandbox: to avoid setgroups error when running apt
RUN echo 'APT::Sandbox::User root;' > /etc/apt/apt.conf.d/00-apt-sandbox && \
    apt-get update && apt-get install -y sudo

# add an "ams" user with sudo right (useful to test MPI code as non-root)
RUN adduser --disabled-password --gecos '' ams && \
    adduser ams sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

USER ams

RUN sudo apt-get update && \
    sudo apt-get install -y lsb-release gnupg software-properties-common  \
    		git \
            pkg-config \
			nlohmann-json3-dev \
			build-essential cmake \
		    libopenmpi-dev \
            libssl-dev \
            libexpected-dev \
            libfmt-dev \
		    libblas-dev liblapack-dev \
            wget \
            lua5.1 liblua5.1-dev lua-posix-dev \
            libzmq3-dev \
            libjansson-dev \
            liblz4-dev \
            libsqlite3-dev \
            uuid-dev \
            libarchive-dev && \
    sudo apt-get clean && \
    sudo rm -rf /var/lib/apt/lists/*

ENV AMS_INSTALL=/home/ams/local/
ARG FLUX_VERSION=0.81.0

#CALIPER + AMQCPP + HDF5 + Flux -DCMAKE_INSTALL_PREFIX=$AMS_INSTALL
RUN mkdir -p $AMS_INSTALL && mkdir -p $HOME/archives/ && \
    cd $HOME/archives/ && \
    wget https://github.com/flux-framework/flux-core/releases/download/v${FLUX_VERSION}/flux-core-${FLUX_VERSION}.tar.gz && \
    tar -xzf flux-core-${FLUX_VERSION}.tar.gz && \
    cd flux-core-${FLUX_VERSION} && \
    pip install ply && \
    ./configure && \
    make && \
    sudo make install && \
    git clone --depth 1 https://github.com/LLNL/Caliper.git && \
    cd Caliper && \
    mkdir build && cd build && \
    cmake .. && \
    make && \
    sudo make install && \
    cd $HOME/archives/ && \
	git clone --depth=1 https://github.com/CopernicaMarketingSoftware/AMQP-CPP.git && \
	mkdir -p $HOME/archives/AMQP-CPP/build/ && \
	cd $HOME/archives/AMQP-CPP/build/ && \
	cmake -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DAMQP-CPP_LINUX_TCP=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DAMQP-CPP_BUILD_SHARED=On .. && \
	make && \
	sudo make install && \
    cd $HOME/archives/ && \
    git clone --depth 1 https://github.com/HDFGroup/hdf5.git && \
    cd hdf5 && \
    mkdir build && cd build && \
    cmake .. && \
    make && \
    sudo make install && \
    rm -rf $HOME/archives/

RUN \
    python3 -m venv $HOME/venv --system-site-packages && \
    . $HOME/venv/bin/activate && \
    pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir "h5py==3.16.0" "mysql-connector-python==9.7.0" "SQLAlchemy==2.0.51" "pika==1.4.0" "flux-python==${FLUX_VERSION}"

ENV AMS_TORCH_PATH=/opt/conda/lib/python3.11/site-packages/torch/share/cmake/Torch/
ENV AMS_CALIPER_PATH=/usr/local/share/cmake/caliper/
ENV AMS_AMQPCPP_PATH=/usr/local/cmake/
ENV AMS_HDF5_PATH=/usr/local/HDF_Group/HDF5/2.3.0/lib/cmake/hdf5/
ENV AMS_NLOHMANN_JSON_DIR=/usr/lib/cmake/nlohmann_json/
ENV AMS_FMT_DIR=/usr/lib/x86_64-linux-gnu/cmake/fmt/
ENV AMS_TL_EXPECTED_DIR=/usr/share/cmake/tl-expected
ENV LD_LIBRARY_PATH="/usr/local/lib/:${LD_LIBRARY_PATH:-}"
WORKDIR /home/ams/
