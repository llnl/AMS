FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-devel AS base
MAINTAINER Loic Pottier <pottier1@llnl.gov>

RUN apt-get update && \
    apt-get install -y lsb-release gnupg software-properties-common  \
    		git \
            pkg-config \
			nlohmann-json3-dev \
			build-essential cmake \
		    libopenmpi-dev \
            libssl-dev \
		    libblas-dev liblapack-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV AMS_INSTALL=/usr/local/

#CALIPER + AMQCPP + HDF5
RUN mkdir -p /archives/ && \
    cd /archives/ && \
    git clone --depth 1 https://github.com/LLNL/Caliper.git && \
    cd Caliper && \
    mkdir build && cd build && \
    cmake -DCMAKE_INSTALL_PREFIX=${AMS_INSTALL} .. && \
    make && \
    make install && \
    cd /archives/ && \
	git clone --depth=1 https://github.com/CopernicaMarketingSoftware/AMQP-CPP.git && \
	mkdir -p /archives/AMQP-CPP/build/ && \
	cd /archives/AMQP-CPP/build/ && \
	cmake -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=0"  -DAMQP-CPP_LINUX_TCP=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${AMS_INSTALL} -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DAMQP-CPP_BUILD_SHARED=On .. && \
	make && \
	make install && \
    cd /archives/ && \
    git clone --depth 1 https://github.com/HDFGroup/hdf5.git && \
    cd hdf5 && \
    mkdir build && cd build && \
    cmake -DCMAKE_INSTALL_PREFIX=${AMS_INSTALL} .. && \
    make && \
    make install

RUN \
    python3 -m venv /venv --system-site-packages && \
    . /venv/bin/activate && \
    python3 -m pip install --no-cache-dir h5py

ENV AMS_TORCH_PATH=/opt/conda/lib/python3.11/site-packages/torch/share/cmake/Torch/
