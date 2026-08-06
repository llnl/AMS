FROM almalinux:8 AS base
MAINTAINER Loic Pottier <pottier1@llnl.gov>

RUN \
    yum install -y dnf && \
    dnf install -y epel-release && \
    crb enable && \
    dnf config-manager --set-enabled powertools && \
    dnf group install -y "Development Tools" && \
    dnf install -y git jq wget sudo gcc-toolset-13 environment-modules cmake perl-devel libcurl-devel libevent-devel openssl-devel openmpi openmpi-devel && \
    dnf install -y lua-devel lua-posix python3.12-devel python3.12-pyyaml python3.12-cffi python3.12-pip zeromq-devel jansson-devel hwloc-devel lz4-devel json-devel expected-devel sqlite-devel libuuid-devel papi-devel ncurses-devel libarchive-devel && \
    dnf upgrade -y && \
    dnf clean all && \
    source /etc/profile && \
    mkdir -p /usr/share/Modules/modulefiles/gcc && \
    /usr/share/Modules/bin/createmodule.sh /opt/rh/gcc-toolset-13/enable > /usr/share/Modules/modulefiles/gcc/13.3.1 && \
    rm -rf /var/cache/dnf

ARG USERNAME=ams
# Create a dedicated group and user
RUN useradd -m ${USERNAME} && \
    usermod -aG wheel ${USERNAME} && \
    echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} && \
    chmod 0440 /etc/sudoers.d/${USERNAME} && \
    visudo -cf /etc/sudoers.d/${USERNAME}

USER $USERNAME
WORKDIR /home/$USERNAME

ARG FLUX_VERSION=0.81.0

#CALIPER + AMQCPP + HDF5 + Flux + fmt
RUN \
    source /etc/profile && \
    module load gcc/13.3.1 mpi/openmpi-x86_64 && \
    mkdir -p archives/ && \
    pushd archives/ && \
    wget https://github.com/flux-framework/flux-core/releases/download/v${FLUX_VERSION}/flux-core-${FLUX_VERSION}.tar.gz && \
    tar -xzf flux-core-${FLUX_VERSION}.tar.gz && \
    pushd flux-core-${FLUX_VERSION} && \
    ./configure && \
    make -j 4 && \
    sudo make install && \
    popd && \
    git clone --depth 1 https://github.com/LLNL/Caliper.git && \
    mkdir -p Caliper/build && \
    pushd Caliper/build && \
    cmake -DWITH_PAPI=On -DWITH_MPI=On .. && \
    make -j 4 && \
    sudo make install && \
    popd && \
	git clone --depth=1 https://github.com/CopernicaMarketingSoftware/AMQP-CPP.git && \
	mkdir -p AMQP-CPP/build/ && \
	pushd AMQP-CPP/build/ && \
	cmake -DAMQP-CPP_LINUX_TCP=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DAMQP-CPP_BUILD_SHARED=On .. && \
	make -j 4 && \
	sudo make install && \
    popd && \
    git clone --depth 1 https://github.com/HDFGroup/hdf5.git && \
    mkdir -p hdf5/build && \
    pushd hdf5/build && \
    cmake .. && \
    make -j 4 && \
    sudo make install && \
    popd && \
    git clone --depth 1 https://github.com/fmtlib/fmt.git -b 12.0.0 && \
    mkdir -p fmt/build && pushd fmt/build && \
    cmake -DBUILD_SHARED_LIBS=On .. && \
    make && \
    sudo make install && \
    popd && \
    popd && \
    rm -rf archives/

RUN \
    pip3 install "h5py==3.16.0" "mysql-connector-python==9.7.0" "SQLAlchemy==2.0.51" "pika==1.4.0" "flux-python==${FLUX_VERSION}" && \
    pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# AMS_TORCH_PATH: given by: $(python3 -c 'import torch; print(torch.__path__[0])')/share/cmake/Torch 
ENV AMS_TORCH_PATH=/home/ams/.local/lib/python3.12/site-packages/torch/share/cmake/Torch
ENV AMS_CALIPER_PATH=/usr/local/share/cmake/caliper/
ENV AMS_AMQPCPP_PATH=/usr/local/cmake/
ENV AMS_HDF5_PATH=/usr/local/HDF_Group/HDF5/2.3.0/lib64/cmake/hdf5/
ENV AMS_NLOHMANN_JSON_DIR=/usr/lib64/cmake/nlohmann_json/
ENV AMS_FMT_DIR=/usr/local/lib64/cmake/fmt/
ENV AMS_TL_EXPECTED_DIR=/usr/share/cmake/tl-expected
ENV LD_LIBRARY_PATH="/usr/local/lib/:${LD_LIBRARY_PATH:-}"
# to let root has access to the python packages
ENV PYTHONPATH=/home/${USERNAME}/.local/lib/python3.12/site-packages/

