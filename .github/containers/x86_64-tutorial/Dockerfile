# ┌──────────────┐
# │  BUILDER     │
# └──────────────┘

FROM fluxrm/flux-core:latest AS builder 
USER root
SHELL ["/bin/bash", "-lc"]

ARG num_jobs
ARG hypre_version

# Install MariaDB + tools
# Install extra packages (e.g., git, python3-pip)
RUN apt-get update && \
    apt-get install -y lsb-release gnupg software-properties-common  \
    			git python3 python3-pip mariadb-server mariadb-client \
			curl \
			rabbitmq-server supervisor \
			python3-venv \
			nlohmann-json3-dev \
			build-essential cmake \
		        libopenmpi-dev libmetis-dev libhypre-dev \
		        libblas-dev liblapack-dev \
			libhdf5-dev hdf5-tools && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /run/mysqld \
 && chown mysql:mysql /run/mysqld

RUN rabbitmq-plugins enable --offline rabbitmq_management
WORKDIR /opt/archives
#HYPRE
RUN curl -L https://github.com/hypre-space/hypre/archive/refs/tags/v${hypre_version}.tar.gz > /opt/archives/hypre-v${hypre_version}.tar.gz && \ 
    tar xzf hypre-v${hypre_version}.tar.gz && cd hypre-${hypre_version}/src && \
    ./configure --prefix /usr/local --enable-shared --disable-static && \
    make -j ${num_jobs} && make install && \
    rm -rf hypre-v${hypre_version}*

#MFEM
RUN git clone --depth=1  https://github.com/mfem/mfem.git /opt/archives/mfem && \
    mkdir -p /opt/archives/mfem/build && cd /opt/archives/mfem/build/ && \
    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local -DMFEM_USE_MPI=ON -DMFEM_USE_METIS_5=ON -DMFEM_USE_HYPRE=ON .. && \
    make -j"$(nproc)" && \
    make install

#CALIPER
RUN cd /opt/archives/ && \
    git clone --depth 1 https://github.com/LLNL/Caliper.git && \
    cd Caliper && \
    mkdir build && cd build && \
    cmake -DCMAKE_INSTALL_PREFIX=/usr/local/ .. && \
    make  && \
    make install

#AMQCPP
RUN cd /opt/archives/ && \
	git clone --depth=1 https://github.com/CopernicaMarketingSoftware/AMQP-CPP.git && \
	mkdir -p /opt/archives/AMQP-CPP/build/ && \
	cd /opt/archives/AMQP-CPP/build/ && \
	cmake -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=0"  -DAMQP-CPP_LINUX_TCP=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DAMQP-CPP_BUILD_SHARED=On .. && \
	make && \
	make install

RUN python3 -m venv /app/venv --system-site-packages
ENV PATH="/app/venv/bin:$PATH"
RUN source /app/venv/bin/activate && \
		python3 -m pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
		python -m pip install --no-cache-dir h5py

ENV AMS_MFEM_PATH=/usr/local/
ENV AMS_TORCH_PATH=/app/venv/lib/python3.11/site-packages/torch/share/cmake/Torch

RUN cd /opt/archives/ && \
	source /app/venv/bin/activate && \
	git clone --depth 1 https://github.com/LLNL/AMS.git -b features/tutorial && \
	mkdir -p /opt/archives/AMS/build-ams/ && \
	cd /opt/archives/AMS/build-ams/ && \
	cmake \
	  -DBUILD_SHARED_LIBS=On \
	  -DWITH_RMQ=On \
	  -DWITH_CALIPER=On \
	  -DWITH_HDF5=On \
          -DWITH_AMS_DEBUG=On \
	  -DCMAKE_BUILD_TYPE=Release \
	  -DCMAKE_INSTALL_PREFIX=/usr/local \
	  -DWITH_CUDA=Off \
	  -DMFEM_DIR=$AMS_MFEM_PATH \
	  -DWITH_MPI=On \
	  -DWITH_TORCH=On \
	  -DTorch_DIR=$AMS_TORCH_PATH \
	  -DWITH_WORKFLOW=On \
	  ../ && \
	  make && \
	  make install

# ┌──────────────┐
# │  RUNTIME     │
# └──────────────┘
FROM fluxrm/flux-core:latest AS runtime 
USER root
SHELL ["/bin/bash", "-lc"]

# 1) Install _only_ runtime deps
RUN apt-get update && \
    apt-get install -y lsb-release gnupg software-properties-common  \
    			git python3 python3-pip mariadb-server mariadb-client \
			curl \
			rabbitmq-server supervisor \
			python3-venv \
			nlohmann-json3-dev \
			build-essential cmake \
		        libopenmpi-dev libmetis-dev libhypre-dev \
		        libblas-dev liblapack-dev \
			libhdf5-dev hdf5-tools && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 2) Copy in your built artifacts
COPY --from=builder /usr/local /usr/local
COPY --from=builder /app/venv    /app/venv

# 3) Copy configs & entrypoint
COPY entrypoint.sh        /usr/local/bin/entrypoint.sh
COPY rabbitmq.conf        /etc/rabbitmq/rabbitmq.conf
COPY supervisord.conf     /etc/supervisor/conf.d/supervisord.conf
ENV PATH="/app/venv/bin:$PATH"

RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["supervisord", "-n", "-c", "/etc/supervisor/conf.d/supervisord.conf"]

