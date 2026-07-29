FROM almalinux:8 AS base
MAINTAINER Loic Pottier <pottier1@llnl.gov>

RUN \
    yum install -y dnf && \
    dnf group install -y "Development Tools" && \
    dnf install -y git wget sudo gcc-toolset-13 environment-modules cmake perl-devel libevent-devel openssl-devel openmpi openmpi-devel libcurl-devel && \
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

FROM base AS setup-spack-env
RUN \
    source /etc/profile && \
    module load gcc/13.3.1 && \
    git clone --depth 1 --branch v1.1.0 https://github.com/spack/spack.git && \
    source spack/share/spack/setup-env.sh && \
    spack compiler find && \
    spack compiler rm gcc@8.5.0

COPY --chown=ams:ams repo repo
RUN mkdir -p ams-spack-env
COPY --chown=ams:ams spack.yaml ams-spack-env/spack.yaml

FROM setup-spack-env AS install-spack-env
RUN \
    source /etc/profile && \
    module load gcc/13.3.1 && \
    source spack/share/spack/setup-env.sh && \
    spack env activate -p ams-spack-env && \
    spack install -j 2 --fail-fast && \
    spack clean --all

COPY --chown=ams:ams load-env.sh load-env.sh
RUN chmod +x load-env.sh
