FROM nvidia/cuda:12.9.1-devel-ubi9 AS base
MAINTAINER Loic Pottier <pottier1@llnl.gov>
RUN \
    yum install -y dnf && \
    dnf install -y git xz perl cmake curl-devel libcurl-devel autoconf automake unzip patch gcc-gfortran bzip2 bzip2-devel file libevent libevent-devel openssl openssl-devel openblas openblas-devel zlib zlib-devel python3-devel && \
    dnf upgrade -y && \
    dnf clean all
COPY repo repo
RUN \
    mkdir -p ams-spack-env
COPY spack.yaml ams-spack-env/spack.yaml

FROM base AS setup-spack-env
RUN \
    git clone --depth=1 https://github.com/spack/spack.git -b v1.0.2 && \
    source spack/share/spack/setup-env.sh && \
    spack env activate -p ams-spack-env && \
    spack external find --all --not-buildable --exclude openssl --exclude openblas
RUN \
    source spack/share/spack/setup-env.sh && \
    spack env activate -p ams-spack-env && \
    spack install && \
    spack env view enable && \
    spack clean --all

ENV SPACK_VIEW=/ams-spack-env/.spack-env/view/share/cmake/