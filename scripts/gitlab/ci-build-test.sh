#!/bin/bash
echo ${CI_PROJECT_DIR}

source scripts/gitlab/setup-env.sh

export CTEST_OUTPUT_ON_FAILURE=1
export NUMEXPR_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1

cleanup() {
  if [ -n "$VIRTUAL_ENV" ]; then
    deactivate
  fi
  rm -rf ci-venv
  rm -rf build
}

build_and_test() {

  echo "*******************************************************************************************"
  echo "Build configuration" \
    "WITH_HDF5 ${WITH_HDF5}" \
    "WITH_MPI ${WITH_MPI}" \
    "WITH_WORKFLOW ${WITH_WORKFLOW}" \
    "WITH_CUDA ${WITH_CUDA}" \
    "WITH_HIP ${WITH_HIP}"
  echo "*******************************************************************************************"

  build_dir="/tmp/ams/$(uuidgen)"
  mkdir -p ${build_dir}
  pushd ${build_dir}

  cleanup

  python -m venv ci-venv
  source ci-venv/bin/activate
  mkdir build
  pushd build

  cmake \
    -DBUILD_SHARED_LIBS=On \
    -DCMAKE_PREFIX_PATH=$INSTALL_DIR \
    -DWITH_CALIPER=On \
    -DWITH_HDF5=${WITH_HDF5} \
    -DAMS_HDF5_DIR=$AMS_HDF5_PATH \
    -DCMAKE_INSTALL_PREFIX=./install \
    -DCMAKE_BUILD_TYPE=Release \
    -DCUDA_ARCH=$AMS_CUDA_ARCH \
    -DWITH_CUDA=${WITH_CUDA} \
    -DWITH_HIP=${WITH_HIP} \
    -DWITH_MPI=${WITH_MPI} \
    -DWITH_TESTS=On \
    -DTorch_DIR=$AMS_TORCH_PATH \
    -DWITH_AMS_DEBUG=On \
    -DWITH_WORKFLOW=${WITH_WORKFLOW} \
    ${CI_PROJECT_DIR} || { echo "CMake failed"; exit 1; }

  make -j || { echo "Building failed"; exit 1; }
  make test || { echo "Tests failed"; exit 1; }
  popd

  cleanup

  popd
  rm -rf ${build_dir}
}

build_and_test
