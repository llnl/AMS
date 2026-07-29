#!/usr/bin/env bash

source /etc/profile
module load gcc/13.3.1 mpi/openmpi-x86_64 
source ${HOME}/spack/share/spack/setup-env.sh
spack env activate -p ${HOME}/ams-spack-env

export SPACK_ROOT=${HOME}/spack/

export AMS_TORCH_PATH="$(spack location -i py-torch)/lib/python3.13/site-packages/torch/share/cmake/Torch/"
export AMS_CALIPER_PATH=$(spack location -i caliper)/share/cmake/caliper/
export AMS_AMQPCPP_PATH=$(spack location -i amqp-cpp)/cmake/
export AMS_HDF5_PATH=$(spack location -i hdf5)/cmake/
export AMS_NLOHMANN_JSON_DIR=$(spack location -i nlohmann-json)/share/cmake/nlohmann_json/
export AMS_FMT_DIR=$(spack location -i fmt)/lib64/cmake/fmt/
export AMS_TL_EXPECTED_DIR=$(spack location -i tl-expected)/share/cmake/tl-expected/
export AMS_MFEM_PATH=$(spack location -i mfem)


export CMAKE_PREFIX_PATH=$(spack location -i protobuf)/lib64/cmake/protobuf/:${CMAKE_PREFIX_PATH}
export LD_LIBRARY_PATH=$(spack location -i flux-core)/lib:$LD_LIBRARY_PATH

exec "$@"