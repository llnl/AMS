#!/usr/bin/env bash

# Need to use modules
source /etc/profile.d/z00_lmod.sh

host=$(hostname)
host=${host//[0-9]/}

#!/usr/bin/env bash

## load relevant modules on tuo/tioga
host=$(hostname)
host=${host//[0-9]/}

SPACK_VER="1.0"

if [[ "$SYS_TYPE" == "toss_4_x86_64_ib_cray" ]]; then
    echo "Loading MPI and ROCm"
    module load gcc/13.3.1-magic
    module load rocm/6.4.1
    module load cray-mpich/8.1.32
    ROCM_ARCH=$(rocm_agent_enumerator | sed -n 1p)
    export CMAKE_CXX_FLAGS="-I${ROCM_PATH}/include/"
elif [[ "$SYS_TYPE" == "toss_4_x86_64_ib" ]]; then
    # Dane
    module load cmake/3.30.5
    module load gcc/13.3.1-magic
    module load mvapich2/2.3.7
fi

## activate spack
echo "Activating Spack"
source /usr/workspace/AMS/ams-spack-environments/${SPACK_VER}/spack/share/spack/setup-env.sh

## activate the spack environment
echo "Activating Spack Environment"
spack env activate /usr/workspace/AMS/ams-spack-environments/${SPACK_VER}/$host

## export the paths (currently cmake needs these)
export AMS_TORCH_PATH=`spack location -i py-torch`
export AMS_HDF5_PATH=`spack location -i hdf5`
export AMS_CALIPER_PATH=`spack location -i caliper`
export AMS_AMQPCPP_PATH=`spack location -i amqp-cpp`
export AMS_ADIAK_PATH=`spack location -i adiak`
export AMS_CUDA_ARCH=${CUDA_ARCH}
export AMS_HIP_ARCH=${ROCM_ARCH}

if [[ "$SYS_TYPE" == "toss_4_x86_64_ib_cray" ]]; then
    # echo "AMS_CUDA_ARCH                = $AMS_CUDA_ARCH"
    echo "AMS_HIP_ARCH                 = $AMS_HIP_ARCH"
fi
echo "AMS_TORCH_PATH               = $AMS_TORCH_PATH"
echo "AMS_HDF5_PATH                = $AMS_HDF5_PATH"
echo "AMS_CALIPER_PATH             = $AMS_CALIPER_PATH"
echo "AMS_AMQPCPP_PATH             = $AMS_AMQPCPP_PATH"
echo "AMS_ADIAK_PATH               = $AMS_ADIAK_PATH"

export AMS_TORCH_PATH=$(echo $AMS_TORCH_PATH/lib/python3.*/site-packages/torch/share/cmake/Torch)
export AMS_AMQPCPP_PATH=$(echo $AMS_AMQPCPP_PATH/cmake)
export AMS_CALIPER_PATH=$(echo $AMS_CALIPER_PATH/share/cmake/caliper)

echo "(for cmake) AMS_TORCH_PATH   = $AMS_TORCH_PATH"
echo "(for cmake) AMS_AMQPCPP_PATH = $AMS_AMQPCPP_PATH"
echo "(for cmake) AMS_CALIPER_PATH = $AMS_CALIPER_PATH"

export AMS_LOG_LEVEL=Debug
