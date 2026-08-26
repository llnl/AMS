# Providing AMS dependencies on a non-LC cluster

Off LLNL Livermore Computing systems there is no shared AMS Spack environment,
so install dependencies yourself and point CMake at them.

## Spack

If your site has Spack and an `ams` package in a reachable repo:

```bash
export SPACK_DISABLE_LOCAL_CONFIG=true
export SPACK_SKIP_MODULES=1
export SPACK_USER_CACHE_PATH=/tmp
export XDG_CACHE_HOME=/tmp

spack install ams
spack load ams
```

For active development, use `spack dev-build ams` from a working copy. If your
Spack does not have an `ams` package, install the dependencies individually:

```bash
spack install nlohmann-json hdf5 py-torch fmt tl-expected
spack install caliper                        # if ENABLE_CALIPER
spack install mpi                            # if ENABLE_MPI, or use site MPI
spack install amqp-cpp openssl libevent      # if ENABLE_RMQ
spack install catch2                         # if ENABLE_TESTS
```

Then locate each install and pass its CMake hint.

## Module system

Many clusters expose some dependencies as modules:

```bash
module load cmake gcc hdf5 cuda openmpi
```

LibTorch, nlohmann_json, fmt, and tl-expected are less commonly available as
modules. Use Spack or manual installs for anything the module stack does not
provide.

## Manual dependencies and hints

| Dependency | Required? | CMake hint |
| --- | --- | --- |
| `nlohmann_json` | yes | `nlohmann_json_DIR=<pfx>/lib/cmake/nlohmann_json` |
| `fmt` | yes | `AMS_FMT_DIR=<pfx>/lib/cmake/fmt` or `fmt_DIR=<pfx>/lib/cmake/fmt` |
| `tl::expected` | yes | `tl-expected_DIR=<pfx>/share/cmake/tl-expected` |
| HDF5 | yes | `HDF5_DIR=<hdf5 root or cmake package dir>` |
| libTorch | yes | `Torch_DIR=<libtorch>/share/cmake/Torch` |
| MPI | if `ENABLE_MPI` | compiler wrappers or `CMAKE_PREFIX_PATH` |
| CUDA | if `ENABLE_CUDA` | CUDA toolkit on path plus `CMAKE_CUDA_ARCHITECTURES=<sm_xx>` |
| ROCm/HIP | if `ENABLE_HIP` | `ROCM_PATH`, `hip_DIR`, or `CMAKE_PREFIX_PATH` |
| Caliper | if `ENABLE_CALIPER` | `caliper_DIR=<pfx>/share/cmake/caliper` |
| amqp-cpp | if `ENABLE_RMQ` | `amqpcpp_DIR=<pfx>/cmake` |
| OpenSSL | if `ENABLE_RMQ` | `OPENSSL_ROOT_DIR=<pfx>` or `CMAKE_PREFIX_PATH` |
| libevent | if `ENABLE_RMQ` | `CMAKE_PREFIX_PATH` |
| Catch2 | if `ENABLE_TESTS` | `AMS_CATCH2_DIR=<pfx>/lib/cmake/Catch2` or package config dir |
| PerfFlowAspect | if `ENABLE_PERFFLOWASPECT` | `perfflowaspect_DIR=<pfx>/share` |
| Zlib | if static HDF5 needs it | `ZLIB_ROOT=<pfx>` or `ZLIB_DIR=<pfx>` |

Put hand-built installs under one prefix when possible:

```bash
export CMAKE_PREFIX_PATH=/opt/ams-deps:$CMAKE_PREFIX_PATH
```

Then pass explicit `*_DIR` hints only for packages CMake still cannot find.
LibTorch usually needs `Torch_DIR` because its CMake package is nested.

## Network-free builds

Configure with local package hints for fmt and tl-expected. Otherwise AMS may
try `FetchContent` from GitHub for those packages.

`ENABLE_TESTS=On` first tries Catch2 package discovery. For network-free test
builds, provide `AMS_CATCH2_DIR=<pfx>/lib/cmake/Catch2` or another directory
containing `Catch2Config.cmake` or `catch2-config.cmake`. If no package hint is
provided and discovery fails, AMS falls back to `FetchContent` from GitHub for
Catch2 v3.11.0; use `-DENABLE_TESTS=Off` when no local Catch2 package is
available.

## Sanity check

Configure a minimal CPU build first:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=On \
  -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=On \
  -DTorch_DIR=/opt/libtorch/share/cmake/Torch \
  -DHDF5_DIR=/opt/hdf5 \
  -DAMS_FMT_DIR=/opt/ams-deps/lib/cmake/fmt \
  -Dnlohmann_json_DIR=/opt/ams-deps/lib/cmake/nlohmann_json \
  -Dtl-expected_DIR=/opt/ams-deps/share/cmake/tl-expected
```

If that succeeds, add MPI, GPU, RabbitMQ, and profiling flags one at a time.
