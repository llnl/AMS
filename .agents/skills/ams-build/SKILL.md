---
name: ams-build
description: >-
  Build and install the AMS library (github.com/LLNL/AMS) from source on an HPC
  cluster via CMake. Use this whenever the user wants to compile, configure,
  build, or install AMS, or is choosing between build variants (with/without
  RabbitMQ, GPU, MPI, Caliper, etc.). ALWAYS use this skill for AMS build/CMake
  questions even when phrased loosely ("get AMS running on Tioga", "why can't
  CMake find Torch", "AMS build with RabbitMQ"). It knows how to gather AMS's
  dependencies two ways: via LLNL Livermore Computing's internal Spack
  environment or by pointing CMake at manually provided libraries.
---

# Installing AMS from source

AMS is a C++17 library built with CMake. The build is usually simple once the
dependency hints are correct.

## Before Spack or LC setup

In restricted environments, set these before any Spack command or before
sourcing `scripts/gitlab/setup-env.sh`:

```bash
export SPACK_DISABLE_LOCAL_CONFIG=true
export SPACK_SKIP_MODULES=1
export SPACK_USER_CACHE_PATH=/tmp
export XDG_CACHE_HOME=/tmp
```

## Dependency provisioning path

Run this check before writing a CMake command:

```bash
if [[ -d /usr/workspace/AMS/ams-spack-environments ]]; then
  echo "LC cluster: use scripts/gitlab/setup-env.sh"
else
  echo "Non-LC: provide dependencies manually (references/manual-deps.md)"
fi
```

On an LLNL Livermore Computing cluster, source the repo setup script. It loads
modules, activates the AMS Spack environment, and exports `AMS_TORCH_PATH`,
`AMS_HDF5_PATH`, `AMS_CALIPER_PATH`, `AMS_AMQPCPP_PATH`,
`AMS_NLOHMANN_JSON_DIR`, `AMS_FMT_DIR`, `AMS_TL_EXPECTED_DIR`,
`AMS_CATCH2_DIR`, and GPU arch variables.

On any other cluster, install dependencies yourself and pass package hints or
`CMAKE_PREFIX_PATH`; see `references/manual-deps.md`.

`$SYS_TYPE` is a secondary LC signal: `toss_4_x86_64_ib` is Dane/CTS-1, while
`toss_4_x86_64_ib_cray` is Tuolumne/Tioga/El Capitan-class ROCm.

## LC Python virtual environments

On LC machines, do not create workflow Python environments with plain
`python3 -m venv myenv`. The AMS Spack environments use a Python external, and
a normal venv can miss Spack-provided Python packages. Use the repository
helper so the venv is based on the Spack Python and can see the Python packages
from the active AMS Spack environment.

Run this from the AMS repository root after the Spack cache exports and
`scripts/gitlab/setup-env.sh`:

```bash
host=$(hostname)
host=${host//[0-9]/}
python3 scripts/make-spack-venv.py \
  --env "/usr/workspace/AMS/ams-spack-environments/1.1/${host}/" \
  --output "venv-${host}" \
  --with-system-flux-python
source "venv-${host}/bin/activate"
```

Use this venv before installing or running Python workflow pieces, for example
with `pip install -e .` or with builds that enable `-DENABLE_WORKFLOW=On`.
The helper links the active system Flux Python bindings through a venv-local shim,
records `flux version`, `which flux`, the Python version, `flux.__file__`, and
the shim path, and warns on activation when the active Flux version differs.
Recreate the venv after system Flux changes.

For LC workflow CMake builds, pass this so pip uses the prepared venv instead
of an isolated build environment:

```bash
-DAMS_PIP_INSTALL_ARGS="--no-build-isolation"
```

Leave `AMS_INSTALL_FLUX_PYTHON=Off` for this path. The prepared venv supplies
system Flux Python. For container or non-LC builds that need pip-managed Flux
Python, enable `-DAMS_INSTALL_FLUX_PYTHON=On` with `-DENABLE_WORKFLOW=On`; the
workflow install target uses the `flux-python` optional dependency.

## Current dependencies and options

Always required: HDF5, libTorch, nlohmann_json, fmt, tl-expected, Threads, and
a C++17 compiler. `fmt` and `tl-expected` can fall back to `FetchContent`, so
network-free builds should pass local package hints.

Current CMake options:

| Option | Purpose |
| --- | --- |
| `ENABLE_MPI` | Enable MPI support. |
| `ENABLE_CUDA` | Enable CUDA support. |
| `ENABLE_HIP` | Enable HIP / ROCm support. |
| `ENABLE_CALIPER` | Enable Caliper profiling. |
| `ENABLE_RMQ` | Enable RabbitMQ database support. |
| `ENABLE_PERFFLOWASPECT` | Enable PerfFlowAspect profiling. |
| `ENABLE_WORKFLOW` | Install Python workflow drivers. |
| `ENABLE_TESTS` | Build Catch2 tests. |
| `AMS_ENABLE_DEBUG` | Enable verbose debug messages. |
| `AMS_INSTALL_FLUX_PYTHON` | Install the Python workflow package with `flux-python` when `ENABLE_WORKFLOW=On`. |
| `AMS_PIP_INSTALL_ARGS` | Extra arguments passed to `pip install` when `ENABLE_WORKFLOW=On`. |

`ENABLE_CUDA` and `ENABLE_HIP` are mutually exclusive.

Important CMake hints:

| Package | CMake hint | LC export |
| --- | --- | --- |
| Torch | `Torch_DIR` | `$AMS_TORCH_PATH` |
| HDF5 | `HDF5_DIR` | `$AMS_HDF5_PATH` |
| Caliper | `caliper_DIR` | `$AMS_CALIPER_PATH` |
| amqp-cpp | `amqpcpp_DIR` | `$AMS_AMQPCPP_PATH` |
| nlohmann_json | `nlohmann_json_DIR` | `$AMS_NLOHMANN_JSON_DIR` |
| fmt | `AMS_FMT_DIR` | `$AMS_FMT_DIR` |
| tl-expected | `tl-expected_DIR` | `$AMS_TL_EXPECTED_DIR` |
| Catch2, when `ENABLE_TESTS=On` | `AMS_CATCH2_DIR` | `$AMS_CATCH2_DIR` |
| CUDA arch | `CMAKE_CUDA_ARCHITECTURES` | `$AMS_CUDA_ARCH` |
| Zlib, if needed | `ZLIB_ROOT` or `ZLIB_DIR` | `$AMS_ZLIB_PATH` |

## Common configure shapes

### Minimal CPU on LC

```bash
export SPACK_DISABLE_LOCAL_CONFIG=true
export SPACK_SKIP_MODULES=1
export SPACK_USER_CACHE_PATH=/tmp
export XDG_CACHE_HOME=/tmp
source scripts/gitlab/setup-env.sh

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=On \
  -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=On \
  -DTorch_DIR="$AMS_TORCH_PATH" \
  -DHDF5_DIR="$AMS_HDF5_PATH" \
  -DAMS_FMT_DIR="$AMS_FMT_DIR" \
  -Dnlohmann_json_DIR="$AMS_NLOHMANN_JSON_DIR" \
  -Dtl-expected_DIR="$AMS_TL_EXPECTED_DIR"
```

### Validated HIP / ROCm path on Tuolumne

Use `amdclang` and `amdclang++` on LC Cray/ROCm systems. The validated
network-free smoke build used `ENABLE_TESTS=Off`; CTest discovery then reports
no tests. To build the Catch2 tests on LC, switch to `ENABLE_TESTS=On` and pass
`-DAMS_CATCH2_DIR="$AMS_CATCH2_DIR"` from `scripts/gitlab/setup-env.sh`.

```bash
export SPACK_DISABLE_LOCAL_CONFIG=true
export SPACK_SKIP_MODULES=1
export SPACK_USER_CACHE_PATH=/tmp
export XDG_CACHE_HOME=/tmp
source scripts/gitlab/setup-env.sh

cmake -S . -B codex-build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=On \
  -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=On \
  -DCMAKE_C_COMPILER=amdclang \
  -DCMAKE_CXX_COMPILER=amdclang++ \
  -DENABLE_HIP=On \
  -DENABLE_MPI=On \
  -DENABLE_CALIPER=On \
  -DENABLE_RMQ=Off \
  -DENABLE_WORKFLOW=Off \
  -DENABLE_TESTS=Off \
  -DAMS_ENABLE_DEBUG=On \
  -DTorch_DIR="$AMS_TORCH_PATH" \
  -DHDF5_DIR="$AMS_HDF5_PATH" \
  -Dcaliper_DIR="$AMS_CALIPER_PATH" \
  -DAMS_FMT_DIR="$AMS_FMT_DIR" \
  -Dnlohmann_json_DIR="$AMS_NLOHMANN_JSON_DIR" \
  -Dtl-expected_DIR="$AMS_TL_EXPECTED_DIR"

cmake --build codex-build -j
ctest --test-dir codex-build -N
```

Expected CTest discovery for that configuration is `Total Tests: 0`.

### RabbitMQ variant

Add `-DENABLE_RMQ=On` and pass `-Damqpcpp_DIR="$AMS_AMQPCPP_PATH"` on LC or the
manual amqp-cpp hint off LC. RabbitMQ support compiles the AMQP client; it does
not start or provision a broker.

## Helper script

`scripts/ams-configure.sh` assembles the common CMake line and maps LC
`AMS_*` exports to current hints:

```bash
scripts/ams-configure.sh
scripts/ams-configure.sh --mpi --rmq
scripts/ams-configure.sh --hip --mpi --caliper
scripts/ams-configure.sh --mpi --tests
scripts/ams-configure.sh --workflow --install-flux-python
scripts/ams-configure.sh --mpi --rmq --dry-run
```

When `--tests` is enabled and `AMS_CATCH2_DIR` is set, the helper forwards the
LC Spack-provided Catch2 package with `-DAMS_CATCH2_DIR="$AMS_CATCH2_DIR"`.

Manual CMake is still needed when forcing LC Cray/ROCm compilers unless the
script is later extended with compiler options.

## Build and install

```bash
cmake --build build -j "$(nproc)"
cmake --install build
```

## Common failure modes

- **`Could NOT find Torch` / `HDF5` / `nlohmann_json` / `fmt` /
  `tl-expected`**: on LC, source `scripts/gitlab/setup-env.sh` after setting
  the Spack cache variables above. Off LC, provide the corresponding package
  hint or add the install prefix to `CMAKE_PREFIX_PATH`.
- **`Could NOT find amqpcpp` / `libevent`**: only appears with
  `-DENABLE_RMQ=On`; provide `amqpcpp_DIR` and libevent/OpenSSL locations.
- **Both CUDA and HIP set**: CMake hard-errors; choose one.
- **Catch2 configure tries GitHub**: `ENABLE_TESTS=On` first tries Catch2
  package discovery. On LC, source `scripts/gitlab/setup-env.sh` and pass
  `-DAMS_CATCH2_DIR="$AMS_CATCH2_DIR"` to use the Spack-provided Catch2
  package. If no package hint is provided and discovery fails, CMake falls back
  to `FetchContent` for Catch2 v3.11.0; in network-free environments, provide a
  local/package Catch2 config directory or configure with
  `-DENABLE_TESTS=Off`.
- **LC setup emits GitHub clone warnings**: if the cache variables are set, the
  warning can be non-fatal when the AMS environment still exports all package
  paths. Do not request network access unless the user explicitly asks.

## Files in this skill

- `references/manual-deps.md` - how to obtain each dependency on a non-LC
  cluster and which CMake variable points at it.
