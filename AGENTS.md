# AGENTS.md

This file provides guidance to LLMs (Claude Code, Codex etc) when working with code in this repository.

## Project Overview

AMS (Autonomous MultiScale Library) is a library to simplify machine learning surrogate model integration in HPC codes.
It enables scientific applications to use ML models as surrogates for expensive physics computations with uncertainty quantification.

**Key components:**
- **AMSLib (C++)**: Core library providing the AMS API for scientific applications
- **AMSWorkflow (Python)**: Workflow orchestration components (AMSBroker, AMSTrain, AMSDeploy, AMSStore, AMSOrchestrator, AMSDBStage)
- **ML Integration**: PyTorch-based surrogate models with uncertainty quantification
- **Data Management**: HDF5 and optional RabbitMQ backends for storing/retrieving training data


## Executable Commands
- **Load Dependencies on Livermore Computing**: `source scripts/gitlab/setup-env.sh`
- **Test**: `ctest --test-dir build --output-on-failure`
- **Lint**: `clang-tidy -p build src/**/*.cpp`
- **Format**: `find src/ -regex '.*\.\(cpp\|hpp\|cu\|cuh\|c\|h\)' -exec clang-format -i {} \;`

## Spack

Before running Spack commands or sourcing `scripts/gitlab/setup-env.sh`, keep
Spack and XDG caches out of the home directory:

```bash
export SPACK_DISABLE_LOCAL_CONFIG=true
export SPACK_SKIP_MODULES=1
export SPACK_USER_CACHE_PATH=/tmp
export XDG_CACHE_HOME=/tmp
```

## Build System

AMS uses [BLT](https://github.com/llnl/blt) to build.

### CMake Configuration

Standard build on Dane or on machine **without** GPU:
```bash
mkdir build && cd build
cmake \
  -DENABLE_HIP=Off \
  -DENABLE_CALIPER=On \
  -Dcaliper_DIR=$AMS_CALIPER_PATH \
  -DTorch_DIR=$AMS_TORCH_PATH \
  -DENABLE_MPI=On \
  -DHDF5_DIR="$AMS_HDF5_PATH" \
  -DENABLE_RMQ=On \
  -Damqpcpp_DIR=$AMS_AMQPCPP_PATH \
  -DENABLE_TESTS=On \
  -DAMS_CATCH2_DIR="$AMS_CATCH2_DIR" \
  -DENABLE_WORKFLOW=On \
  -DAMS_ENABLE_DEBUG=On \
  -DAMS_FMT_DIR="$AMS_FMT_DIR" \
  -Dnlohmann_json_DIR="$AMS_NLOHMANN_JSON_DIR" \
  -Dtl-expected_DIR="$AMS_TL_EXPECTED_DIR" \
  ..
make -j6
make install
```

Standard build on Tioga/Tuo or on machine with AMD GPUs:

```bash
export SPACK_DISABLE_LOCAL_CONFIG=true
export SPACK_SKIP_MODULES=1
export SPACK_USER_CACHE_PATH=/tmp
export XDG_CACHE_HOME=/tmp
source scripts/gitlab/setup-env.sh

cmake \
  -DBUILD_SHARED_LIBS=On \
  -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=On \
  -DCMAKE_C_COMPILER=amdclang \
  -DCMAKE_CXX_COMPILER=amdclang++ \
  -DENABLE_HIP=On \
  -DENABLE_CALIPER=On \
  -Dcaliper_DIR=$AMS_CALIPER_PATH \
  -DTorch_DIR=$AMS_TORCH_PATH \
  -DENABLE_MPI=On \
  -DHDF5_DIR="$AMS_HDF5_PATH" \
  -DENABLE_RMQ=Off \
  -DENABLE_TESTS=Off \
  -DENABLE_WORKFLOW=Off \
  -DAMS_ENABLE_DEBUG=On \
  -DAMS_FMT_DIR="$AMS_FMT_DIR" \
  -Dnlohmann_json_DIR="$AMS_NLOHMANN_JSON_DIR" \
  -Dtl-expected_DIR="$AMS_TL_EXPECTED_DIR" \
  ..
```

If you want to build on a system with NVIDIA GPU you can just use `-DENABLE_CUDA=On` and `-DCMAKE_CUDA_ARCHITECTURES="$AMS_CUDA_ARCH"`.

Make sure to set `-DBUILD_SHARED_LIBS=On -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=On` to both `On` (dynamic library) or both `Off` (static library).

The most minimal build will need:
```
  -DTorch_DIR=$AMS_TORCH_PATH \
  -DHDF5_DIR="$AMS_HDF5_PATH" \
  -DAMS_FMT_DIR="$AMS_FMT_DIR" \
  -Dnlohmann_json_DIR="$AMS_NLOHMANN_JSON_DIR" \
  -Dtl-expected_DIR="$AMS_TL_EXPECTED_DIR" \
```

Test builds also need `-DAMS_CATCH2_DIR="$AMS_CATCH2_DIR"` when using the LC
Spack-provided Catch2 package.

For some builds you might need to specify the correct Zlib with `-DZLIB_DIR="$AMS_ZLIB_PATH"` (if `AMS_ZLIB_PATH` is defined).

#### Compilers

On Tuolumne and Tioga machine from LC, you must use `amdclang` as your compiler.
Make sure to add to the CMake command line `-DCMAKE_C_COMPILER="amdclang"   -DCMAKE_CXX_COMPILER="amdclang++"`.

Unless specified otherwise you can use `gcc` and let CMake discover the correct compiler.

### CMake Options

Required dependencies:
- HDF5, Torch, nlohmann_json, fmt, tl-expected, Threads, and a C++17 compiler.

Optional features:
- `ENABLE_MPI`: Enable MPI support
- `ENABLE_CUDA` / `ENABLE_HIP`: GPU acceleration (mutually exclusive)
- `ENABLE_CALIPER`: Caliper profiling support
- `ENABLE_PERFFLOWASPECT`: PerfFlowAspect profiling (requires PFA-enabled clang/llvm)
- `ENABLE_RMQ`: RabbitMQ backend for distributed data management
- `AMS_ENABLE_DEBUG`: Enable verbose debug output (defines `LIBAMS_VERBOSE` and `__AMS_DEBUG__`)
- `ENABLE_TESTS`: Build test suite (uses Catch2)
- `ENABLE_WORKFLOW`: Install Python workflow drivers

## Running Tests

```bash
cd build
make test
# or for detailed output:
ctest --output-on-failure
# or
CTEST_OUTPUT_ON_FAILURE=1 make test
# or to run a specific test
ctest --output-on-failure -R "testName"
```

Tests use Catch2 framework (v3.11.0). On LC systems,
`scripts/gitlab/setup-env.sh` exports `AMS_CATCH2_DIR` for the Spack-provided
Catch2 package. Pass `-DAMS_CATCH2_DIR="$AMS_CATCH2_DIR"` with
`-DENABLE_TESTS=On` to avoid network access.

If Catch2 package discovery fails and no `AMS_CATCH2_DIR` hint is provided,
CMake falls back to `FetchContent` from GitHub. In network-free environments,
provide a local/package Catch2 config directory with `AMS_CATCH2_DIR` or
configure with `-DENABLE_TESTS=Off`.

Test directory structure:
- `tests/AMSlib/ams_interface/`: End-to-end AMS interface tests
- `tests/AMSlib/db/`: Database backend tests (HDF5)
- `tests/AMSlib/torch/`: PyTorch model inference tests
- `tests/AMSlib/wf/`: Workflow component tests
- `tests/AMSlib/models/`: Test model generation scripts

## Code Architecture

### Core AMS API (`src/AMSlib/`)

Main API is defined in `src/AMSlib/include/AMS.h`:

1. **Initialization**: `AMSInit()` / `AMSFinalize()` - Setup and teardown
2. **Model Registration**: `AMSRegisterAbstractModel()` - Register a surrogate model with domain name, threshold, and model path
3. **Executor Creation**: `AMSCreateExecutor()` - Create an executor for a registered model
4. **Execution**: `AMSExecute()` / `AMSCExecute()` - Execute with surrogate model or physics fallback
5. **Cleanup**: `AMSDestroyExecutor()` - Destroy executor

**Key concepts:**
- **Uncertainty Quantification**: Models return `Tuple[[Tensor[N, ...], Tensor[N, 1]]` where second tensor contains uncertainty scores (lower = more confident)
- **Threshold**: Controls when to use surrogate vs physics (based on uncertainty)
- **Hybrid Execution**: Automatically falls back to physics computation when uncertainty exceeds threshold

### Workflow System (`src/AMSlib/wf/`)

The `AMSWorkflow` class orchestrates hybrid execution:

- **Evaluation Pipeline**: 
  1. Predict using surrogate model
  2. Check uncertainty against threshold
  3. For high-uncertainty samples: execute physics and store data
  4. For low-uncertainty samples: use ML predictions

- **Model Updates**: Supports dynamic model updates via RabbitMQ
- **Data Storage**: Stores training data to HDF5 or RabbitMQ backends
- **Distributed Execution**: MPI-aware for parallel processing

Key files:
- `src/AMSlib/wf/workflow.hpp`: Main workflow class
- `src/AMSlib/wf/action.hpp`: Action concept for data transformations
- `src/AMSlib/wf/eval_context.hpp`: Evaluation context management
- `src/AMSlib/wf/basedb.hpp`: Database backend interface

### ML Components (`src/AMSlib/ml/`)

- `surrogate.hpp`: Surrogate model wrapper around PyTorch models
- `Model.hpp`: PyTorch model loading and inference
- `AbstractModel.hpp`: Abstract interface for ML models

### Python Workflow (`src/AMSWorkflow/`)

Components for outer training/deployment loop:
- `AMSBroker`: Message broker for distributed coordination
- `AMSTrain`: Training orchestration
- `AMSDeploy`: Model deployment
- `AMSStore`: Data storage management
- `AMSOrchestrator`: Workflow orchestration
- `AMSDBStage`: Database staging

Install with: `pip install -e .` from project root

## Code Style
- **Standards**: C++17, strictly. Prefer standard library over external dependencies where possible.
- **Ownership**: Use smart pointers or value semantics. NO raw `new`/`delete`.
- **Safety**: Use `tl::expected` for error handling; avoid raw exceptions in performance-critical paths.
- **Headers**: Prefer `#pragma once` over traditional include guards.
- **Formatting**: Strictly follow the project's `.clang-format`. Run it after every file modification.
- **Memory leaks**: Test the code with Valgrind if you suspect memroy leaks

Format Python code with:
```bash
ruff format <file>
```

## Development Workflow

**Main branch**: `develop` (not `main`)

**Creating PRs**: Always target `develop` as the base branch.

**Python requirements**: Tests require `h5py` installed (`pip install h5py`)

## Installation

Recommended: Use Spack for dependency management:
```bash
spack install ams
# or for development:
spack dev-build ams
```

See INSTALL.md for manual installation details.

## Repository Structure

```
src/
├── AMSlib/           # C++ library
│   ├── include/      # Public API headers
│   ├── ml/           # ML model components
│   └── wf/           # Workflow system
└── AMSWorkflow/      # Python workflow tools
    ├── ams/          # Python package
    └── ams_wf/       # Workflow drivers

tests/
├── AMSlib/           # C++ tests (Catch2)
└── AMSWorkflow/      # Python tests

examples/
├── ideal_gas/        # Example: ideal gas law application
└── bnm_opt/          # Example: optimization application

cmake/                # CMake modules
docs/                 # Sphinx documentation
```

## Common Patterns

**Type aliases in AMS.h:**
- `AMSExecutor`: Executor handle (int64_t)
- `AMSCAbstrModel`: Model handle (int)
- `DomainLambda`: C++ lambda callback type
- `DomainCFn`: C function pointer callback type

**Device support:**
- AMS uses custom resource manager for memory management across CPU/GPU
- Set allocator: `AMSSetAllocator(AMSResourceType resource, const char* name)`
- Supported resources: Host, Device (CUDA/HIP)

**Database configuration:**
- File system DB: `AMSConfigureFSDatabase(AMSDBType db_type, const char* db_path)`
- RabbitMQ DB: Enable with `-DENABLE_RMQ=On` at build time

## Boundaries & Guardrails
- **Always**: Run tests that are impacted by your changes. For example, to re-run the
  core tests: `ctest --output-on-failure -R "CORE::"` or `ctest --output-on-failure -R "CORE::TENSOR_INT"`
  to re-run one specific test.
- **Always**: Run `./scripts/run-code-quality.sh --staged --clang-format --ruff --fix` before testing your changes
- **Ask First**: Before adding new external dependencies to `CMakeLists.txt`.
- **Never**: Use C-style casts; instead use `static_cast` or `reinterpret_cast`.
- **Never**: Over-engineer solutions with superfluous safety checking.
- **Never**: Use modifying git commands unless explicitly asked to by the user.
- **Never**:  Run all the tests with `ctest` unless explicitly asked to by the user or before
  commiting to a branch.
