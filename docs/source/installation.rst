Installation
============

AMS is built with CMake and C++17. The main build task is making sure CMake can
find the required packages.

Requirements
------------

Core dependencies:

* CMake >= 3.18
* C++17 compatible compiler
* HDF5
* LibTorch
* nlohmann_json
* fmt
* tl-expected

Optional dependencies:

* CUDA for NVIDIA GPU support
* HIP / ROCm for AMD GPU support
* MPI for distributed execution
* Caliper for profiling
* amqp-cpp, OpenSSL, and libevent for RabbitMQ support
* Catch2 for tests

Current CMake Options
---------------------

Use the current ``ENABLE_*`` options:

.. list-table::
   :header-rows: 1

   * - Option
     - Purpose
   * - ``ENABLE_MPI``
     - Enable MPI support.
   * - ``ENABLE_CUDA``
     - Enable CUDA support.
   * - ``ENABLE_HIP``
     - Enable HIP / ROCm support.
   * - ``ENABLE_CALIPER``
     - Enable Caliper profiling.
   * - ``ENABLE_RMQ``
     - Enable RabbitMQ database support.
   * - ``ENABLE_WORKFLOW``
     - Install Python workflow drivers.
   * - ``ENABLE_TESTS``
     - Build Catch2-based tests.
   * - ``AMS_ENABLE_DEBUG``
     - Enable verbose AMS debug messages.
   * - ``AMS_INSTALL_FLUX_PYTHON``
     - Install the Python workflow package with the ``flux-python`` optional dependency when ``ENABLE_WORKFLOW=On``.
   * - ``AMS_PIP_INSTALL_ARGS``
     - Extra arguments passed to ``pip install`` when ``ENABLE_WORKFLOW=On``.

``ENABLE_CUDA`` and ``ENABLE_HIP`` are mutually exclusive.

Dependency Hints
----------------

When packages are outside default CMake search paths, pass explicit hints:

.. list-table::
   :header-rows: 1

   * - Package
     - CMake variable
     - LC setup export
   * - LibTorch
     - ``Torch_DIR``
     - ``$AMS_TORCH_PATH``
   * - HDF5
     - ``HDF5_DIR``
     - ``$AMS_HDF5_PATH``
   * - Caliper
     - ``caliper_DIR``
     - ``$AMS_CALIPER_PATH``
   * - amqp-cpp
     - ``amqpcpp_DIR``
     - ``$AMS_AMQPCPP_PATH``
   * - nlohmann_json
     - ``nlohmann_json_DIR``
     - ``$AMS_NLOHMANN_JSON_DIR``
   * - fmt
     - ``AMS_FMT_DIR``
     - ``$AMS_FMT_DIR``
   * - tl-expected
     - ``tl-expected_DIR``
     - ``$AMS_TL_EXPECTED_DIR``
   * - Catch2, when ``ENABLE_TESTS=On``
     - ``AMS_CATCH2_DIR``
     - ``$AMS_CATCH2_DIR``

LC HIP / ROCm Build
-------------------

On Tuolumne, Tioga, and similar LC ROCm systems, keep Spack caches out of the
home directory, source the LC setup script, use ``amdclang``/``amdclang++``,
and pass the dependency exports:

.. code-block:: bash

   export SPACK_DISABLE_LOCAL_CONFIG=true
   export SPACK_SKIP_MODULES=1
   export SPACK_USER_CACHE_PATH=/tmp
   export XDG_CACHE_HOME=/tmp
   source scripts/gitlab/setup-env.sh

   cmake -S . -B build \
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

   cmake --build build -j
   ctest --test-dir build -N

The validated HIP configuration used ``ENABLE_TESTS=Off`` and CTest reported
``Total Tests: 0``. To build Catch2 tests on LC, switch to
``-DENABLE_TESTS=On`` and add ``-DAMS_CATCH2_DIR="$AMS_CATCH2_DIR"``.

LC Workflow Python Environments
-------------------------------

On LC systems, do not create AMS Workflow environments with plain
``python3 -m venv``. The AMS Spack environments use a Python external, and LC
Flux provides the compatible ``flux`` Python bindings for the active Flux
installation. Create the venv through the repository helper so it links both
Spack Python packages and a venv-local shim for system Flux Python:

.. code-block:: bash

   host=$(hostname)
   host=${host//[0-9]/}
   python3 scripts/make-spack-venv.py \
      --env "/usr/workspace/AMS/ams-spack-environments/1.1/${host}/" \
      --output "venv-${host}" \
      --with-system-flux-python
   source "venv-${host}/bin/activate"

The helper writes system Flux metadata into the venv and warns on activation if the
active ``flux version`` differs from the recorded one. The shim exposes Flux
without putting unrelated LC Python packages ahead of AMS Spack packages.
Recreate the venv after system Flux changes. For LC workflow CMake builds, pass
``-DAMS_PIP_INSTALL_ARGS="--no-build-isolation"`` so pip uses the prepared venv
instead of an isolated build environment. Do not enable
``AMS_INSTALL_FLUX_PYTHON`` for this path; Flux Python is supplied by
the prepared system-backed venv, not by pip.

Non-System Flux Workflow Installs
---------------------------------

When building AMS Workflow in an environment that does not provide compatible
system Flux Python bindings, enable the workflow Flux extra:

.. code-block:: bash

   cmake -S . -B build \
      -DENABLE_WORKFLOW=On \
      -DAMS_INSTALL_FLUX_PYTHON=On

With this flag, the ``PyAMS`` target runs pip against the generated package tree
as ``pip install <build-dir>[flux]``, which installs the ``flux-python``
optional dependency declared by ``pyproject.toml``. Use this for container or
non-LC builds that rely on pip-managed Flux Python. Leave it off when using
``scripts/make-spack-venv.py --with-system-flux-python``.

Convenience Script
------------------

``scripts/ams-configure.sh`` can assemble the common CMake command:

.. code-block:: bash

   scripts/ams-configure.sh --hip --mpi --caliper
   scripts/ams-configure.sh --mpi --tests
   scripts/ams-configure.sh --workflow --install-flux-python
   scripts/ams-configure.sh --mpi --rmq --dry-run

When ``--tests`` is enabled and ``AMS_CATCH2_DIR`` is set, the helper forwards
``-DAMS_CATCH2_DIR="$AMS_CATCH2_DIR"`` so LC builds use the Spack-provided
Catch2 package.

Manual CMake is still needed when forcing LC Cray/ROCm compilers unless the
script is later extended with compiler options.

Tests
-----

``ENABLE_TESTS=On`` enters the Catch2 test tree and first looks for a Catch2
CMake package. On LC systems, ``scripts/gitlab/setup-env.sh`` exports
``AMS_CATCH2_DIR`` for the Spack-provided Catch2 package; pass it with
``-DAMS_CATCH2_DIR="$AMS_CATCH2_DIR"``.

If package discovery fails and no ``AMS_CATCH2_DIR`` hint is provided, CMake
falls back to ``FetchContent`` from GitHub for Catch2 v3.11.0. In network-free
environments outside LC, provide a local/package Catch2 config directory with
``AMS_CATCH2_DIR`` or configure with ``-DENABLE_TESTS=Off``.

For more examples, see the repository ``INSTALL.md``.
