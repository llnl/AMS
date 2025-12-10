Installation
============

This page provides detailed instructions for installing AMS on various systems.

Requirements
------------

AMS has several dependencies that need to be installed before building:

Core Dependencies
~~~~~~~~~~~~~~~~~

* **CMake** >= 3.25
* **C++17** compatible compiler (GCC >= 8.5)
* **Python** >= 3.10 (for Python bindings)
* **PyTorch** >= 2.0 (for ML model support)
* **HDF5** (for data storage)

Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~

* **CUDA** >= 11.0 (for NVIDIA GPU support)
* **HIP** >= 6.4 (for AMD GPU support)
* **MPI** (for distributed computing)
* **RabbitMQ/AMQP-CPP** (for message queue support)
* **Caliper** (for performance profiling)

Installation Methods
--------------------

Using Spack
~~~~~~~~~~~

TBD

Manual Build with CMake
~~~~~~~~~~~~~~~~~~~~~~~

For a basic installation:

.. code-block:: bash

   git clone https://github.com/LLNL/AMS.git
   cd AMS
   mkdir build && cd build

   cmake \
      -DWITH_RMQ=On \
      -Damqpcpp_DIR=$AMS_AMQPCPP_PATH \
      -DWITH_CALIPER=On \
      -DWITH_HDF5=On \
      -DHDF5_Dir=$AMS_HDF5_PATH \
      -DCMAKE_INSTALL_PREFIX=./install \
      -DCMAKE_BUILD_TYPE=Release \
      -DWITH_CUDA=On \
      -DWITH_MPI=On \
      -DWITH_TESTS=On \
      -DTorch_DIR=$AMS_TORCH_PATH
      -DWITH_EXAMPLES=On \
      ..

   make -j 4
   make install

For complete installation instructions, see the repository's INSTALL.md file.
