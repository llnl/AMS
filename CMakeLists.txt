# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# AMSLib Project Developers
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.18)


if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

cmake_policy(SET CMP0074 NEW)

project(AMS VERSION 0.1.1 LANGUAGES CXX C)


# NOTE: This may break some of our integrations with the applications. But flux requires > C++20, RMQ requires C++17 and although AMS does not have 
# any restrictions on the CXX standard the application may impose those. The solution would be to compile RMQ with an older GCC version  (8) and 
# have flux to be an external of the system environment
set(CMAKE_CXX_STANDARD 17)

# Enable this to get compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT EXISTS ${CMAKE_BINARY_DIR}/CMakeCache.txt)
  if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
  endif()
endif()

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake;${CMAKE_MODULE_PATH};")

set(AMS_APP_LIBRARIES "")
set(AMS_APP_DEFINES "")
set(AMS_APP_INCLUDES "")

set(AMS_EXAMPLE_LIBRARIES "")
set(AMS_EXAMPLE_DEFINES "")
set(AMS_EXAMPLE_INCLUDES "")

# ------------------------------------------------------------------------------
option(WITH_CUDA           "Option to enable CUDA" OFF)
option(WITH_HIP            "Option to enable HIP" OFF)
option(WITH_MPI            "Option to enable MPI" OFF)
option(WITH_CALIPER        "Use Caliper for Profiling" OFF)
option(WITH_TESTS          "Compile tests" OFF)
option(WITH_RMQ            "Use RabbitMQ as a database back end (require a reachable and running RabbitMQ server service)" OFF)
option(WITH_AMS_DEBUG      "Enable verbose messages" OFF)
option(WITH_PERFFLOWASPECT "Use PerfFlowAspect for Profiling" OFF)
option(WITH_WORKFLOW       "Install python drivers used by the outer workflow" OFF)
option(WITH_AMS_LIB        "Install C++ library to support scientific applications" ON)

# ------------------------------------------------------------------------------
find_package(nlohmann_json REQUIRED)
list(APPEND AMS_APP_LIBRARIES nlohmann_json::nlohmann_json)

# ------------------------------------------------------------------------------
find_package(Threads REQUIRED)
# ------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
include(cmake/FetchAndAddFmt.cmake)
FetchAndAddFmt("On")
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
include(cmake/FetchAndAddTlExpected.cmake)
FetchAndAddTlExpected("On")
#-------------------------------------------------------------------------------


set(MPI_DIRECTORIES "")
if (WITH_MPI)
  find_package(MPI REQUIRED)
  message(STATUS "MPICC:  ${MPI_C_COMPILER}")
  message(STATUS "MPICXX: ${MPI_CXX_COMPILER}")
  message(STATUS "MPI Library used: " MPI::MPI_CXX)
  foreach(LIBRARY ${MPI_C_LIBRARIES})
    get_filename_component(LIB_DIR ${LIBRARY} DIRECTORY)
    list(APPEND MPI_DIRECTORIES ${LIB_DIR})
  endforeach()
  list(REMOVE_DUPLICATES MPI_DIRECTORIES)
  message(STATUS "MPI Liraries: ${MPI_DIRECTORIES}")
  list(APPEND AMS_APP_DEFINES "__AMS_ENABLE_MPI__")
endif()

if(WITH_CUDA AND WITH_HIP)
  message(FATAL "AMS Does not support compilation with both CUDA and HIP enabled.")
elseif (WITH_CUDA)
  find_package(CUDAToolkit)
  list(APPEND AMS_APP_DEFINES "__AMS_ENABLE_CUDA__")
elseif (WITH_HIP)
  find_package(HIP REQUIRED)
  list(APPEND AMS_APP_DEFINES "__AMS_ENABLE_HIP__")
  list(APPEND CMAKE_CXX_FLAGS "-I${ROCM_PATH}/include/")
endif()

# ------------------------------------------------------------------------------
if (WITH_CALIPER)
  find_package(caliper REQUIRED)
  message(STATUS "Caliper Directory is ${caliper_DIR}")
  list(APPEND AMS_APP_DEFINES "__AMS_ENABLE_CALIPER__")
endif()

if (WITH_AMS_DEBUG)
  list(APPEND AMS_APP_DEFINES "LIBAMS_VERBOSE")
  list(APPEND AMS_APP_DEFINES "__AMS_DEBUG__")
endif()

if (AMS_HDF5_DIR)
       if (HDF5_USE_STATIC_LIBRARIES)
         find_package(HDF5 COMPONENTS C static NO_DEFAULT_PATH PATHS ${AMS_HDF5_DIR} ${AMS_HDF5_DIR}/share/cmake)
         set(AMS_HDF5_TARGET hdf5-static)
         set(AMS_HDF5_LIB_TYPE "static")
       else()
           find_package(HDF5 COMPONENTS C shared NO_DEFAULT_PATH PATHS ${AMS_HDF5_DIR} ${AMS_HDF5_DIR}/share/cmake)
           set(AMS_HDF5_LIB_TYPE "shared")
           set(AMS_HDF5_TARGET hdf5-shared)
       endif()
else()
       find_package(HDF5 REQUIRED COMPONENTS C CXX)
       set(AMS_HDF5_TARGET HDF5::HDF5)
endif()
message(STATUS "HDF5 Dir is ${HDF5_FOUND}")
list(APPEND AMS_APP_DEFINES "__AMS_ENABLE_HDF5__")

if (WITH_RMQ)
  if (WITH_CUDA)
    add_compile_definitions(THRUST_IGNORE_CUB_VERSION_CHECK)
  endif()
  list(APPEND AMS_APP_DEFINES "__AMS_ENABLE_RMQ__")

  find_package(amqpcpp REQUIRED)
  get_target_property(amqpcpp_INCLUDE_DIR amqpcpp INTERFACE_INCLUDE_DIRECTORIES)
  list(APPEND AMS_APP_INCLUDES ${amqpcpp_INCLUDE_DIR})

  find_package(OpenSSL REQUIRED)
  set(AMS_OPENSSL_FOUND_ROOT "")
  if (OPENSSL_FOUND)
    message(STATUS "OpenSSL libraries found: ${OPENSSL_LIBRARIES}")
    message(STATUS "OpenSSL includes found: " ${OPENSSL_INCLUDE_DIR})
    get_filename_component(AMS_OPENSSL_FOUND_ROOT "${OPENSSL_SSL_LIBRARY}" DIRECTORY)
    get_filename_component(AMS_OPENSSL_FOUND_ROOT "${AMS_OPENSSL_FOUND_ROOT}" DIRECTORY)
    message(STATUS "OPENSSL Root dir is ${AMS_OPENSSL_FOUND_ROOT}")
  else()
    message(STATUS "OpenSSL Not Found")
  endif()
  set(AMS_LIBEVENT_HINTS ${MPI_DIRECTORIES})
  
  find_package(libevent REQUIRED) # event loop library
endif() # WITH_RMQ

# ------------------------------------------------------------------------------
find_package(Torch REQUIRED)
# This is annoying, torch populates all my cuda flags
# and resets them
set(CMAKE_CUDA_FLAGS "")
set(CMAKE_CUDA_ARCHITECTURES ON)


# ------------------------------------------------------------------------------
if (WITH_RZ)
  find_package(MPI REQUIRED)
  add_subdirectory(rz)
  list(APPEND AMS_APP_INCLUDES "${RZ_AMS_INCLUDES}" "${MPI_INCLUDE_PATH}")
  list(APPEND AMS_APP_LIB_DIRS "${RZ_AMS_LIBDIRS}")
  list(APPEND AMS_APP_LIBRARIES "${RZ_AMS_LIBRARIES}" "${MPI_C_LIBRARIES}")
  list(APPEND AMS_APP_DEFINES "${RZ_AMS_DEFINES}")
endif()

if (WITH_PERFFLOWASPECT)
  find_package(perfflowaspect CONFIG REQUIRED)

  list(APPEND AMS_APP_DEFINES "__AMS_ENABLE_PERFFLOWASPECT__")
  list(APPEND AMS_APP_LIB_DIRS "${PERFFLOWASPECT_LIB_DIR}")

  list(APPEND AMS_APP_LIBRARIES "perfflow_runtime")
endif()


add_subdirectory(src)

if (WITH_TESTS)
  include(CTest)
  add_subdirectory(tests)
endif()
