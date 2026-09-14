# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# AMSLib Project Developers
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.24)
cmake_policy(SET CMP0104 NEW)

# Define the project
project(AMSExamples LANGUAGES CXX)


option(WITH_AMS_EXAMPLE_CUDA  "Option to enable CUDA" Off)
option(WITH_AMS_EXAMPLE_HIP "Option to enable HIP" Off)
option(WITH_AMS_DOWNLOAD_DATA "Download benchmark datasets at configure time" OFF)

if(WITH_AMS_EXAMPLE_CUDA AND WITH_AMS_EXAMPLE_HIP)
  message(FATAL_ERROR
    "WITH_AMS_EXAMPLE_CUDA=${WITH_AMS_EXAMPLE_CUDA} and "
    "WITH_AMS_EXAMPLE_HIP=${WITH_AMS_EXAMPLE_HIP} are mutually exclusive."
  )
endif()


set(AMS_INPUT_DIR "${CMAKE_BINARY_DIR}/inputs")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (WITH_AMS_EXAMPLE_CUDA)
message(STATUS "HERE")
  set(THREADS_PREFER_PTHREAD_FLAG ON)
  find_package(Threads REQUIRED)
  enable_language(CUDA)
  find_package(CUDAToolkit)
elseif (WITH_AMS_EXAMPLE_HIP)
  enable_language(HIP)
  find_package(hip REQUIRED)
endif()

if (WITH_AMS_DOWNLOAD_DATA)
enable_testing()
add_subdirectory(inputs)
endif()


add_subdirectory(blackscholes)
