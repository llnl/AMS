# Copyright 2022-2023 Lawrence Livermore National Security, LLC and other
# AMSLib Project Developers
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


cmake_minimum_required(VERSION 3.20)
cmake_policy(SET CMP0104 NEW)

# Define the project
project(EX4 LANGUAGES CXX C)

set(AMS_EXAMPLE_SRC ex4.cpp)
set(AMS_SOL_SRC sol4.cpp)

option(WITH_MPI            "Option to enable MPI" OFF)
option(WITH_CALIPER        "Use Caliper for Profiling" OFF)

set(CMAKE_CXX_STANDARD 14)      # Enable C++14
set(CMAKE_CXX_STANDARD_REQUIRED ON) # Require the specified standard

find_package(AMS REQUIRED)

if (WITH_CALIPER)
  find_package(caliper REQUIRED)
endif()

if (WITH_MPI)
  find_package(MPI REQUIRED)
endif()


add_executable(EX4 ${AMS_EXAMPLE_SRC})
target_link_libraries(EX4 PRIVATE AMS::AMS)

add_executable(SOL4 ${AMS_SOL_SRC})
target_link_libraries(SOL4 PRIVATE AMS::AMS)

