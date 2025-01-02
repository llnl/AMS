# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# AMSLib Project Developers
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


cmake_minimum_required(VERSION 3.10)
cmake_policy(SET CMP0104 NEW)

# Define the project
project(BOptions LANGUAGES CXX)

set(AMS_EXAMPLE_SRC binomial_options.cpp kernel.cpp)


option(WITH_MPI            "Option to enable MPI" OFF)
option(WITH_CALIPER        "Use Caliper for Profiling" OFF)

set(CMAKE_CXX_STANDARD 14)      # Enable C++14
set(CMAKE_CXX_STANDARD_REQUIRED ON) # Require the specified standard

enable_language(CUDA)

find_package(AMS REQUIRED)
find_package(CUDAToolkit REQUIRED)

if (WITH_CALIPER)
  find_package(caliper REQUIRED)
endif()

if (WITH_MPI)
  find_package(MPI REQUIRED)
endif()

function(ADDExec binary_name use_ams)
  target_link_libraries(${binary_name} PRIVATE ${MFEM_LIBRARIES})
  target_include_directories(${binary_name} PRIVATE ${MFEM_INCLUDE_DIRS})
  # we always use ams to avoid if def conflicts.
  # but we only enable it for the use_ams case
  target_link_libraries(${binary_name} PRIVATE AMS::AMS)
  if (${use_ams})
    target_compile_definitions(${binary_name} PRIVATE USE_AMS)
  endif()

  if (WITH_MPI)
    target_link_libraries(${binary_name} PRIVATE MPI::MPI_CXX)
    target_compile_definitions(${binary_name} PRIVATE "__ENABLE_MPI__")
  endif()


  target_link_libraries(${binary_name} PRIVATE 
    CUDA::cudart
    CUDA::cublas
    CUDA::cusparse
    CUDA::curand 
  )
  set_source_files_properties(kernel.cpp PROPERTIES LANGUAGE CUDA)
  set_target_properties(${binary_name} PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
  target_compile_definitions(${binary_name} PRIVATE "__ENABLE_CUDA__")
endfunction()

add_executable(no_ams_boptions ${AMS_EXAMPLE_SRC} ${MINIAPP_INCLUDES})
ADDExec(no_ams_boptions FALSE)

add_executable(ams_boptions ${AMS_EXAMPLE_SRC} ${MINIAPP_INCLUDES})
ADDExec(ams_boptions TRUE)


