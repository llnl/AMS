# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# AMSLib Project Developers
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.10)
cmake_policy(SET CMP0104 NEW)

# Define the project
project(IdealGasExample LANGUAGES CXX)


option(WITH_CUDA           "Option to enable CUDA" On)
option(WITH_MPI            "Option to enable MPI" OFF)
option(WITH_CALIPER        "Use Caliper for Profiling" OFF)

set(CMAKE_CXX_STANDARD 14)      # Enable C++14
set(CMAKE_CXX_STANDARD_REQUIRED ON) # Require the specified standard

if (WITH_CUDA)
  message(WARNING "CUDA is ${CMAKE_CUDA_ARCHITECTURES}")
  enable_language(CUDA)
  find_package(CUDAToolkit)
  message(WARNING "CUDA is ${CMAKE_CUDA_ARCHITECTURES}")
endif()

if (MFEM_DIR)
  include(${PROJECT_SOURCE_DIR}/cmake/FindMFEM.cmake)
else()
  find_package(MFEM REQUIRED)
endif()


set(AMS_EXAMPLE_SRC ${MINIAPP_INCLUDES} main.cpp app/eos_ams.cpp)

if (WITH_CALIPER)
  find_package(caliper REQUIRED)
endif()

if (WITH_MPI)
  find_package(MPI REQUIRED)
endif()

find_package(AMS REQUIRED)

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


  if (WITH_CUDA)
    target_link_libraries(${binary_name} PRIVATE 
      CUDA::cudart          # CUDA Runtime
      CUDA::cublas          # cuBLAS
      CUDA::cusparse        # cuSPARSE
      CUDA::curand          # cuRAND
    )
    set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)
    set_source_files_properties(main.cpp PROPERTIES COMPILE_FLAGS "--expt-extended-lambda")
    set_source_files_properties(app/eos_ams.cpp PROPERTIES LANGUAGE CUDA)
    set_source_files_properties(app/eos_ams.cpp PROPERTIES COMPILE_FLAGS "--expt-extended-lambda")
    set_target_properties(${binary_name} PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
    target_compile_definitions(${binary_name} PRIVATE "__ENABLE_CUDA__")
  endif()
endfunction()

add_executable(no_ams_example ${AMS_EXAMPLE_SRC} ${MINIAPP_INCLUDES})
ADDExec(no_ams_example FALSE)

add_executable(ams_example ${AMS_EXAMPLE_SRC} ${MINIAPP_INCLUDES})
ADDExec(ams_example TRUE)

if (WITH_WORKFLOW)
  set(TRAIN_DEVICE "cpu")
  if (WITH_CUDA)
    set(TRAIN_DEVICE "gpu")
  endif()
  #  configure_file ("${CMAKE_CURRENT_SOURCE_DIR}/prune.py" "${CMAKE_CURRENT_BINARY_DIR}/prune.py")
  #  configure_file ("${CMAKE_CURRENT_SOURCE_DIR}/train.py" "${CMAKE_CURRENT_BINARY_DIR}/train.py")
  #  configure_file ("${CMAKE_CURRENT_SOURCE_DIR}/sub_selection.py" "${CMAKE_CURRENT_BINARY_DIR}/sub_selection.py")
  #  configure_file ("${CMAKE_CURRENT_SOURCE_DIR}/example_run.sh" "${CMAKE_CURRENT_BINARY_DIR}/example_run.sh" @ONLY)
endif()
