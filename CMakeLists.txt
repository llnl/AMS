# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# AMSLib Project Developers
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.18)

if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

project(AMS VERSION 0.1.1 LANGUAGES CXX C)

get_filename_component(AMS_CXX_COMPILER_NAME "${CMAKE_CXX_COMPILER}" NAME)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND AMS_CXX_COMPILER_NAME STREQUAL "gcc")
  message(WARNING "CMAKE_CXX_COMPILER points to gcc; forcing libstdc++ on C++ link lines")
  link_libraries("$<$<LINK_LANGUAGE:CXX>:stdc++>")
endif()

# NOTE: This may break some of our integrations with the applications. But flux requires > C++20, RMQ requires C++17 and although AMS does not have
# any restrictions on the CXX standard the application may impose those. The solution would be to compile RMQ with an older GCC version  (8) and
# have flux to be an external of the system environment
set(CMAKE_CXX_STANDARD 17)

# Enable this to get compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT EXISTS "${CMAKE_BINARY_DIR}/CMakeCache.txt")
  if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
  endif()
endif()

option(ENABLE_MPI "Enable MPI support" OFF)
option(ENABLE_CUDA "Enable CUDA support" OFF)
option(ENABLE_HIP "Enable HIP support" OFF)
option(ENABLE_CALIPER "Use Caliper for profiling" OFF)
option(ENABLE_TESTS "Build AMS tests" OFF)
option(ENABLE_WORKFLOW "Install python drivers used by the outer workflow" OFF)
option(ENABLE_RMQ "Use RabbitMQ as a database back end" OFF)
option(ENABLE_PERFFLOWASPECT "Use PerfFlowAspect for profiling" OFF)
option(AMS_ENABLE_DEBUG "Enable verbose AMS messages" OFF)
option(AMS_DEFER_STATIC_TPL_RESOLUTION
       "Defer selected static TPL resolution to downstream final links when building shared AMS"
       OFF)

set(AMS_THIRDPARTY_DIR "${PROJECT_SOURCE_DIR}/thirdparty" CACHE PATH
    "Path to AMS-local third-party dependencies")
set(AMS_FMT_DIR "" CACHE PATH
    "Path to a directory containing fmt-config.cmake or fmtConfig.cmake")
set(HDF5_WITH_INTEL_RUNTIME "" CACHE FILEPATH
    "Optional Intel runtime library to link when using static HDF5")

if (ENABLE_CUDA AND ENABLE_HIP)
  message(FATAL_ERROR "AMS does not support compilation with both CUDA and HIP enabled.")
endif()

if (NOT BLT_SOURCE_DIR)
  set(BLT_SOURCE_DIR "${PROJECT_SOURCE_DIR}/blt" CACHE PATH "Path to a BLT checkout")
endif()

if (NOT EXISTS "${BLT_SOURCE_DIR}/SetupBLT.cmake")
  message(FATAL_ERROR
          "BLT was not found. Set BLT_SOURCE_DIR to a BLT checkout or provide a local "
          "'${PROJECT_SOURCE_DIR}/blt' checkout for standalone builds.")
endif()

list(PREPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" "${BLT_SOURCE_DIR}/cmake")
include("${BLT_SOURCE_DIR}/SetupBLT.cmake")

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

set(AMS_APP_LIBRARIES "")
set(AMS_APP_DEFINES "")
set(AMS_APP_INCLUDES "")

set(AMS_EXAMPLE_LIBRARIES "")
set(AMS_EXAMPLE_DEFINES "")
set(AMS_EXAMPLE_INCLUDES "")

set(AMS_PACKAGE_DEPENDENCY_BLOCK "")

function(ams_append_package_dependency line)
  set(AMS_PACKAGE_DEPENDENCY_BLOCK
      "${AMS_PACKAGE_DEPENDENCY_BLOCK}${line}\n"
      PARENT_SCOPE)
endfunction()

function(inherit_target_nostatic)
  set(options)
  set(singleValueArgs TO FROM OBJECT)
  set(multiValueArgs)
  cmake_parse_arguments(arg "${options}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN})

  if (NOT DEFINED arg_TO)
    message(FATAL_ERROR "Must provide a TO argument to inherit_target_nostatic")
  endif()

  if (NOT DEFINED arg_FROM)
    message(FATAL_ERROR "Must provide a FROM argument to inherit_target_nostatic")
  endif()

  set(_scope INTERFACE)

  get_target_property(_interface_system_includes ${arg_FROM} INTERFACE_SYSTEM_INCLUDE_DIRECTORIES)
  if (_interface_system_includes)
    target_include_directories(${arg_TO} SYSTEM ${_scope} ${_interface_system_includes})
  endif()

  get_target_property(_interface_includes ${arg_FROM} INTERFACE_INCLUDE_DIRECTORIES)
  if (_interface_includes)
    target_include_directories(${arg_TO} ${_scope} ${_interface_includes})
  endif()

  get_target_property(_interface_defines ${arg_FROM} INTERFACE_COMPILE_DEFINITIONS)
  if (_interface_defines)
    target_compile_definitions(${arg_TO} ${_scope} ${_interface_defines})
  endif()

  if (${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.13.0")
    get_target_property(_interface_link_options ${arg_FROM} INTERFACE_LINK_OPTIONS)
    if (_interface_link_options)
      target_link_options(${arg_TO} ${_scope} ${_interface_link_options})
    endif()
  endif()

  get_target_property(_interface_compile_options ${arg_FROM} INTERFACE_COMPILE_OPTIONS)
  if (_interface_compile_options)
    target_compile_options(${arg_TO} ${_scope} ${_interface_compile_options})
  endif()
endfunction()

function(remove_static_from_libs)
  set(options)
  set(singleValueArgs TO)
  set(multiValueArgs FROM)
  cmake_parse_arguments(arg "${options}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN})

  if (NOT DEFINED arg_TO)
    message(FATAL_ERROR "Must provide a TO argument to remove_static_from_libs")
  endif()

  if (NOT DEFINED arg_FROM)
    message(FATAL_ERROR "Must provide a FROM argument to remove_static_from_libs")
  endif()

  if (BUILD_SHARED_LIBS AND AMS_DEFER_STATIC_TPL_RESOLUTION)
    set(retval)
    foreach(THIS_LIB ${arg_FROM})
      if (TARGET ${THIS_LIB})
        get_target_property(target_type ${THIS_LIB} TYPE)
        if (target_type STREQUAL STATIC_LIBRARY)
          add_library("${THIS_LIB}::nostatic" INTERFACE IMPORTED)
          inherit_target_nostatic(TO "${THIS_LIB}::nostatic" FROM ${THIS_LIB})
          list(APPEND retval "${THIS_LIB}::nostatic")
        else()
          list(APPEND retval ${THIS_LIB})
        endif()
      else()
        get_filename_component(THIS_EXT ${THIS_LIB} EXT)
        if (NOT ".a" STREQUAL "${THIS_EXT}")
          list(APPEND retval ${THIS_LIB})
        endif()
      endif()
    endforeach()
  else()
    set(retval ${arg_FROM})
  endif()

  set(${arg_TO} ${retval} PARENT_SCOPE)
endfunction()

if (AMS_DEFER_STATIC_TPL_RESOLUTION AND NOT BUILD_SHARED_LIBS)
  message(STATUS
          "AMS_DEFER_STATIC_TPL_RESOLUTION is ignored when BUILD_SHARED_LIBS=OFF.")
endif()

if (ENABLE_RMQ)
  find_package(Threads REQUIRED)
endif()

include(cmake/FetchAndAddFmt.cmake)
set(_ams_fmt_args)
if (AMS_FMT_DIR)
  list(APPEND _ams_fmt_args PATHS "${AMS_FMT_DIR}" NO_DEFAULT_PATH)
endif()
find_package(fmt CONFIG QUIET ${_ams_fmt_args})
if (NOT TARGET fmt::fmt)
  message(STATUS "fmt not found via CMake package discovery; falling back to FetchContent")
  FetchAndAddFmt(ON)
endif()

find_package(nlohmann_json CONFIG REQUIRED)

include(cmake/FetchAndAddTlExpected.cmake)
find_package(tl-expected QUIET)
if (NOT TARGET tl::expected)
  message(STATUS "tl::expected not found via CMake package discovery; falling back to FetchContent")
  FetchAndAddTlExpected(ON)
endif()
if (NOT TARGET tl::expected)
  message(FATAL_ERROR
          "Unable to find tl::expected. Provide a tl-expected package or allow "
          "FetchContent to populate it.")
endif()

set(MPI_DIRECTORIES "")
if (ENABLE_MPI)
  find_package(MPI REQUIRED COMPONENTS C CXX)
  message(STATUS "MPICC:  ${MPI_C_COMPILER}")
  message(STATUS "MPICXX: ${MPI_CXX_COMPILER}")
  message(STATUS "MPI Library used: MPI::MPI_CXX")
  foreach(LIBRARY ${MPI_C_LIBRARIES})
    get_filename_component(LIB_DIR ${LIBRARY} DIRECTORY)
    list(APPEND MPI_DIRECTORIES ${LIB_DIR})
  endforeach()
  list(REMOVE_DUPLICATES MPI_DIRECTORIES)
  message(STATUS "MPI Libraries: ${MPI_DIRECTORIES}")
  list(APPEND AMS_APP_DEFINES "__AMS_ENABLE_MPI__")
endif()

if (ENABLE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  list(APPEND AMS_APP_DEFINES "__AMS_ENABLE_CUDA__")
elseif (ENABLE_HIP)
  find_package(HIP REQUIRED)
  list(APPEND AMS_APP_DEFINES "__AMS_ENABLE_HIP__")
  if (DEFINED ROCM_PATH)
    list(APPEND CMAKE_CXX_FLAGS "-I${ROCM_PATH}/include/")
  endif()
endif()

if (ENABLE_CALIPER)
  find_package(caliper REQUIRED)
  message(STATUS "Caliper directory is ${caliper_DIR}")
  list(APPEND AMS_APP_DEFINES "__AMS_ENABLE_CALIPER__")
endif()

if (AMS_ENABLE_DEBUG)
  list(APPEND AMS_APP_DEFINES "LIBAMS_VERBOSE" "__AMS_DEBUG__")
endif()

set(AMS_HDF5_MODE "")
set(AMS_HDF5_LINK_TARGETS "")

if (DEFINED HDF5_DIR AND NOT "${HDF5_DIR}" STREQUAL "")
  # Try old packaging system.
  if (HDF5_USE_STATIC_LIBRARIES)
    set(_mode static)
  else()
    set(_mode shared)
  endif()
  find_package(hdf5 CONFIG QUIET
               COMPONENTS C ${_mode}
               NO_DEFAULT_PATH
               HINTS ${HDF5_DIR}/lib/cmake/hdf5
                     ${HDF5_DIR}/lib64/cmake/hdf5
                     ${HDF5_DIR}/share/cmake
                     ${HDF5_DIR})
  if (TARGET "hdf5-${_mode}")
    set(AMS_HDF5_LINK_TARGETS "hdf5-${_mode}")
    if (HDF5_USE_STATIC_LIBRARIES)
      find_package(ZLIB REQUIRED)
      list(APPEND AMS_HDF5_LINK_TARGETS ZLIB::ZLIB)
      if (HDF5_WITH_INTEL_RUNTIME)
        list(APPEND AMS_HDF5_LINK_TARGETS "${HDF5_WITH_INTEL_RUNTIME}")
      endif()
      set(AMS_HDF5_MODE "HDF5_STATIC_TARGET")
    else()
      set(AMS_HDF5_MODE "HDF5_SHARED_TARGET")
    endif()
  endif()
endif()

if (NOT AMS_HDF5_MODE)
  # Try new packaing system.
  find_package(HDF5 CONFIG QUIET COMPONENTS C)
  if (NOT HDF5_FOUND)
    find_package(HDF5 REQUIRED COMPONENTS C)
  endif()

  if (TARGET HDF5::HDF5)
    set(AMS_HDF5_LINK_TARGETS HDF5::HDF5)
  elseif (TARGET hdf5::hdf5)
    set(AMS_HDF5_LINK_TARGETS hdf5::hdf5)
  elseif (TARGET HDF5::C)
    set(AMS_HDF5_LINK_TARGETS HDF5::C)
  else()
    message(FATAL_ERROR
            "Unable to determine imported HDF5 targets for AMS. Provide either "
            "an HDF5 package with hdf5-static or hdf5-shared, or a modern HDF5 package with imported targets.")
  endif()

  set(AMS_HDF5_MODE "HDF5_STANDARD_TARGET")
endif()

message(STATUS "AMS HDF5 mode is ${AMS_HDF5_MODE}")
list(APPEND AMS_APP_DEFINES "__AMS_ENABLE_HDF5__")

if (ENABLE_RMQ)
  if (ENABLE_CUDA)
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
    message(STATUS "OpenSSL includes found: ${OPENSSL_INCLUDE_DIR}")
    get_filename_component(AMS_OPENSSL_FOUND_ROOT "${OPENSSL_SSL_LIBRARY}" DIRECTORY)
    get_filename_component(AMS_OPENSSL_FOUND_ROOT "${AMS_OPENSSL_FOUND_ROOT}" DIRECTORY)
    message(STATUS "OpenSSL root dir is ${AMS_OPENSSL_FOUND_ROOT}")
  else()
    message(STATUS "OpenSSL not found")
  endif()

  set(AMS_LIBEVENT_HINTS ${MPI_DIRECTORIES})
  find_package(libevent REQUIRED)
endif()

find_package(Torch REQUIRED)
# This is annoying, torch populates all my cuda flags
# and resets them
set(CMAKE_CUDA_FLAGS "")
set(CMAKE_CUDA_ARCHITECTURES ON)

if (ENABLE_PERFFLOWASPECT)
  find_package(perfflowaspect CONFIG REQUIRED)
  list(APPEND AMS_APP_DEFINES "__AMS_ENABLE_PERFFLOWASPECT__")
  list(APPEND AMS_APP_LIB_DIRS "${PERFFLOWASPECT_LIB_DIR}")
  list(APPEND AMS_APP_LIBRARIES "perfflow_runtime")
endif()

if (NOT BUILD_SHARED_LIBS)
  ams_append_package_dependency("find_dependency(fmt CONFIG REQUIRED)")
  ams_append_package_dependency("find_dependency(Torch REQUIRED)")
  if (ENABLE_MPI)
    ams_append_package_dependency("find_dependency(MPI REQUIRED COMPONENTS C CXX)")
  endif()
  if (ENABLE_CUDA)
    ams_append_package_dependency("find_dependency(CUDAToolkit REQUIRED)")
  endif()
  if (ENABLE_HIP)
    ams_append_package_dependency("find_dependency(HIP REQUIRED)")
  endif()
  if (ENABLE_CALIPER)
    ams_append_package_dependency("find_dependency(caliper REQUIRED)")
  endif()
  if (AMS_HDF5_MODE STREQUAL "HDF5_STATIC_TARGET")
    ams_append_package_dependency("find_dependency(hdf5 CONFIG REQUIRED COMPONENTS C static)")
    ams_append_package_dependency("find_dependency(ZLIB REQUIRED)")
  elseif (AMS_HDF5_MODE STREQUAL "HDF5_SHARED_TARGET")
    ams_append_package_dependency("find_dependency(hdf5 CONFIG REQUIRED COMPONENTS C shared)")
  else()
    ams_append_package_dependency("find_dependency(HDF5 REQUIRED COMPONENTS C)")
  endif()
  if (ENABLE_RMQ)
    ams_append_package_dependency("find_dependency(Threads REQUIRED)")
    ams_append_package_dependency("find_dependency(amqpcpp REQUIRED)")
    ams_append_package_dependency("find_dependency(OpenSSL REQUIRED)")
  endif()
  if (ENABLE_PERFFLOWASPECT)
    ams_append_package_dependency("find_dependency(perfflowaspect CONFIG REQUIRED)")
  endif()
endif()

add_subdirectory(src)

if (ENABLE_TESTS)
  include(CTest)
  add_subdirectory(tests)
endif()
