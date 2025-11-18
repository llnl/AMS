# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# AMSLib Project Developers
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(FetchContent)
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.11.0        # <- pick a version you like
)
FetchContent_MakeAvailable(Catch2)

# Optional: bring in Catch2's CTest adapter (catch_discover_tests)
# (The module lives under Catch2/extras)
list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/extras)
include(Catch)

set(AMS_TEST_ROOT "${CMAKE_CURRENT_BINARY_DIR}")
message(INFO "AMS_TEST_ROOT is ${AMS_TEST_ROOT}")
add_subdirectory(models)
add_subdirectory(torch)
add_subdirectory(db)
add_subdirectory(wf)
add_subdirectory(ams_interface)
#add_subdirectory(perf_regression)

