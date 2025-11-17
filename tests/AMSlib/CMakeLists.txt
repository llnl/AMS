# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# AMSLib Project Developers
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(${CMAKE_SOURCE_DIR}/cmake/FetchAndAddCatch.cmake)
FetchAndAddCatch(ON)

set(AMS_TEST_ROOT "${CMAKE_CURRENT_BINARY_DIR}")
message(INFO "AMS_TEST_ROOT is ${AMS_TEST_ROOT}")
add_subdirectory(models)
add_subdirectory(torch)
add_subdirectory(db)
add_subdirectory(wf)
add_subdirectory(ams_interface)
#add_subdirectory(perf_regression)
