# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# AMSLib Project Developers
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.21)

project(AMSConsumer LANGUAGES CXX)

find_package(AMS REQUIRED)

add_executable(ams_consumer main.cpp)
target_compile_features(ams_consumer PRIVATE cxx_std_17)
target_link_libraries(ams_consumer PRIVATE AMS::AMS)
