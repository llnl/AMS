# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# AMSLib Project Developers
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(AMS_DOWNSTREAM_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/consumer_mock")
set(AMS_DOWNSTREAM_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/consumer_mock-build")
set(AMS_DOWNSTREAM_PACKAGE_DIR "${PROJECT_BINARY_DIR}/src/AMSlib")

set(AMS_DOWNSTREAM_CONFIGURE_ARGS
    -DAMS_DIR:PATH=${AMS_DOWNSTREAM_PACKAGE_DIR}
    -DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}
    -DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}
)

if (CMAKE_BUILD_TYPE)
  list(APPEND AMS_DOWNSTREAM_CONFIGURE_ARGS
       -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE})
endif()

foreach (_hint fmt_DIR Torch_DIR HDF5_DIR ZLIB_DIR caliper_DIR
                amqpcpp_DIR OpenSSL_ROOT_DIR CUDAToolkit_DIR HIP_DIR)
  if (DEFINED ${_hint} AND NOT "${${_hint}}" STREQUAL "")
    list(APPEND AMS_DOWNSTREAM_CONFIGURE_ARGS
         -D${_hint}:PATH=${${_hint}})
  endif()
endforeach()

add_test(NAME AMS_DOWNSTREAM_CONSUMER_CONFIGURE
         COMMAND ${CMAKE_COMMAND}
                 -S ${AMS_DOWNSTREAM_SOURCE_DIR}
                 -B ${AMS_DOWNSTREAM_BINARY_DIR}
                 ${AMS_DOWNSTREAM_CONFIGURE_ARGS})
set_tests_properties(AMS_DOWNSTREAM_CONSUMER_CONFIGURE
                     PROPERTIES
                     FIXTURES_SETUP AMSDownstream
                     LABELS DOWNSTREAM)

add_test(NAME AMS_DOWNSTREAM_CONSUMER_BUILD
         COMMAND ${CMAKE_COMMAND}
                 --build ${AMS_DOWNSTREAM_BINARY_DIR}
                 --config $<CONFIG>)
set_tests_properties(AMS_DOWNSTREAM_CONSUMER_BUILD
                     PROPERTIES
                     FIXTURES_REQUIRED AMSDownstream
                     LABELS DOWNSTREAM)
