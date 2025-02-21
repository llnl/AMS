# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# AMSLib Project Developers
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

find_package(Python3 REQUIRED COMPONENTS Interpreter)
message(STATUS "Python executable: ${Python3_EXECUTABLE}")

# Check if h5py is installed
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import h5py"
    RESULT_VARIABLE H5PY_CHECK_RESULT
    OUTPUT_QUIET
    ERROR_QUIET
)

if(NOT H5PY_CHECK_RESULT EQUAL 0)
    message(FATAL_ERROR
        "Missing Python dependency: h5py\n"
        "Try running:\n"
        "    pip install h5py\n"
        "Or ensure it is available in your environment."
    )
endif()

if (WITH_AMS_LIB)
  add_subdirectory(AMSlib)
endif()

if (WITH_WORKFLOW)
  add_subdirectory(AMSWorkflow)
endif()
