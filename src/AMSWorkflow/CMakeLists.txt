find_package(Python COMPONENTS Interpreter REQUIRED)

add_subdirectory(ams_wf)
add_subdirectory(ams)

configure_file("setup.py" "${CMAKE_CURRENT_BINARY_DIR}/setup.py" COPYONLY)
configure_file("${CMAKE_HOME_DIRECTORY}/pyproject.toml" "${CMAKE_BINARY_DIR}/pyproject.toml" COPYONLY)

file(GLOB_RECURSE pyfiles *.py ams_wf/*.py ams/*.py)

# detect virtualenv and set Pip args accordingly
set(AMS_PY_APP "${CMAKE_BINARY_DIR}")
set(_ams_py_install_target "${AMS_PY_APP}")
if(AMS_INSTALL_FLUX_PYTHON)
  set(_ams_py_install_target "${AMS_PY_APP}[flux]")
endif()
if(DEFINED ENV{VIRTUAL_ENV} OR DEFINED ENV{CONDA_PREFIX})
  set(_pip_args)
else()
  set(_pip_args "--user")
endif()
if(AMS_PIP_INSTALL_ARGS)
  separate_arguments(_ams_extra_pip_args NATIVE_COMMAND "${AMS_PIP_INSTALL_ARGS}")
  list(APPEND _pip_args ${_ams_extra_pip_args})
endif()

message(STATUS "AMS Python Source files are ${pyfiles}")
message(STATUS "AMS Python built cmd is : ${Python_EXECUTABLE} -m pip install ${_pip_args} ${_ams_py_install_target}")

add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/timestamp"
    COMMAND ${CMAKE_COMMAND} -E touch "${CMAKE_CURRENT_BINARY_DIR}/timestamp"
    COMMAND ${Python_EXECUTABLE} -m pip install ${_pip_args} "${_ams_py_install_target}"
    DEPENDS ${pyfiles}
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
    COMMENT "Build AMS-WF Python Modules and Applications"
    BYPRODUCTS "${CMAKE_CURRENT_BINARY_DIR}/timestamp"
    )

add_custom_target(PyAMS ALL
  DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/timestamp"
  )
