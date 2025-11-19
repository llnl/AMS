# Copyright (c) Lawrence Livermore National Security, LLC and other AMS
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute

# ------------------------------------------------------------------------------
# handle sources and headers
set(AMS_LIB_SRC wf/debug.cpp wf/logger.cpp wf/utils.cpp wf/SmallVector.cpp ml/surrogate.cpp wf/basedb.cpp AMSTensor.cpp wf/interface.cpp wf/resource_manager.cpp ml/Model.cpp ml/AbstractModel.cpp AMS.cpp)

  list(APPEND AMS_LIB_SRC wf/hdf5db.cpp)

if (WITH_RMQ)
  list(APPEND AMS_LIB_SRC wf/rmqdb.cpp)
endif()

add_library(AMS ${AMS_LIB_SRC})

get_target_property(AMS_TYPE AMS TYPE)

# ------------------------------------------------------------------------------
# setup the lib first
message(STATUS "ALL INCLUDES ARE ${AMS_APP_INCLUDES}")
target_compile_definitions(AMS PRIVATE ${AMS_APP_DEFINES})
message(STATUS "All AMS Internal Defines are ${AMS_APP_DEFINES}")
target_include_directories(AMS PRIVATE 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/>)
target_include_directories(AMS PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(AMS PUBLIC fmt::fmt tl::expected)

if (WITH_CUDA)
  target_link_libraries(AMS PRIVATE CUDA::cudart)
elseif (WITH_HIP)
  target_link_libraries(AMS PRIVATE hip::host)
endif()

target_link_libraries(AMS PUBLIC $<BUILD_INTERFACE:${AMS_HDF5_TARGET}> PRIVATE $<INSTALL_INTERFACE:${AMS_HDF5_TARGET}>)

if (WITH_CALIPER)
  target_link_libraries(AMS PUBLIC $<BUILD_INTERFACE:caliper> PRIVATE $<INSTALL_INTERFACE:caliper>)
endif()

if (WITH_MPI)
  target_link_libraries(AMS PUBLIC $<BUILD_INTERFACE:MPI::MPI_CXX> PRIVATE $<INSTALL_INTERFACE:MPI::MPI_CXX>)
endif()

if (WITH_RMQ)
  target_link_libraries(AMS PUBLIC  $<BUILD_INTERFACE:amqpcpp> PRIVATE $<INSTALL_INTERFACE:amqpcpp>)

  if (OPENSSL_FOUND)
    target_link_libraries(AMS PUBLIC
                               $<BUILD_INTERFACE:OpenSSL::SSL>
                               $<BUILD_INTERFACE:OpenSSL::Crypto>
                               PRIVATE
                               $<INSTALL_INTERFACE:OpenSSL::SSL>
                               $<INSTALL_INTERFACE:OpenSSL::Crypto>)
  endif()
  # NOTE: We set here the event/event pthreads as public. As there is no easy way
  # to do a find package(libevent) and RMQ is not exposing that properly.
  message(STATUS "Event libs are ${LIBEVENT_LIBRARY}")
  message(STATUS "Event libs are ${LIBEVENT_LIBRARY} and ${LIBEVENT_THREAD}")
  target_link_libraries(AMS PUBLIC ${LIBEVENT_LIBRARY} ${LIBEVENT_THREAD})
endif()

target_link_libraries(AMS PUBLIC
    		$<BUILD_INTERFACE:stdc++fs torch> PRIVATE
    		$<INSTALL_INTERFACE:stdc++fs torch>)


configure_file ("${CMAKE_CURRENT_SOURCE_DIR}/include/AMS-config.h.in" "${PROJECT_BINARY_DIR}/include/AMS-config.h")
configure_file ("${CMAKE_CURRENT_SOURCE_DIR}/include/AMS.h" "${PROJECT_BINARY_DIR}/include/AMS.h" COPYONLY)
configure_file ("${CMAKE_CURRENT_SOURCE_DIR}/include/AMSTypes.hpp" "${PROJECT_BINARY_DIR}/include/AMSTypes.hpp" COPYONLY)
configure_file ("${CMAKE_CURRENT_SOURCE_DIR}/include/SmallVector.hpp" "${PROJECT_BINARY_DIR}/include/SmallVector.hpp" COPYONLY)
configure_file ("${CMAKE_CURRENT_SOURCE_DIR}/include/ArrayRef.hpp" "${PROJECT_BINARY_DIR}/include/ArrayRef.hpp" COPYONLY)
configure_file ("${CMAKE_CURRENT_SOURCE_DIR}/include/AMSTensor.hpp" "${PROJECT_BINARY_DIR}/include/AMSTensor.hpp" COPYONLY)

# setup the exec
#SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath -Wl,$ORIGIN")
# ------------------------------------------------------------------------------
# installation paths

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Install the AMS library
install(TARGETS AMS
    EXPORT AMSTargets
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"      # For shared libraries
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"      # For static libraries
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"      # For executables (Windows-specific)
    INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}" # Install headers directory
)

# Export the AMS targets for use by external projects
install(EXPORT AMSTargets
    NAMESPACE AMS::                              # Prefix target names with AMS::
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/AMS" # Install location for CMake config
)

# Install the public headers
install(FILES
    ${PROJECT_BINARY_DIR}/include/AMS.h
    ${PROJECT_BINARY_DIR}/include/AMSTensor.hpp
    ${PROJECT_BINARY_DIR}/include/AMSTypes.hpp
    ${PROJECT_BINARY_DIR}/include/ArrayRef.hpp
    ${PROJECT_BINARY_DIR}/include/SmallVector.hpp
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/"    # Headers installed into AMS subdir
)


# Generate a version file for the package
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/AMSConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

# Configure the package configuration file
configure_package_config_file(
    "${CMAKE_SOURCE_DIR}/cmake/AMSConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/AMSConfig.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/AMS"
)

# Install the generated config and version files
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/AMSConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/AMSConfigVersion.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/AMS"
)

