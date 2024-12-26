# Copyright (c) Lawrence Livermore National Security, LLC and other AMS
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute

# ------------------------------------------------------------------------------
# handle sources and headers
#set global library path to link with tests if necessary
set(LIBRARY_OUTPUT_PATH ${AMS_LIB_OUT_PATH})
set(AMS_LIB_SRC wf/debug.cpp wf/logger.cpp wf/utils.cpp wf/SmallVector.cpp ml/surrogate.cpp wf/hdf5db.cpp wf/basedb.cpp AMSTensor.cpp wf/interface.cpp wf/resource_manager.cpp AMS.cpp)

#if (WITH_CUDA)
#  list(APPEND AMS_LIB_SRC wf/cuda/utilities.cpp)
#endif()

#if (WITH_HDF5)
#  list(APPEND AMS_LIB_SRC wf/hdf5db.cpp)
#endif()
#
#if (WITH_RMQ)
#  list(APPEND AMS_LIB_SRC wf/rmqdb.cpp)
#endif()

add_library(AMS ${AMS_LIB_SRC})

# ------------------------------------------------------------------------------
# setup the lib first
message(STATUS "ALL INCLUDES ARE ${AMS_APP_INCLUDES}")
target_compile_definitions(AMS PRIVATE ${AMS_APP_DEFINES})
target_include_directories(AMS PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/include)

if (WITH_CUDA)
  target_link_libraries(AMS PRIVATE CUDA::cudart)
endif()

if (WITH_HDF5)
  target_link_libraries(AMS PRIVATE ${AMS_HDF5_TARGET})
endif()

if (WITH_MPI)
  target_link_libraries(AMS PRIVATE MPI::MPI_CXX)
endif()

target_link_libraries(AMS PRIVATE stdc++fs torch)


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

# Install the library
install(TARGETS AMS
    EXPORT AMSTargets
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
)


install(EXPORT AMSTargets
    NAMESPACE AMS::
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/AMS"
)

#
install(FILES ${PROJECT_BINARY_DIR}/include/AMS.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES ${PROJECT_BINARY_DIR}/include/AMSTensor.hpp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES ${PROJECT_BINARY_DIR}/include/AMSTypes.hpp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES ${PROJECT_BINARY_DIR}/include/ArrayRef.hpp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES ${PROJECT_BINARY_DIR}/include/SmallVector.hpp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
#install(FILES ${PROJECT_BINARY_DIR}/include/AMS-config.h DESTINATION include)
