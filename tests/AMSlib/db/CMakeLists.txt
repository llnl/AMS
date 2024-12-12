function(BUILD_UNIT_TEST exe source)
  add_executable(${exe} ${source})

  target_include_directories(${exe} PRIVATE ${caliper_INCLUDE_DIR} ${MPI_INCLUDE_PATH})
  target_include_directories(${exe} PRIVATE ${CMAKE_SOURCE_DIR}/src/AMSlib/)
  target_include_directories(${exe} PRIVATE ${CMAKE_BINARY_DIR}/include/)
  target_link_directories(${exe} PRIVATE ${AMS_APP_LIB_DIRS})
  target_link_libraries(${exe} PRIVATE ${AMS_APP_LIBRARIES} stdc++fs AMS)

  target_compile_definitions(${exe} PRIVATE ${AMS_APP_DEFINES})

  if(WITH_CUDA)
    set_target_properties(${exe} PROPERTIES CUDA_ARCHITECTURES "${AMS_CUDA_ARCH}")
    set_property(TARGET ${exe} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    set_source_files_properties(${source} PROPERTIES LANGUAGE CUDA)

    target_compile_definitions(${exe} PRIVATE "-D__ENABLE_CUDA__ -DLIBAMS_VERBOSE")
  endif()
endfunction()

function(ADD_HDF5_UNIT_TEST name exec)
  add_test(NAME ${name} COMMAND ${exec} ${ARGN})
  set_tests_properties(${name} PROPERTIES LABELS HDF5_UNIT_TEST)
endfunction()

if (WITH_HDF5)
BUILD_UNIT_TEST(hdf5_create hdf5_open.cpp)
BUILD_UNIT_TEST(hdf5_store hdf5_store.cpp)
ADD_HDF5_UNIT_TEST(DB::HDF5::CREATE COMMAND bash -c "rm -f ${CMAKE_CURRENT_BINARY_DIR}/create_test*.h5\;${CMAKE_CURRENT_BINARY_DIR}/hdf5_create  ${CMAKE_CURRENT_BINARY_DIR} test create_test")
ADD_HDF5_UNIT_TEST(DB::HDF5::STORE COMMAND bash -c "rm -f ${CMAKE_CURRENT_BINARY_DIR}/store_test*.h5\;${CMAKE_CURRENT_BINARY_DIR}/hdf5_store ${CMAKE_CURRENT_BINARY_DIR} test store_test")
endif()

