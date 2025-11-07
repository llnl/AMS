function(BUILD_UNIT_TEST exe source)
  add_executable(${exe} ${source})

  target_include_directories(${exe} PRIVATE ${caliper_INCLUDE_DIR} ${MPI_INCLUDE_PATH})
  target_include_directories(${exe} PRIVATE ${CMAKE_SOURCE_DIR}/src/AMSlib/)
  target_include_directories(${exe} PRIVATE ${CMAKE_BINARY_DIR}/include/)
  target_link_libraries(${exe} PRIVATE stdc++fs AMS torch Catch2::Catch2WithMain)
  target_include_directories(${exe} PRIVATE ${AMS_TEST_ROOT})
  target_compile_features(${exe} PRIVATE cxx_std_17)

  target_link_libraries(${exe} PRIVATE ${AMS_HDF5_TARGET})

  if (WITH_CALIPER)
    message(STATUS "Building witth caliper ${exe}")
    target_link_libraries(${exe} PRIVATE caliper)
  endif()

  if (WITH_RMQ)
    target_link_libraries(${exe} PRIVATE amqpcpp)
    if (OPENSSL_FOUND)
      target_link_libraries(${exe} PRIVATE OpenSSL::SSL OpenSSL::Crypto)
    endif()
    # NOTE: We set here the event/event pthreads as public. As there is no easy way
    # to do a find package(libevent) and RMQ is not exposing that properly.
    target_link_libraries(${exe} PRIVATE ${LIBEVENT_LIBRARY} ${LIBEVENT_THREAD})
  endif()

  if (WITH_MPI)
    target_link_libraries(${exe} PRIVATE MPI::MPI_CXX)
  endif()

  target_compile_definitions(${exe} PRIVATE ${AMS_APP_DEFINES})
endfunction()

function(ADD_DB_UNIT_TEST name exec)
  add_test(NAME ${name} COMMAND ${exec}  -s --reporter console)
  set_tests_properties(${name} PROPERTIES LABELS HDF5_UNIT_TEST)
endfunction()

BUILD_UNIT_TEST(db_hdf5 db_hdf5.cpp)
ADD_DB_UNIT_TEST(DB::HDF5 db_hdf5)
