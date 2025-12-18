function(ADD_WORKFLOW_UNIT_TEST name exec)
  string(JOIN " " args ${ARGN})
  add_test(NAME ${name} COMMAND ${exec} -s --reporter console)
  set_tests_properties(${name} PROPERTIES LABELS WORKFLOW_UNIT_TEST)
endfunction()

function(BUILD_UNIT_TEST exe source)
  add_executable(${exe} ${source})

  target_compile_features(${exe} PRIVATE cxx_std_17)
  target_include_directories(${exe} PRIVATE ${CMAKE_SOURCE_DIR}/src/AMSlib/)
  target_include_directories(${exe} PRIVATE ${CMAKE_BINARY_DIR}/include/)
  target_include_directories(${exe} PRIVATE ${AMS_TEST_ROOT})
  target_link_libraries(${exe} PRIVATE stdc++fs AMS torch Catch2::Catch2WithMain)
  target_compile_definitions(${exe} PRIVATE ${AMS_APP_DEFINES} CATCH_CONFIG_PREFIX_ALL)

  target_link_libraries(${exe} PRIVATE ${AMS_HDF5_TARGET})


  if(WITH_CUDA)
    target_link_libraries(${exe} PRIVATE CUDA::cudart)
  elseif(WITH_HIP)
    target_link_libraries(${exe} PRIVATE hip::host)
  endif()

  if (WITH_CALIPER)
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
endfunction()

BUILD_UNIT_TEST(operations operations.cpp)
ADD_WORKFLOW_UNIT_TEST(WORKFLOW::OPERATIONS operations)

BUILD_UNIT_TEST(evaluate_in_and_outs evaluate_in_and_outs.cpp)
ADD_WORKFLOW_UNIT_TEST(WORKFLOW::EVALUATE_IN_OUTS evaluate_in_and_outs)

BUILD_UNIT_TEST(tensor_bundle tensor_bundle.cpp)
ADD_WORKFLOW_UNIT_TEST(WORKFLOW::TENSOR_BUNDLE tensor_bundle)

BUILD_UNIT_TEST(eval_context eval_context.cpp)
ADD_WORKFLOW_UNIT_TEST(WORKFLOW::EVAL_CONTEXT eval_context)

BUILD_UNIT_TEST(pointwise pointwise_layout_transform.cpp)
ADD_WORKFLOW_UNIT_TEST(WORKFLOW::POINTWISE pointwise)

BUILD_UNIT_TEST(action action.cpp)
ADD_WORKFLOW_UNIT_TEST(WORKFLOW::ACTION action)

BUILD_UNIT_TEST(pipeline pipeline.cpp)
ADD_WORKFLOW_UNIT_TEST(WORKFLOW::PIPELINE pipeline)

BUILD_UNIT_TEST(policy policy.cpp)
ADD_WORKFLOW_UNIT_TEST(WORKFLOW::POLICY policy)
