function(ADD_WORKFLOW_UNIT_TEST name exec)
  add_test(NAME ${name} COMMAND ${exec} ${ARGN})
  set_tests_properties(${name} PROPERTIES LABELS WORKFLOW_UNIT_TEST)
endfunction()

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
  set_tests_properties(${name} PROPERTIES LABELS WORKFLOW_UNIT_TEST)
endfunction()

BUILD_UNIT_TEST(domain_to_application domain_to_application.cpp)
BUILD_UNIT_TEST(subselect_tensors subselect_tensors.cpp)
BUILD_UNIT_TEST(scatter_physics scatter_physics.cpp)

ADD_WORKFLOW_UNIT_TEST(WORKFLOW::HOST::DomainToApplication::FLOAT::FLOAT domain_to_application float float cpu)
ADD_WORKFLOW_UNIT_TEST(WORKFLOW::HOST::DomainToApplication::DOUBLE::FLOAT domain_to_application double float cpu)
ADD_WORKFLOW_UNIT_TEST(WORKFLOW::HOST::DomainToApplication::FLOAT::DOUBLE domain_to_application float double cpu)
ADD_WORKFLOW_UNIT_TEST(WORKFLOW::HOST::DomainToApplication::DOUBLE::DOUBLE domain_to_application double double cpu)


ADD_WORKFLOW_UNIT_TEST(WORKFLOW::HOST::SubSelect::Double subselect_tensors double cpu)
ADD_WORKFLOW_UNIT_TEST(WORKFLOW::HOST::SubSelect::Float subselect_tensors float cpu)

ADD_WORKFLOW_UNIT_TEST(WORKFLOW::HOST::ScatterPhysics::Double scatter_physics double cpu)
ADD_WORKFLOW_UNIT_TEST(WORKFLOW::HOST::ScatterPhysics::Float scatter_physics float cpu)


if (ENABLE_CUDA)
ADD_WORKFLOW_UNIT_TEST(WORKFLOW::DEVICE::DomainToApplication::FLOAT::FLOAT domain_to_application float float cuda)
ADD_WORKFLOW_UNIT_TEST(WORKFLOW::DEVICE::DomainToApplication::DOUBLE::FLOAT domain_to_application double float cuda)
ADD_WORKFLOW_UNIT_TEST(WORKFLOW::DEVICE::DomainToApplication::FLOAT::DOUBLE domain_to_application float double cuda)
ADD_WORKFLOW_UNIT_TEST(WORKFLOW::DEVICE::DomainToApplication::DOUBLE::DOUBLE domain_to_application double double cuda)

ADD_WORKFLOW_UNIT_TEST(WORKFLOW::DEVICE::SubSelect::Double subselect_tensors double cuda)
ADD_WORKFLOW_UNIT_TEST(WORKFLOW::DEVICE::SubSelect::Float subselect_tensors float cuda)

ADD_WORKFLOW_UNIT_TEST(WORKFLOW::DEVICE::ScatterPhysics::Double scatter_physics double cuda)
ADD_WORKFLOW_UNIT_TEST(WORKFLOW::DEVICE::ScatterPhysics::Float scatter_physics float cuda)
endif()

