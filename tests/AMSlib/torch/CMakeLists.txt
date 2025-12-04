
function(ADD_TORCH_UNIT_TEST name exec)
  add_test(NAME ${name} COMMAND ${exec}  -s --reporter console)
  set_tests_properties(${name} PROPERTIES LABELS TORCH_UNIT_TEST)
endfunction()


function(BUILD_UNIT_TEST exe source)
  add_executable(${exe} ${source})
  target_compile_features(${exe} PRIVATE cxx_std_17)
  if (WITH_CUDA)
    target_link_libraries(${exe} PRIVATE CUDA::cudart)
  elseif(WITH_HIP)
    target_link_libraries(${exe} PRIVATE hip::host)
  endif()
  
  target_include_directories(${exe} PRIVATE ${CMAKE_SOURCE_DIR}/src/AMSlib/)
  target_include_directories(${exe} PRIVATE ${CMAKE_BINARY_DIR}/include/)
  target_include_directories(${exe} PRIVATE ${AMS_TEST_ROOT})
  target_link_libraries(${exe} PRIVATE stdc++fs AMS torch Catch2::Catch2WithMain)
  target_link_libraries(${exe} PRIVATE ${AMS_HDF5_TARGET})
  target_compile_definitions(${exe} PRIVATE ${AMS_APP_DEFINES} CATCH_CONFIG_PREFIX_ALL)
endfunction()


BUILD_UNIT_TEST(ams_surrogate_tests evaluate_model.cpp)
ADD_TORCH_UNIT_TEST(Surrogate ams_surrogate_tests)

BUILD_UNIT_TEST(ams_test_base_model test_model.cpp)
ADD_TORCH_UNIT_TEST(BaseModel ams_test_base_model)

BUILD_UNIT_TEST(ams_test_inference_model test_inference_model.cpp)
ADD_TORCH_UNIT_TEST(InferenceModel ams_test_inference_model)
