
function(ADD_TORCH_UNIT_TEST name exec)
  add_test(NAME ${name} COMMAND ${exec} ${ARGN})
  set_tests_properties(${name} PROPERTIES LABELS TORCH_UNIT_TEST)
endfunction()


function(BUILD_UNIT_TEST exe source)
  add_executable(${exe} ${source})
  add_dependencies(${exe} generate_cpu_models)

  if (ENABLE_CUDA)
    add_dependencies(${exe} generate_gpu_models)
  endif()

  target_include_directories(${exe} PRIVATE ${caliper_INCLUDE_DIR} ${MPI_INCLUDE_PATH})
  target_include_directories(${exe} PRIVATE ${CMAKE_SOURCE_DIR}/src/AMSlib/)
  target_include_directories(${exe} PRIVATE ${CMAKE_BINARY_DIR}/include/)
  target_link_directories(${exe} PRIVATE ${AMS_APP_LIB_DIRS})
  target_link_libraries(${exe} PRIVATE ${AMS_APP_LIBRARIES} stdc++fs)

  target_compile_definitions(${exe} PRIVATE ${AMS_APP_DEFINES})

  if(WITH_CUDA)
    set_target_properties(${exe} PROPERTIES CUDA_ARCHITECTURES "${AMS_CUDA_ARCH}")
    set_property(TARGET ${exe} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    set_source_files_properties(${source} PROPERTIES LANGUAGE CUDA)

    target_compile_definitions(${exe} PRIVATE "-D__ENABLE_CUDA__ -DLIBAMS_VERBOSE")
  endif()
endfunction()

# Output files for generated models (example: .pt files)
set(GENERATED_CPU_MODELS
  ${CMAKE_CURRENT_BINARY_DIR}/models/double_cpu_duq_max.pt  
  ${CMAKE_CURRENT_BINARY_DIR}/models/double_cpu_duq_mean.pt  
  ${CMAKE_CURRENT_BINARY_DIR}/models/double_cpu_random.pt  
  ${CMAKE_CURRENT_BINARY_DIR}/models/single_cpu_duq_max.pt  
  ${CMAKE_CURRENT_BINARY_DIR}/models/single_cpu_duq_mean.pt  
  ${CMAKE_CURRENT_BINARY_DIR}/models/single_cpu_random.pt
)

# Custom command to generate models
add_custom_command(
  OUTPUT ${GENERATED_CPU_MODELS}
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/generate.sh ${CMAKE_CURRENT_BINARY_DIR}/models/ "cpu"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/generate.sh;${CMAKE_CURRENT_SOURCE_DIR}/generate.py" 
    COMMENT "Generating PyTorch models..."
)

# Define a target to generate models
add_custom_target(generate_cpu_models ALL
  DEPENDS ${GENERATED_CPU_MODELS}
    COMMENT "Generate cpu models before ctests..."
)

if (ENABLE_CUDA)
set(GENERATED_GPU_MODELS
  ${CMAKE_CURRENT_BINARY_DIR}/models/double_gpu_duq_max.pt  
  ${CMAKE_CURRENT_BINARY_DIR}/models/double_gpu_duq_mean.pt  
  ${CMAKE_CURRENT_BINARY_DIR}/models/double_gpu_random.pt  
  ${CMAKE_CURRENT_BINARY_DIR}/models/single_gpu_duq_max.pt  
  ${CMAKE_CURRENT_BINARY_DIR}/models/single_gpu_duq_mean.pt  
  ${CMAKE_CURRENT_BINARY_DIR}/models/single_gpu_random.pt
)

# Custom command to generate models
add_custom_command(
  OUTPUT ${GENERATED_GPU_MODELS}
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/generate.sh ${CMAKE_CURRENT_BINARY_DIR}/models/ "gpu"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/generate.sh;${CMAKE_CURRENT_SOURCE_DIR}/generate.py" 
    COMMENT "Generating PyTorch models..."
)

# Define a target to generate models
add_custom_target(generate_gpu_models ALL
  DEPENDS ${GENERATED_GPU_MODELS}
    COMMENT "Generate gpu models before ctests..."
)
endif()




BUILD_UNIT_TEST(ams_verify_dtype "verify_dtype.cpp;${CMAKE_SOURCE_DIR}/src/AMSlib/ml/surrogate.cpp;${CMAKE_SOURCE_DIR}/src/AMSlib/wf/logger.cpp;${CMAKE_SOURCE_DIR}/src/AMSlib/wf/debug.cpp")
BUILD_UNIT_TEST(ams_verify_device "verify_device.cpp;${CMAKE_SOURCE_DIR}/src/AMSlib/ml/surrogate.cpp;${CMAKE_SOURCE_DIR}/src/AMSlib/wf/logger.cpp;${CMAKE_SOURCE_DIR}/src/AMSlib/wf/debug.cpp")
BUILD_UNIT_TEST(ams_evaluate_model "evaluate_model.cpp;${CMAKE_SOURCE_DIR}/src/AMSlib/ml/surrogate.cpp;${CMAKE_SOURCE_DIR}/src/AMSlib/wf/logger.cpp;${CMAKE_SOURCE_DIR}/src/AMSlib/wf/debug.cpp")
BUILD_UNIT_TEST(ams_convert_and_evaluate_model "evalute_model_conversions.cpp;${CMAKE_SOURCE_DIR}/src/AMSlib/ml/surrogate.cpp;${CMAKE_SOURCE_DIR}/src/AMSlib/wf/logger.cpp;${CMAKE_SOURCE_DIR}/src/AMSlib/wf/debug.cpp;${CMAKE_SOURCE_DIR}/src/AMSlib/util/SmallVector.cpp;")

#detect datatype
ADD_TORCH_UNIT_TEST(Surrogate::DType::HOST::Double::Random ams_verify_dtype double ${CMAKE_CURRENT_BINARY_DIR}/models/double_cpu_random.pt)
ADD_TORCH_UNIT_TEST(Surrogate::DType::HOST::Double::Duq_mean ams_verify_dtype double ${CMAKE_CURRENT_BINARY_DIR}/models/double_cpu_duq_mean.pt)
ADD_TORCH_UNIT_TEST(Surrogate::DType::HOST::Double::Duq_max ams_verify_dtype double ${CMAKE_CURRENT_BINARY_DIR}/models/double_cpu_duq_max.pt)

ADD_TORCH_UNIT_TEST(Surrogate::DType::HOST::Single::Random ams_verify_dtype single ${CMAKE_CURRENT_BINARY_DIR}/models/single_cpu_random.pt)
ADD_TORCH_UNIT_TEST(Surrogate::DType::HOST::Single::Duq_mean ams_verify_dtype single ${CMAKE_CURRENT_BINARY_DIR}/models/single_cpu_duq_mean.pt)
ADD_TORCH_UNIT_TEST(Surrogate::DType::HOST::Single::Duq_max ams_verify_dtype single ${CMAKE_CURRENT_BINARY_DIR}/models/single_cpu_duq_max.pt)

#detect device
ADD_TORCH_UNIT_TEST(Surrogate::DResource::HOST::Double::Random ams_verify_device cpu ${CMAKE_CURRENT_BINARY_DIR}/models/double_cpu_random.pt)
ADD_TORCH_UNIT_TEST(Surrogate::DResource::HOST::Double::Duq_mean ams_verify_device cpu ${CMAKE_CURRENT_BINARY_DIR}/models/double_cpu_duq_mean.pt)
ADD_TORCH_UNIT_TEST(Surrogate::DResource::HOST::Double::Duq_max ams_verify_device cpu ${CMAKE_CURRENT_BINARY_DIR}/models/double_cpu_duq_max.pt)

ADD_TORCH_UNIT_TEST(Surrogate::DResource::HOST::Single::Random ams_verify_device cpu ${CMAKE_CURRENT_BINARY_DIR}/models/single_cpu_random.pt)
ADD_TORCH_UNIT_TEST(Surrogate::DResource::HOST::Single::Duq_mean ams_verify_device cpu ${CMAKE_CURRENT_BINARY_DIR}/models/single_cpu_duq_mean.pt)
ADD_TORCH_UNIT_TEST(Surrogate::DResource::HOST::Single::Duq_max ams_verify_device cpu ${CMAKE_CURRENT_BINARY_DIR}/models/single_cpu_duq_max.pt)


ADD_TORCH_UNIT_TEST(Surrogate::Evaluate::HOST::Random::Double ams_evaluate_model "2000,8" "2000,8" ${CMAKE_CURRENT_BINARY_DIR}/models/double_cpu_random.pt "random")
ADD_TORCH_UNIT_TEST(Surrogate::Evaluate::HOST::Random::Single ams_evaluate_model "2000,8" "2000,8" ${CMAKE_CURRENT_BINARY_DIR}/models/single_cpu_random.pt "random")

ADD_TORCH_UNIT_TEST(Surrogate::Evaluate::HOST::UQMean::Double ams_evaluate_model "2000,8" "2000,8" ${CMAKE_CURRENT_BINARY_DIR}/models/double_cpu_duq_mean.pt "duq_mean")
ADD_TORCH_UNIT_TEST(Surrogate::Evaluate::HOST::UQMean::Single ams_evaluate_model "2000,8" "2000,8" ${CMAKE_CURRENT_BINARY_DIR}/models/single_cpu_duq_mean.pt "duq_mean")

ADD_TORCH_UNIT_TEST(Surrogate::Evaluate::HOST::UQMax::Double ams_evaluate_model "2000,8" "2000,8" ${CMAKE_CURRENT_BINARY_DIR}/models/double_cpu_duq_max.pt "duq_max")
ADD_TORCH_UNIT_TEST(Surrogate::Evaluate::HOST::UQMax::Single ams_evaluate_model "2000,8" "2000,8" ${CMAKE_CURRENT_BINARY_DIR}/models/single_cpu_duq_max.pt "duq_max")

ADD_TORCH_UNIT_TEST(Surrogate::EvaluateAndCat::HOST::Random::Double ams_convert_and_evaluate_model "2000,8" "2000,8" ${CMAKE_CURRENT_BINARY_DIR}/models/double_cpu_random.pt "random")
ADD_TORCH_UNIT_TEST(Surrogate::EvaluateAndCat::HOST::Random::Single ams_convert_and_evaluate_model "2000,8" "2000,8" ${CMAKE_CURRENT_BINARY_DIR}/models/single_cpu_random.pt "random")

ADD_TORCH_UNIT_TEST(Surrogate::EvaluateAndCat::HOST::UQMean::Double ams_convert_and_evaluate_model "2000,8" "2000,8" ${CMAKE_CURRENT_BINARY_DIR}/models/double_cpu_duq_mean.pt "duq_mean")
ADD_TORCH_UNIT_TEST(Surrogate::EvaluateAndCat::HOST::UQMean::Single ams_convert_and_evaluate_model "2000,8" "2000,8" ${CMAKE_CURRENT_BINARY_DIR}/models/single_cpu_duq_mean.pt "duq_mean")

ADD_TORCH_UNIT_TEST(Surrogate::EvaluateAndCat::HOST::UQMax::Double ams_convert_and_evaluate_model "2000,8" "2000,8" ${CMAKE_CURRENT_BINARY_DIR}/models/double_cpu_duq_max.pt "duq_max")
ADD_TORCH_UNIT_TEST(Surrogate::EvaluateAndCat::HOST::UQMax::Single ams_convert_and_evaluate_model "2000,8" "2000,8" ${CMAKE_CURRENT_BINARY_DIR}/models/single_cpu_duq_max.pt "duq_max")

if(ENABLE_CUDA)

#detect datatype
ADD_TORCH_UNIT_TEST(Surrogate::DType::DEVICE::Double::Random ams_verify_dtype double ${CMAKE_CURRENT_BINARY_DIR}/models/double_gpu_random.pt)
ADD_TORCH_UNIT_TEST(Surrogate::DType::DEVICE::Double::Duq_mean ams_verify_dtype double ${CMAKE_CURRENT_BINARY_DIR}/models/double_gpu_duq_mean.pt)
ADD_TORCH_UNIT_TEST(Surrogate::DType::DEVICE::Double::Duq_max ams_verify_dtype double ${CMAKE_CURRENT_BINARY_DIR}/models/double_gpu_duq_max.pt)

ADD_TORCH_UNIT_TEST(Surrogate::DType::DEVICE::Single::Random ams_verify_dtype single ${CMAKE_CURRENT_BINARY_DIR}/models/single_gpu_random.pt)
ADD_TORCH_UNIT_TEST(Surrogate::DType::DEVICE::Single::Duq_mean ams_verify_dtype single ${CMAKE_CURRENT_BINARY_DIR}/models/single_gpu_duq_mean.pt)
ADD_TORCH_UNIT_TEST(Surrogate::DType::DEVICE::Single::Duq_max ams_verify_dtype single ${CMAKE_CURRENT_BINARY_DIR}/models/single_gpu_duq_max.pt)

#detect device
ADD_TORCH_UNIT_TEST(Surrogate::DResource::DEVICE::Double::Random ams_verify_device gpu ${CMAKE_CURRENT_BINARY_DIR}/models/double_gpu_random.pt)
ADD_TORCH_UNIT_TEST(Surrogate::DResource::DEVICE::Double::Duq_mean ams_verify_device gpu ${CMAKE_CURRENT_BINARY_DIR}/models/double_gpu_duq_mean.pt)
ADD_TORCH_UNIT_TEST(Surrogate::DResource::DEVICE::Double::Duq_max ams_verify_device gpu ${CMAKE_CURRENT_BINARY_DIR}/models/double_gpu_duq_max.pt)

ADD_TORCH_UNIT_TEST(Surrogate::DResource::DEVICE::Single::Random ams_verify_device gpu ${CMAKE_CURRENT_BINARY_DIR}/models/single_gpu_random.pt)
ADD_TORCH_UNIT_TEST(Surrogate::DResource::DEVICE::Single::Duq_mean ams_verify_device gpu ${CMAKE_CURRENT_BINARY_DIR}/models/single_gpu_duq_mean.pt)
ADD_TORCH_UNIT_TEST(Surrogate::DResource::DEVICE::Single::Duq_max ams_verify_device gpu ${CMAKE_CURRENT_BINARY_DIR}/models/single_gpu_duq_max.pt)


ADD_TORCH_UNIT_TEST(Surrogate::Evaluate::DEVICE::Random::Double ams_evaluate_model "2000,8" "2000,8" ${CMAKE_CURRENT_BINARY_DIR}/models/double_gpu_random.pt "random")
ADD_TORCH_UNIT_TEST(Surrogate::Evaluate::DEVICE::Random::Single ams_evaluate_model "2000,8" "2000,8" ${CMAKE_CURRENT_BINARY_DIR}/models/single_gpu_random.pt "random")

ADD_TORCH_UNIT_TEST(Surrogate::Evaluate::DEVICE::UQMean::Double ams_evaluate_model "2000,8" "2000,8" ${CMAKE_CURRENT_BINARY_DIR}/models/double_gpu_duq_mean.pt "duq_mean")
ADD_TORCH_UNIT_TEST(Surrogate::Evaluate::DEVICE::UQMean::Single ams_evaluate_model "2000,8" "2000,8" ${CMAKE_CURRENT_BINARY_DIR}/models/single_gpu_duq_mean.pt "duq_mean")

ADD_TORCH_UNIT_TEST(Surrogate::Evaluate::DEVICE::UQMax::Double ams_evaluate_model "2000,8" "2000,8" ${CMAKE_CURRENT_BINARY_DIR}/models/double_gpu_duq_max.pt "duq_max")
ADD_TORCH_UNIT_TEST(Surrogate::Evaluate::DEVICE::UQMax::Single ams_evaluate_model "2000,8" "2000,8" ${CMAKE_CURRENT_BINARY_DIR}/models/single_gpu_duq_max.pt "duq_max")

ADD_TORCH_UNIT_TEST(Surrogate::EvaluateAndCat::DEVICE::Random::Double ams_convert_and_evaluate_model "2000,8" "2000,8" ${CMAKE_CURRENT_BINARY_DIR}/models/double_gpu_random.pt "random")
ADD_TORCH_UNIT_TEST(Surrogate::EvaluateAndCat::DEVICE::Random::Single ams_convert_and_evaluate_model "2000,8" "2000,8" ${CMAKE_CURRENT_BINARY_DIR}/models/single_gpu_random.pt "random")

ADD_TORCH_UNIT_TEST(Surrogate::EvaluateAndCat::DEVICE::UQMean::Double ams_convert_and_evaluate_model "2000,8" "2000,8" ${CMAKE_CURRENT_BINARY_DIR}/models/double_gpu_duq_mean.pt "duq_mean")
ADD_TORCH_UNIT_TEST(Surrogate::EvaluateAndCat::DEVICE::UQMean::Single ams_convert_and_evaluate_model "2000,8" "2000,8" ${CMAKE_CURRENT_BINARY_DIR}/models/single_gpu_duq_mean.pt "duq_mean")

ADD_TORCH_UNIT_TEST(Surrogate::EvaluateAndCat::DEVICE::UQMax::Double ams_convert_and_evaluate_model "2000,8" "2000,8" ${CMAKE_CURRENT_BINARY_DIR}/models/double_gpu_duq_max.pt "duq_max")
ADD_TORCH_UNIT_TEST(Surrogate::EvaluateAndCat::DEVICE::UQMax::Single ams_convert_and_evaluate_model "2000,8" "2000,8" ${CMAKE_CURRENT_BINARY_DIR}/models/single_gpu_duq_max.pt "duq_max")

endif()


