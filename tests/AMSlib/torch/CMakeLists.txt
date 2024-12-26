
function(ADD_TORCH_UNIT_TEST name exec)
  add_test(NAME ${name} COMMAND ${exec} ${ARGN})
  set_tests_properties(${name} PROPERTIES LABELS TORCH_UNIT_TEST)
endfunction()


function(BUILD_UNIT_TEST exe source)
  add_executable(${exe} ${source})
  add_dependencies(${exe} generate_cpu_models)

  if (WITH_CUDA)
    add_dependencies(${exe} generate_gpu_models)
    target_link_libraries(${exe} PRIVATE CUDA::cudart)
  endif()

  target_include_directories(${exe} PRIVATE ${CMAKE_SOURCE_DIR}/src/AMSlib/)
  target_include_directories(${exe} PRIVATE ${CMAKE_BINARY_DIR}/include/)
  target_link_libraries(${exe} PRIVATE stdc++fs AMS torch)
  if (WITH_HDF5)
    target_link_libraries(${exe} PRIVATE ${AMS_HDF5_TARGET})
  endif()


  target_compile_definitions(${exe} PRIVATE ${AMS_APP_DEFINES})
endfunction()


BUILD_UNIT_TEST(ams_verify_dtype verify_dtype.cpp)
BUILD_UNIT_TEST(ams_verify_device verify_device.cpp)
BUILD_UNIT_TEST(ams_evaluate_model evaluate_model.cpp)
BUILD_UNIT_TEST(ams_convert_and_evaluate_model evalute_model_conversions.cpp)

#detect datatype
ADD_TORCH_UNIT_TEST(Surrogate::DType::HOST::Double::Random ams_verify_dtype double ${TORCH_MODEL_DIR}/double_cpu_random.pt)
ADD_TORCH_UNIT_TEST(Surrogate::DType::HOST::Double::Duq_mean ams_verify_dtype double ${TORCH_MODEL_DIR}/double_cpu_duq_mean.pt)
ADD_TORCH_UNIT_TEST(Surrogate::DType::HOST::Double::Duq_max ams_verify_dtype double ${TORCH_MODEL_DIR}/double_cpu_duq_max.pt)

ADD_TORCH_UNIT_TEST(Surrogate::DType::HOST::Single::Random ams_verify_dtype single ${TORCH_MODEL_DIR}/single_cpu_random.pt)
ADD_TORCH_UNIT_TEST(Surrogate::DType::HOST::Single::Duq_mean ams_verify_dtype single ${TORCH_MODEL_DIR}/single_cpu_duq_mean.pt)
ADD_TORCH_UNIT_TEST(Surrogate::DType::HOST::Single::Duq_max ams_verify_dtype single ${TORCH_MODEL_DIR}/single_cpu_duq_max.pt)

#detect device
ADD_TORCH_UNIT_TEST(Surrogate::DResource::HOST::Double::Random ams_verify_device cpu ${TORCH_MODEL_DIR}/double_cpu_random.pt)
ADD_TORCH_UNIT_TEST(Surrogate::DResource::HOST::Double::Duq_mean ams_verify_device cpu ${TORCH_MODEL_DIR}/double_cpu_duq_mean.pt)
ADD_TORCH_UNIT_TEST(Surrogate::DResource::HOST::Double::Duq_max ams_verify_device cpu ${TORCH_MODEL_DIR}/double_cpu_duq_max.pt)

ADD_TORCH_UNIT_TEST(Surrogate::DResource::HOST::Single::Random ams_verify_device cpu ${TORCH_MODEL_DIR}/single_cpu_random.pt)
ADD_TORCH_UNIT_TEST(Surrogate::DResource::HOST::Single::Duq_mean ams_verify_device cpu ${TORCH_MODEL_DIR}/single_cpu_duq_mean.pt)
ADD_TORCH_UNIT_TEST(Surrogate::DResource::HOST::Single::Duq_max ams_verify_device cpu ${TORCH_MODEL_DIR}/single_cpu_duq_max.pt)


ADD_TORCH_UNIT_TEST(Surrogate::Evaluate::HOST::Random::Double ams_evaluate_model "2000,8" "2000,8" ${TORCH_MODEL_DIR}/double_cpu_random.pt "random")
ADD_TORCH_UNIT_TEST(Surrogate::Evaluate::HOST::Random::Single ams_evaluate_model "2000,8" "2000,8" ${TORCH_MODEL_DIR}/single_cpu_random.pt "random")

ADD_TORCH_UNIT_TEST(Surrogate::Evaluate::HOST::UQMean::Double ams_evaluate_model "2000,8" "2000,8" ${TORCH_MODEL_DIR}/double_cpu_duq_mean.pt "duq_mean")
ADD_TORCH_UNIT_TEST(Surrogate::Evaluate::HOST::UQMean::Single ams_evaluate_model "2000,8" "2000,8" ${TORCH_MODEL_DIR}/single_cpu_duq_mean.pt "duq_mean")

ADD_TORCH_UNIT_TEST(Surrogate::Evaluate::HOST::UQMax::Double ams_evaluate_model "2000,8" "2000,8" ${TORCH_MODEL_DIR}/double_cpu_duq_max.pt "duq_max")
ADD_TORCH_UNIT_TEST(Surrogate::Evaluate::HOST::UQMax::Single ams_evaluate_model "2000,8" "2000,8" ${TORCH_MODEL_DIR}/single_cpu_duq_max.pt "duq_max")

ADD_TORCH_UNIT_TEST(Surrogate::EvaluateAndCat::HOST::Random::Double ams_convert_and_evaluate_model "2000,8" "2000,8" ${TORCH_MODEL_DIR}/double_cpu_random.pt "random")
ADD_TORCH_UNIT_TEST(Surrogate::EvaluateAndCat::HOST::Random::Single ams_convert_and_evaluate_model "2000,8" "2000,8" ${TORCH_MODEL_DIR}/single_cpu_random.pt "random")

ADD_TORCH_UNIT_TEST(Surrogate::EvaluateAndCat::HOST::UQMean::Double ams_convert_and_evaluate_model "2000,8" "2000,8" ${TORCH_MODEL_DIR}/double_cpu_duq_mean.pt "duq_mean")
ADD_TORCH_UNIT_TEST(Surrogate::EvaluateAndCat::HOST::UQMean::Single ams_convert_and_evaluate_model "2000,8" "2000,8" ${TORCH_MODEL_DIR}/single_cpu_duq_mean.pt "duq_mean")

ADD_TORCH_UNIT_TEST(Surrogate::EvaluateAndCat::HOST::UQMax::Double ams_convert_and_evaluate_model "2000,8" "2000,8" ${TORCH_MODEL_DIR}/double_cpu_duq_max.pt "duq_max")
ADD_TORCH_UNIT_TEST(Surrogate::EvaluateAndCat::HOST::UQMax::Single ams_convert_and_evaluate_model "2000,8" "2000,8" ${TORCH_MODEL_DIR}/single_cpu_duq_max.pt "duq_max")

if(WITH_CUDA)

#detect datatype
ADD_TORCH_UNIT_TEST(Surrogate::DType::DEVICE::Double::Random ams_verify_dtype double ${TORCH_MODEL_DIR}/double_gpu_random.pt)
ADD_TORCH_UNIT_TEST(Surrogate::DType::DEVICE::Double::Duq_mean ams_verify_dtype double ${TORCH_MODEL_DIR}/double_gpu_duq_mean.pt)
ADD_TORCH_UNIT_TEST(Surrogate::DType::DEVICE::Double::Duq_max ams_verify_dtype double ${TORCH_MODEL_DIR}/double_gpu_duq_max.pt)

ADD_TORCH_UNIT_TEST(Surrogate::DType::DEVICE::Single::Random ams_verify_dtype single ${TORCH_MODEL_DIR}/single_gpu_random.pt)
ADD_TORCH_UNIT_TEST(Surrogate::DType::DEVICE::Single::Duq_mean ams_verify_dtype single ${TORCH_MODEL_DIR}/single_gpu_duq_mean.pt)
ADD_TORCH_UNIT_TEST(Surrogate::DType::DEVICE::Single::Duq_max ams_verify_dtype single ${TORCH_MODEL_DIR}/single_gpu_duq_max.pt)

#detect device
ADD_TORCH_UNIT_TEST(Surrogate::DResource::DEVICE::Double::Random ams_verify_device gpu ${TORCH_MODEL_DIR}/double_gpu_random.pt)
ADD_TORCH_UNIT_TEST(Surrogate::DResource::DEVICE::Double::Duq_mean ams_verify_device gpu ${TORCH_MODEL_DIR}/double_gpu_duq_mean.pt)
ADD_TORCH_UNIT_TEST(Surrogate::DResource::DEVICE::Double::Duq_max ams_verify_device gpu ${TORCH_MODEL_DIR}/double_gpu_duq_max.pt)

ADD_TORCH_UNIT_TEST(Surrogate::DResource::DEVICE::Single::Random ams_verify_device gpu ${TORCH_MODEL_DIR}/single_gpu_random.pt)
ADD_TORCH_UNIT_TEST(Surrogate::DResource::DEVICE::Single::Duq_mean ams_verify_device gpu ${TORCH_MODEL_DIR}/single_gpu_duq_mean.pt)
ADD_TORCH_UNIT_TEST(Surrogate::DResource::DEVICE::Single::Duq_max ams_verify_device gpu ${TORCH_MODEL_DIR}/single_gpu_duq_max.pt)


ADD_TORCH_UNIT_TEST(Surrogate::Evaluate::DEVICE::Random::Double ams_evaluate_model "2000,8" "2000,8" ${TORCH_MODEL_DIR}/double_gpu_random.pt "random")
ADD_TORCH_UNIT_TEST(Surrogate::Evaluate::DEVICE::Random::Single ams_evaluate_model "2000,8" "2000,8" ${TORCH_MODEL_DIR}/single_gpu_random.pt "random")

ADD_TORCH_UNIT_TEST(Surrogate::Evaluate::DEVICE::UQMean::Double ams_evaluate_model "2000,8" "2000,8" ${TORCH_MODEL_DIR}/double_gpu_duq_mean.pt "duq_mean")
ADD_TORCH_UNIT_TEST(Surrogate::Evaluate::DEVICE::UQMean::Single ams_evaluate_model "2000,8" "2000,8" ${TORCH_MODEL_DIR}/single_gpu_duq_mean.pt "duq_mean")

ADD_TORCH_UNIT_TEST(Surrogate::Evaluate::DEVICE::UQMax::Double ams_evaluate_model "2000,8" "2000,8" ${TORCH_MODEL_DIR}/double_gpu_duq_max.pt "duq_max")
ADD_TORCH_UNIT_TEST(Surrogate::Evaluate::DEVICE::UQMax::Single ams_evaluate_model "2000,8" "2000,8" ${TORCH_MODEL_DIR}/single_gpu_duq_max.pt "duq_max")

ADD_TORCH_UNIT_TEST(Surrogate::EvaluateAndCat::DEVICE::Random::Double ams_convert_and_evaluate_model "2000,8" "2000,8" ${TORCH_MODEL_DIR}/double_gpu_random.pt "random")
ADD_TORCH_UNIT_TEST(Surrogate::EvaluateAndCat::DEVICE::Random::Single ams_convert_and_evaluate_model "2000,8" "2000,8" ${TORCH_MODEL_DIR}/single_gpu_random.pt "random")

ADD_TORCH_UNIT_TEST(Surrogate::EvaluateAndCat::DEVICE::UQMean::Double ams_convert_and_evaluate_model "2000,8" "2000,8" ${TORCH_MODEL_DIR}/double_gpu_duq_mean.pt "duq_mean")
ADD_TORCH_UNIT_TEST(Surrogate::EvaluateAndCat::DEVICE::UQMean::Single ams_convert_and_evaluate_model "2000,8" "2000,8" ${TORCH_MODEL_DIR}/single_gpu_duq_mean.pt "duq_mean")

ADD_TORCH_UNIT_TEST(Surrogate::EvaluateAndCat::DEVICE::UQMax::Double ams_convert_and_evaluate_model "2000,8" "2000,8" ${TORCH_MODEL_DIR}/double_gpu_duq_max.pt "duq_max")
ADD_TORCH_UNIT_TEST(Surrogate::EvaluateAndCat::DEVICE::UQMax::Single ams_convert_and_evaluate_model "2000,8" "2000,8" ${TORCH_MODEL_DIR}/single_gpu_duq_max.pt "duq_max")

endif()


