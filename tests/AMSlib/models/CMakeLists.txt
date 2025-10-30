# Output files for generated models (example: .pt files)

set(TORCH_MODEL_DIR "${CMAKE_CURRENT_BINARY_DIR}" CACHE STRING "Directory to store torch generated models")

set(GENERATED_CPU_MODELS
  ${CMAKE_CURRENT_BINARY_DIR}/double_cpu_duq_max.pt  
  ${CMAKE_CURRENT_BINARY_DIR}/double_cpu_duq_mean.pt  
  ${CMAKE_CURRENT_BINARY_DIR}/double_cpu_random.pt  
  ${CMAKE_CURRENT_BINARY_DIR}/single_cpu_duq_max.pt  
  ${CMAKE_CURRENT_BINARY_DIR}/single_cpu_duq_mean.pt  
  ${CMAKE_CURRENT_BINARY_DIR}/single_cpu_random.pt

  ${CMAKE_CURRENT_BINARY_DIR}/linear_traced_double_cpu_duq_max.pt  
  ${CMAKE_CURRENT_BINARY_DIR}/linear_traced_double_cpu_duq_mean.pt  
  ${CMAKE_CURRENT_BINARY_DIR}/linear_traced_double_cpu_random.pt  
  ${CMAKE_CURRENT_BINARY_DIR}/linear_traced_single_cpu_duq_max.pt  
  ${CMAKE_CURRENT_BINARY_DIR}/linear_traced_single_cpu_duq_mean.pt  
  ${CMAKE_CURRENT_BINARY_DIR}/linear_traced_single_cpu_random.pt

  ${CMAKE_CURRENT_BINARY_DIR}/linear_scripted_double_cpu_duq_max.pt  
  ${CMAKE_CURRENT_BINARY_DIR}/linear_scripted_double_cpu_duq_mean.pt  
  ${CMAKE_CURRENT_BINARY_DIR}/linear_scripted_double_cpu_random.pt  
  ${CMAKE_CURRENT_BINARY_DIR}/linear_scripted_single_cpu_duq_max.pt  
  ${CMAKE_CURRENT_BINARY_DIR}/linear_scripted_single_cpu_duq_mean.pt  
  ${CMAKE_CURRENT_BINARY_DIR}/linear_scripted_single_cpu_random.pt


)

# Custom command to generate models
add_custom_command(
  OUTPUT ${GENERATED_CPU_MODELS}
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/generate.sh ${CMAKE_CURRENT_BINARY_DIR}/ "cpu"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/generate.sh;${CMAKE_CURRENT_SOURCE_DIR}/generate.py;${CMAKE_CURRENT_SOURCE_DIR}/generate_linear_model.py" 
    COMMENT "Generating CPU PyTorch models..."
)

# Define a target to generate models
add_custom_target(generate_cpu_models ALL
  DEPENDS ${GENERATED_CPU_MODELS}
    COMMENT "Generate cpu models before ctests..."
)

if (WITH_CUDA OR WITH_HIP)
set(GENERATED_GPU_MODELS
  ${CMAKE_CURRENT_BINARY_DIR}/double_gpu_duq_max.pt  
  ${CMAKE_CURRENT_BINARY_DIR}/double_gpu_duq_mean.pt  
  ${CMAKE_CURRENT_BINARY_DIR}/double_gpu_random.pt  
  ${CMAKE_CURRENT_BINARY_DIR}/single_gpu_duq_max.pt  
  ${CMAKE_CURRENT_BINARY_DIR}/single_gpu_duq_mean.pt  
  ${CMAKE_CURRENT_BINARY_DIR}/single_gpu_random.pt

  ${CMAKE_CURRENT_BINARY_DIR}/linear_traced_double_gpu_duq_max.pt  
  ${CMAKE_CURRENT_BINARY_DIR}/linear_traced_double_gpu_duq_mean.pt  
  ${CMAKE_CURRENT_BINARY_DIR}/linear_traced_double_gpu_random.pt  
  ${CMAKE_CURRENT_BINARY_DIR}/linear_traced_single_gpu_duq_max.pt  
  ${CMAKE_CURRENT_BINARY_DIR}/linear_traced_single_gpu_duq_mean.pt  
  ${CMAKE_CURRENT_BINARY_DIR}/linear_traced_single_gpu_random.pt

  ${CMAKE_CURRENT_BINARY_DIR}/linear_scripted_double_gpu_duq_max.pt  
  ${CMAKE_CURRENT_BINARY_DIR}/linear_scripted_double_gpu_duq_mean.pt  
  ${CMAKE_CURRENT_BINARY_DIR}/linear_scripted_double_gpu_random.pt  
  ${CMAKE_CURRENT_BINARY_DIR}/linear_scripted_single_gpu_duq_max.pt  
  ${CMAKE_CURRENT_BINARY_DIR}/linear_scripted_single_gpu_duq_mean.pt  
  ${CMAKE_CURRENT_BINARY_DIR}/linear_scripted_single_gpu_random.pt
)

# Custom command to generate models
message(INFO "Device is ${device}")
add_custom_command(
  OUTPUT ${GENERATED_GPU_MODELS}
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/generate.sh ${CMAKE_CURRENT_BINARY_DIR}/ "gpu" 
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/generate.sh;${CMAKE_CURRENT_SOURCE_DIR}/generate.py" 
    COMMENT "Generating GPU PyTorch models..."
)

# Define a target to generate models
add_custom_target(generate_gpu_models ALL
  DEPENDS ${GENERATED_GPU_MODELS}
    COMMENT "Generate gpu models before ctests..."
)
endif()

