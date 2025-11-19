set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/generate.sh"
  "${CMAKE_CURRENT_SOURCE_DIR}/generate.py"
  "${CMAKE_CURRENT_SOURCE_DIR}/generate_linear_model.py"
  "${CMAKE_CURRENT_SOURCE_DIR}/generate_base_models.py"
)

# Where to drop the .pt files
set(TORCH_MODEL_DIR "${CMAKE_CURRENT_BINARY_DIR}" CACHE STRING "Directory to store torch generated models")

# (keep your GENERATED_*_MODELS lists, unchanged)
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
endif()

# A stamp to indicate models are present
set(MODELS_STAMP "${CMAKE_CURRENT_BINARY_DIR}/.ams_models.stamp")

set(GENERATOR_SCRIPTS
  "${CMAKE_CURRENT_SOURCE_DIR}/generate.sh"
  "${CMAKE_CURRENT_SOURCE_DIR}/generate.py"
  "${CMAKE_CURRENT_SOURCE_DIR}/generate_linear_model.py"
  "${CMAKE_CURRENT_SOURCE_DIR}/generate_base_models.py"
)

# 1. Stamp missing -> regenerate
if (NOT EXISTS "${MODELS_STAMP}")
  set(NEED_REGEN TRUE)
else()
  # 2. Any generator script newer than stamp -> regenerate
  foreach(gen ${GENERATOR_SCRIPTS})
    if (${gen} IS_NEWER_THAN "${MODELS_STAMP}")
      set(NEED_REGEN TRUE)
    endif()
  endforeach()
endif()

# Only run at configure time when models are missing (or scripts changed -> reconfigure)
if (NEED_REGEN)
  message(STATUS "Generating PyTorch models to be used at testing time")
  execute_process(
    COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/generate.sh" "${CMAKE_CURRENT_BINARY_DIR}/"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    RESULT_VARIABLE CPU_GEN_RV
  )
  if (NOT CPU_GEN_RV EQUAL 0)
    message(FATAL_ERROR "CPU model generation failed with code ${CPU_GEN_RV}")
  endif()
  file(WRITE "${MODELS_STAMP}" "ok\n")
endif()

# Optional: a phony target so devs can explicitly rebuild models on demand
add_custom_target(regenerate_ams_models
  COMMAND "${CMAKE_COMMAND}" -E rm -f "${MODELS_STAMP}"
  COMMAND "${CMAKE_COMMAND}" -S "${CMAKE_SOURCE_DIR}" -B "${CMAKE_BINARY_DIR}"
  COMMENT "Forcing model regeneration by reconfiguring the build"
)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/ams_test_simple_models.hpp
  ${CMAKE_CURRENT_BINARY_DIR}/ams_test_simple_models.hpp
  COPYONLY
)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/ams_test_linear_models.hpp
  ${CMAKE_CURRENT_BINARY_DIR}/ams_test_linear_models.hpp
  COPYONLY
)
