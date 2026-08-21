set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/generate.sh"
  "${CMAKE_CURRENT_SOURCE_DIR}/generate.py"
  "${CMAKE_CURRENT_SOURCE_DIR}/generate_linear_model.py"
  "${CMAKE_CURRENT_SOURCE_DIR}/generate_base_models.py"
  "${CMAKE_CURRENT_SOURCE_DIR}/generate_graph_models.py"
  "${CMAKE_CURRENT_SOURCE_DIR}/generate_mgn_graph_diffusion.py"
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
  ${CMAKE_CURRENT_BINARY_DIR}/homogeneous_graph.pt
  ${CMAKE_CURRENT_BINARY_DIR}/heterogeneous_graph.pt
  ${CMAKE_CURRENT_BINARY_DIR}/homogeneous_graph_bad_key.pt
  ${CMAKE_CURRENT_BINARY_DIR}/homogeneous_graph_bad_shape.pt
)

if (ENABLE_CUDA OR ENABLE_HIP)
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
  "${CMAKE_CURRENT_SOURCE_DIR}/generate_graph_models.py"
  "${CMAKE_CURRENT_SOURCE_DIR}/ams_model.py"
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

find_program(AMS_PYTHON_EXECUTABLE NAMES python3 python REQUIRED)

# MGN diffusion artifacts generated by the opt-in training workflow stay in the
# build tree. Normal parity testing reads the checked-in model and self-contained
# fixture manifest from MGN_GRAPH_DIFFUSION_FIXTURE_DIR instead.
set(MGN_GRAPH_DIFFUSION_DIR
  "${CMAKE_CURRENT_BINARY_DIR}/mgn_graph_diffusion"
)
set(MGN_GRAPH_DIFFUSION_FIXTURE_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/mgn_graph_diffusion"
)
set(MGN_GRAPH_DIFFUSION_CHECKPOINT
  "${MGN_GRAPH_DIFFUSION_DIR}/mgn_graph_diffusion_checkpoint.pt"
)
set(MGN_GRAPH_DIFFUSION_MODEL
  "${MGN_GRAPH_DIFFUSION_DIR}/mgn_graph_diffusion.pt"
)
set(MGN_GRAPH_DIFFUSION_FEASIBILITY_MODEL
  "${MGN_GRAPH_DIFFUSION_DIR}/mgn_graph_diffusion_feasibility.pt"
)
set(MGN_GRAPH_DIFFUSION_MANIFEST
  "${MGN_GRAPH_DIFFUSION_DIR}/fixtures.json"
)
set(MGN_GRAPH_DIFFUSION_METRICS
  "${MGN_GRAPH_DIFFUSION_DIR}/training_metrics.json"
)
add_custom_command(
  OUTPUT
    "${MGN_GRAPH_DIFFUSION_CHECKPOINT}"
    "${MGN_GRAPH_DIFFUSION_MODEL}"
    "${MGN_GRAPH_DIFFUSION_MANIFEST}"
  BYPRODUCTS
    "${MGN_GRAPH_DIFFUSION_FEASIBILITY_MODEL}"
    "${MGN_GRAPH_DIFFUSION_METRICS}"
  COMMAND
    "${AMS_PYTHON_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/generate_mgn_graph_diffusion.py"
    --out-dir "${MGN_GRAPH_DIFFUSION_DIR}"
    --mode feasibility
  COMMAND
    "${AMS_PYTHON_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/generate_mgn_graph_diffusion.py"
    --out-dir "${MGN_GRAPH_DIFFUSION_DIR}"
    --mode train
  COMMAND
    "${AMS_PYTHON_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/generate_mgn_graph_diffusion.py"
    --out-dir "${MGN_GRAPH_DIFFUSION_DIR}"
    --mode fixtures
  DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/generate_mgn_graph_diffusion.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/ams_model.py"
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  COMMENT "Generating pure-Torch MGN graph diffusion model and fixtures"
  VERBATIM
)

add_custom_target(generate_mgn_graph_diffusion
  DEPENDS
    "${MGN_GRAPH_DIFFUSION_CHECKPOINT}"
    "${MGN_GRAPH_DIFFUSION_MODEL}"
    "${MGN_GRAPH_DIFFUSION_MANIFEST}"
)

# The deployability gate remains part of normal testing. It also checks that the
# committed manifest still describes the current model and fixture constants.
add_test(
  NAME MGN_DIFFUSION_FEASIBILITY
  COMMAND
    "${AMS_PYTHON_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/generate_mgn_graph_diffusion.py"
    --out-dir "${MGN_GRAPH_DIFFUSION_DIR}"
    --fixture-dir "${MGN_GRAPH_DIFFUSION_FIXTURE_DIR}"
    --mode feasibility
)
set_tests_properties(MGN_DIFFUSION_FEASIBILITY PROPERTIES
  LABELS MGN_DIFFUSION
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

if (ENABLE_TRAINING_TESTS)
  # CTest fixture requirements make failures stop the opt-in generation chain:
  # feasibility -> training -> fixture export.
  set_tests_properties(MGN_DIFFUSION_FEASIBILITY PROPERTIES
    FIXTURES_SETUP MGN_DIFFUSION_FEASIBLE
  )

  add_test(
    NAME MGN_DIFFUSION_TRAIN
    COMMAND
      "${AMS_PYTHON_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/generate_mgn_graph_diffusion.py"
      --out-dir "${MGN_GRAPH_DIFFUSION_DIR}"
      --mode train
  )
  set_tests_properties(MGN_DIFFUSION_TRAIN PROPERTIES
    ENVIRONMENT "PYTHONUNBUFFERED=1"
    FIXTURES_REQUIRED MGN_DIFFUSION_FEASIBLE
    FIXTURES_SETUP MGN_DIFFUSION_TRAINED
    LABELS MGN_DIFFUSION
    TIMEOUT 3600
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  )

  add_test(
    NAME MGN_DIFFUSION_FIXTURES
    COMMAND
      "${AMS_PYTHON_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/generate_mgn_graph_diffusion.py"
      --out-dir "${MGN_GRAPH_DIFFUSION_DIR}"
      --mode fixtures
  )
  set_tests_properties(MGN_DIFFUSION_FIXTURES PROPERTIES
    FIXTURES_REQUIRED MGN_DIFFUSION_TRAINED
    LABELS MGN_DIFFUSION
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  )
endif()
