
add_executable(blck_cpu driver.cpp cpu_compute.cpp)

if (WITH_AMS_EXAMPLE_HIP)
  add_executable(blck_hip driver.cpp gpu_compute.cpp)
  set_source_files_properties(gpu_compute.cpp PROPERTIES LANGUAGE HIP)
  target_link_libraries(blck_hip PRIVATE hip::device hip::host)
  target_include_directories(blck_hip
    PRIVATE
      ${PROJECT_SOURCE_DIR}/include
  )
  target_compile_definitions(blck_hip PRIVATE AMS_EXAMPLE_ENABLE_HIP)
elseif(WITH_AMS_EXAMPLE_CUDA)
  add_executable(blck_cuda driver.cpp gpu_compute.cpp)
  set_source_files_properties(gpu_compute.cpp PROPERTIES LANGUAGE CUDA)
  target_link_libraries(blck_cuda PRIVATE CUDA::cudart)
  target_include_directories(blck_cuda
    PRIVATE
      ${PROJECT_SOURCE_DIR}/include
  )
  target_compile_definitions(blck_cuda PRIVATE AMS_EXAMPLE_ENABLE_CUDA)
endif()


target_include_directories(blck_cpu
  PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)


if (WITH_AMS_DOWNLOAD_DATA)
set(BS_ZST "${AMS_INPUT_DIR}/blackscholes_input.bin.zst")
set(BS_BIN "${AMS_INPUT_DIR}/blackscholes_input.bin")

  ams_download_dataset(
    NAME     blackscholes_input_zst
    URL      "https://zenodo.org/records/17992861/files/random_input.bin.zst?download=1"
    SHA256   "b895a9e5206ed60e5fd63529ceeb5e63d823149de4559f578fcab58cb2908ee4"
    OUT_FILE "${BS_ZST}"
  )

# Always define the build-time decompression target (it will run when needed)
ams_add_zstd_decompress_target(
  NAME     inputs_blackscholes
  ZST_FILE "${BS_ZST}"
  OUT_FILE "${BS_BIN}"
)

add_dependencies(blck_cpu inputs_blackscholes)
add_test(
  NAME BLACKSCHOLES::CPU
  COMMAND blck_cpu
          ${BS_BIN}
)
set_tests_properties(BLACKSCHOLES::CPU
  PROPERTIES
    LABELS "benchmark;blackscholes;cpu;no-ams"
    TIMEOUT 600
)

if (WITH_AMS_EXAMPLE_HIP)
add_dependencies(blck_hip inputs_blackscholes)
add_test(
  NAME BLACKSCHOLES::CPU
  COMMAND blck_hip
          ${BS_BIN}
)
set_tests_properties(BLACKSCHOLES::GPU
  PROPERTIES
    LABELS "benchmark;blackscholes;gpu;no-ams"
    TIMEOUT 600
)
elseif(WITH_AMS_EXAMPLE_CUDA)
add_dependencies(blck_cpu inputs_blackscholes)
add_test(
  NAME BLACKSCHOLES::GPU
  COMMAND blck_cuda
          ${BS_BIN}
)
set_tests_properties(BLACKSCHOLES::CPU
  PROPERTIES
    LABELS "benchmark;blackscholes;gpu;no-ams"
    TIMEOUT 600
)
endif()


endif()
