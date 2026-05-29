function(ADD_AMS_UNIT_TEST name exec)
  add_test(NAME ${name} COMMAND ${exec}  -s --reporter console)
  set_tests_properties(${name} PROPERTIES LABELS AMS_INTERFACE)
endfunction()


function(BUILD_UNIT_TEST exe source)
  blt_add_executable(NAME ${exe} SOURCES ${source})
  target_compile_features(${exe} PRIVATE cxx_std_17)
  if (ENABLE_CUDA)
    target_link_libraries(${exe} PRIVATE CUDA::cudart)
  elseif (ENABLE_HIP)
    target_link_libraries(${exe} PRIVATE hip::host)
    target_link_libraries(${exe} PRIVATE Threads::Threads)
  endif()
  
  target_include_directories(${exe} PRIVATE ${CMAKE_SOURCE_DIR}/src/AMSlib/)
  target_include_directories(${exe} PRIVATE ${CMAKE_BINARY_DIR}/include/)
  target_include_directories(${exe} PRIVATE ${AMS_TEST_ROOT})
  target_include_directories(${exe} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
  set(catch2_target Catch2::Catch2WithMain)
  # Optional third argument lets tests provide their own main.
  if (${ARGC} GREATER 2)
    set(catch2_target ${ARGV2})
  endif()
  # Optional fourth argument adds that custom main source when needed.
  if (${ARGC} GREATER 3)
    target_sources(${exe} PRIVATE ${ARGV3})
  endif()
  target_link_libraries(${exe} PRIVATE stdc++fs AMS torch ${catch2_target})
  target_link_libraries(${exe} PRIVATE ${AMS_HDF5_LINK_TARGETS})
  target_compile_definitions(${exe} PRIVATE ${AMS_APP_DEFINES} CATCH_CONFIG_PREFIX_ALL)
endfunction()

BUILD_UNIT_TEST(ams_explicit_end_to_end ams_ete.cpp Catch2::Catch2)
ADD_AMS_UNIT_TEST(AMS_EXPLICIT ams_explicit_end_to_end)

BUILD_UNIT_TEST(int_interface int_interface.cpp Catch2::Catch2 ../ams_catch_main.cpp)
ADD_AMS_UNIT_TEST(AMS_INT_INTERFACE int_interface)

BUILD_UNIT_TEST(ams_graph_fallback test_graph_fallback.cpp Catch2::Catch2 ../ams_catch_main.cpp)
ADD_AMS_UNIT_TEST(AMS_GRAPH_FALLBACK ams_graph_fallback)

BUILD_UNIT_TEST(ams_graph_surrogate test_graph_surrogate.cpp Catch2::Catch2 ../ams_catch_main.cpp)
ADD_AMS_UNIT_TEST(AMS_GRAPH_SURROGATE ams_graph_surrogate)

BUILD_UNIT_TEST(ams_graph_mgn_surrogate test_graph_mgn_surrogate.cpp)
if (TARGET generate_mgn_graph_diffusion)
  add_dependencies(ams_graph_mgn_surrogate generate_mgn_graph_diffusion)
endif()
ADD_AMS_UNIT_TEST(AMS_GRAPH_MGN_SURROGATE ams_graph_mgn_surrogate)
