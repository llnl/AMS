function(ADD_API_UNIT_TEST gname name cmd)
  add_test(NAME ${name} COMMAND bash -c "${cmd}")
  set_tests_properties(${name} PROPERTIES LABELS ${gname})
endfunction()


function(JSON_TESTS db_type)
  set(FS_PATH "${CMAKE_CURRENT_BINARY_DIR}")
  set(AMS_DB_TEST_TYPE ${db_type})
  set(JSON_FP "${CMAKE_CURRENT_BINARY_DIR}/${db_type}.json")
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/json_configs/env_2_models_fs_rand_uq.json.in" "${JSON_FP}" @ONLY)
  
  # Tests Random models with different percentages both models store to file
  ADD_API_UNIT_TEST(APIEnvAPI AMS::ENV::Random10::Random50::Double::DB::${db_type}::HOST "AMS_OBJECTS=${JSON_FP} ${CMAKE_CURRENT_BINARY_DIR}/ams_end_to_end_env  0 8 8 \"double\" 1 128 app_random_10 app_random_50;AMS_OBJECTS=${JSON_FP} python3 ${CMAKE_CURRENT_SOURCE_DIR}/verify_ete.py 0 8 8 \"double\" 128 app_random_10 app_random_50")


  # Tests delta-uq models with different aggregation both models store to file
  ADD_API_UNIT_TEST(APIEnvAPI AMS::ENV::DuqMean::DuqMax::Double::DB::${db_type}::HOST "AMS_OBJECTS=${JSON_FP} ${CMAKE_CURRENT_BINARY_DIR}/ams_end_to_end_env  0 8 8 \"double\" 1 128 app_uq_mean app_uq_max;AMS_OBJECTS=${JSON_FP} python3 ${CMAKE_CURRENT_SOURCE_DIR}/verify_ete.py 0 8 8 \"double\" 128 app_uq_mean app_uq_max")


  # Tests detla uq model with a random uq model both models store to files
  ADD_API_UNIT_TEST(APIEnvAPI AMS::ENV::Random::DuqMax::Double::DB::${db_type}::HOST "AMS_OBJECTS=${JSON_FP} ${CMAKE_CURRENT_BINARY_DIR}/ams_end_to_end_env  0 8 8 \"double\" 1 128 app_random_10 app_uq_max;AMS_OBJECTS=${JSON_FP} python3 ${CMAKE_CURRENT_SOURCE_DIR}/verify_ete.py 0 8 8 \"double\" 128 app_random_10 app_uq_max")

  # Tests detla uq model with no model. uq model both store to files
  ADD_API_UNIT_TEST(APIEnvAPI AMS::ENV::Random::NoModel::Double::DB::${db_type}::HOST "AMS_OBJECTS=${JSON_FP} ${CMAKE_CURRENT_BINARY_DIR}/ams_end_to_end_env  0 8 8 \"double\" 1 128 app_random_10 app_no_model;AMS_OBJECTS=${JSON_FP} python3 ${CMAKE_CURRENT_SOURCE_DIR}/verify_ete.py 0 8 8 \"double\" 128 app_random_10 app_no_model")

  # Tests 2 delta uq models with no deb . uq model both store to files
  ADD_API_UNIT_TEST(APIEnvAPI AMS::ENV::DuqMean::DuqMax::Double::NODB::${db_type}::HOST "AMS_OBJECTS=${JSON_FP} ${CMAKE_CURRENT_BINARY_DIR}/ams_end_to_end_env  0 8 8 \"double\" 1 128 app_uq_mean_ndb app_uq_max_ndb;AMS_OBJECTS=${JSON_FP} python3 ${CMAKE_CURRENT_SOURCE_DIR}/verify_ete.py 0 8 8 \"double\" 128 app_uq_mean_ndb app_uq_max_ndb")

  # Tests null models null dbs 
  ADD_API_UNIT_TEST(APIEnvAPI AMS::ENV::None::None::Double::NODB::${db_type}::HOST "AMS_OBJECTS=${JSON_FP} ${CMAKE_CURRENT_BINARY_DIR}/ams_end_to_end_env  0 8 8 \"double\" 1 128 app_no_model_no_db  app_no_model_no_db ;AMS_OBJECTS=${JSON_FP} python3 ${CMAKE_CURRENT_SOURCE_DIR}/verify_ete.py 0 8 8 \"double\" 128 app_no_model_no_db app_no_model_no_db")

  unset(AMS_DB_TEST_TYPE)
  unset(JSON_FP)
endfunction()


function(CHECK_RMQ_CONFIG file)
  # Read the JSON file.
  file(READ ${file} MY_JSON_STRING)
  message(STATUS "RabbitMQ config ${file}")

  string(JSON DB_CONF GET ${MY_JSON_STRING} db)
  string(JSON DB_CONF GET ${DB_CONF} rmq_config)
  string(JSON RMQ_HOST GET ${DB_CONF} "service-host")
  string(JSON RMQ_PORT GET ${DB_CONF} "service-port")

  if(NOT "${RMQ_HOST}" STREQUAL "" AND NOT "${RMQ_PORT}" STREQUAL "0")
  message(STATUS "RabbitMQ config ${file}: ${RMQ_HOST}:${RMQ_PORT}")
  else()
    message(WARNING "RabbitMQ config file ${file} looks empty! Make sure to fill these fields before running the tests")
  endif()
endfunction()


function(INTEGRATION_TEST_ENV)
  JSON_TESTS("hdf5")
  set(JSON_FP "${CMAKE_CURRENT_BINARY_DIR}/hdf5.json")
  unset(JSON_FP)
endfunction()

function(INTEGRATION_TEST_RMQ)
    if(EXISTS "${CMAKE_CURRENT_BINARY_DIR}/rmq.json")
      # If file exists we do not overwrite it
      message(STATUS "Ctest will use ${CMAKE_CURRENT_BINARY_DIR}/rmq.json as RabbitMQ configuration for testing. Make sure RabbitMQ parameters are valid.")
    else()
      message(STATUS "Copying empty configuration to ${CMAKE_CURRENT_BINARY_DIR}/rmq.json")
      configure_file("${CMAKE_CURRENT_SOURCE_DIR}/json_configs/rmq.json.in" "rmq.json" @ONLY)
    endif()
    set(JSON_FP "${CMAKE_CURRENT_BINARY_DIR}/rmq.json")
    CHECK_RMQ_CONFIG(${JSON_FP})
    add_test(NAME AMSEndToEndFromJSON::NoModel::Double::DB::rmq::HOST COMMAND bash -c "AMS_OBJECTS=${JSON_FP} ${CMAKE_CURRENT_BINARY_DIR}/ams_rmq 0 8 8 \"double\" 2 128; AMS_OBJECTS=${JSON_FP} python3 ${CMAKE_CURRENT_SOURCE_DIR}/verify_rmq.py 0 8 8 \"double\" 2 128")
endfunction()

function (INTEGRATION_TEST)
  #######################################################
  # TEST: output format 
  # UQ: Random
  #######################################################

  ADD_API_UNIT_TEST(API_DIRECT_QUERY AMS::API::ModelDefine::Random::Double::DB::OnlyPhysics::None::HOST "${CMAKE_CURRENT_BINARY_DIR}/ams_end_to_end 0 8 8 ${TORCH_MODEL_DIR}/linear_scripted_single_cpu_random.pt \"double\" \"random\" 0.0 1 128 \"none\" \"./\"; python3 ${CMAKE_CURRENT_SOURCE_DIR}/verify_ete.py 0 8 8 ${CMAKE_CURRENT_SOURCE_DIR}/linear_scripted_cpu.pt \"double\" \"random\" 0.0 1 128 \"none\" \"./\"")
  ADD_API_UNIT_TEST(API_DIRECT_QUERY AMS::API::ModelDefine::Random::Double::DB::OnlyModel::None::HOST "${CMAKE_CURRENT_BINARY_DIR}/ams_end_to_end 0 8 8 ${TORCH_MODEL_DIR}/linear_scripted_single_cpu_random.pt \"double\" \"random\" 0.0 1 128 \"none\" \"./\"; python3 ${CMAKE_CURRENT_SOURCE_DIR}/verify_ete.py 0 8 8 ${CMAKE_CURRENT_SOURCE_DIR}/linear_scripted_cpu.pt \"double\" \"random\" 1.0 1 128 \"none\" \"./\"")


  if (WITH_HDF5)
  #######################################################
  # TEST: hdf5 output format 
  # UQ: Random
  # 3 Thresholds: 0, 0.5, 1.0
  # precision: double
  #######################################################

    ADD_API_UNIT_TEST(API_DIRECT_QUERY AMS::API::ModelDefine::Random::Double::DB::OnlyPhysics::HDF5::HOST "${CMAKE_CURRENT_BINARY_DIR}/ams_end_to_end 0 8 8 ${TORCH_MODEL_DIR}/linear_scripted_single_cpu_random.pt \"double\" \"random\" 0.0 1 128 \"hdf5\" \"./\"; python3 ${CMAKE_CURRENT_SOURCE_DIR}/verify_ete.py 0 8 8 ${CMAKE_CURRENT_SOURCE_DIR}/linear_scripted_cpu.pt \"double\" \"random\" 0.0 1 128 \"hdf5\" \"./\"")
    ADD_API_UNIT_TEST(API_DIRECT_QUERY AMS::API::ModelDefine::Random::Double::DB::OnlyModel::HDF5::HOST "${CMAKE_CURRENT_BINARY_DIR}/ams_end_to_end 0 8 8 ${TORCH_MODEL_DIR}/linear_scripted_single_cpu_random.pt \"double\" \"random\" 1.0 1 128 \"hdf5\" \"./\"; python3 ${CMAKE_CURRENT_SOURCE_DIR}/verify_ete.py 0 8 8 ${CMAKE_CURRENT_SOURCE_DIR}/linear_scripted_cpu.pt \"double\" \"random\" 1.0 1 128 \"hdf5\" \"./\"")
    ADD_API_UNIT_TEST(API_DIRECT_QUERY AMS::API::ModelDefine::Random::Double::DB::HALF::HDF5::HOST "${CMAKE_CURRENT_BINARY_DIR}/ams_end_to_end 0 8 8 ${TORCH_MODEL_DIR}/linear_scripted_single_cpu_random.pt \"double\" \"random\" 0.5 1 128 \"hdf5\" \"./\"; python3 ${CMAKE_CURRENT_SOURCE_DIR}/verify_ete.py 0 8 8 ${CMAKE_CURRENT_SOURCE_DIR}/linear_scripted_cpu.pt \"double\" \"random\" 0.5 1 128 \"hdf5\" \"./\"")

  #######################################################
  # TEST: hdf5 output format 
  # UQ: Random
  # 3 Thresholds: 0, 0.5, 1.0
  # precision: single
  #######################################################


    ADD_API_UNIT_TEST(API_DIRECT_QUERY AMS::API::ModelDefine::Random::Single::DB::OnlyPhysics::HDF5::HOST "${CMAKE_CURRENT_BINARY_DIR}/ams_end_to_end 0 8 8 ${TORCH_MODEL_DIR}/linear_scripted_single_cpu_random.pt \"float\" \"random\" 0.0 1 128 \"hdf5\" \"./\"; python3 ${CMAKE_CURRENT_SOURCE_DIR}/verify_ete.py 0 8 8 ${CMAKE_CURRENT_SOURCE_DIR}/linear_scripted_cpu.pt \"float\" \"random\" 0.0 1 128 \"hdf5\" \"./\"")
    ADD_API_UNIT_TEST(API_DIRECT_QUERY AMS::API::ModelDefine::Random::Single::DB::OnlyModel::HDF5::HOST "${CMAKE_CURRENT_BINARY_DIR}/ams_end_to_end 0 8 8 ${TORCH_MODEL_DIR}/linear_scripted_single_cpu_random.pt \"float\" \"random\" 1.0 1 128 \"hdf5\" \"./\"; python3 ${CMAKE_CURRENT_SOURCE_DIR}/verify_ete.py 0 8 8 ${CMAKE_CURRENT_SOURCE_DIR}/linear_scripted_cpu.pt \"float\" \"random\" 1.0 1 128 \"hdf5\" \"./\"")
    ADD_API_UNIT_TEST(API_DIRECT_QUERY AMS::API::ModelDefine::Random::Single::DB::HALF::HDF5::HOST "${CMAKE_CURRENT_BINARY_DIR}/ams_end_to_end 0 8 8 ${TORCH_MODEL_DIR}/linear_scripted_single_cpu_random.pt \"float\" \"random\" 0.5 1 128 \"hdf5\" \"./\"; python3 ${CMAKE_CURRENT_SOURCE_DIR}/verify_ete.py 0 8 8 ${CMAKE_CURRENT_SOURCE_DIR}/linear_scripted_cpu.pt \"float\" \"random\" 0.5 1 128 \"hdf5\" \"./\"")
  endif()


  #######################################################
  # TEST: hdf5 output format 
  # UQ: deltauq-mean
  # 3 Thresholds: 0, 0.5, 1.0
  # precision: double
  #######################################################
  ADD_API_UNIT_TEST(API_DIRECT_QUERY AMS::API::ModelDefine::DeltaUQMean::Double::DB::OnlyPhysics::HDF5::HOST "${CMAKE_CURRENT_BINARY_DIR}/ams_end_to_end 0 8 8 ${TORCH_MODEL_DIR}/double_cpu_duq_mean.pt \"double\" \"deltaUQ (mean)\" 0.0 1 128 \"hdf5\" \"./\"; python3 ${CMAKE_CURRENT_SOURCE_DIR}/verify_ete.py 0 8 8 ${TORCH_MODEL_DIR}/double_cpu_duq_mean.pt \"double\" \"deltaUQ (mean)\" 0.0 1 128 \"hdf5\" \"./\"")
  ADD_API_UNIT_TEST(API_DIRECT_QUERY AMS::API::ModelDefine::DeltaUQMean::Double::DB::OnlyModel::HDF5::HOST "${CMAKE_CURRENT_BINARY_DIR}/ams_end_to_end 0 8 8 ${TORCH_MODEL_DIR}/double_cpu_duq_mean.pt \"double\" \"deltaUQ (mean)\" 1.0 1 128 \"hdf5\" \"./\"; python3 ${CMAKE_CURRENT_SOURCE_DIR}/verify_ete.py 0 8 8 ${TORCH_MODEL_DIR}/double_cpu_duq_mean.pt \"double\" \"deltaUQ (mean)\" 1.0 1 128 \"hdf5\" \"./\"")
  ADD_API_UNIT_TEST(API_DIRECT_QUERY AMS::API::ModelDefine::DeltaUQMean::Double::DB::HALF::HDF5::HOST "${CMAKE_CURRENT_BINARY_DIR}/ams_end_to_end 0 8 8 ${TORCH_MODEL_DIR}/double_cpu_duq_mean.pt \"double\" \"deltaUQ (mean)\" 0.5 1 128 \"hdf5\" \"./\"; python3 ${CMAKE_CURRENT_SOURCE_DIR}/verify_ete.py 0 8 8 ${TORCH_MODEL_DIR}/double_cpu_duq_mean.pt \"double\" \"deltaUQ (mean)\" 0.5 1 128 \"hdf5\" \"./\"")

  #######################################################
  # TEST: hdf5 output format 
  # UQ: deltauq-mean
  # 3 Thresholds: 0, 0.5, 1.0
  # precision: single
  #######################################################
  ADD_API_UNIT_TEST(API_DIRECT_QUERY AMS::API::ModelDefine::DeltaUQMean::Single::DB::OnlyPhysics::HDF5::HOST "${CMAKE_CURRENT_BINARY_DIR}/ams_end_to_end 0 8 8 ${TORCH_MODEL_DIR}/single_cpu_duq_mean.pt \"float\" \"deltaUQ (mean)\" 0.0 1 128 \"hdf5\" \"./\"; python3 ${CMAKE_CURRENT_SOURCE_DIR}/verify_ete.py 0 8 8 ${TORCH_MODEL_DIR}/single_cpu_duq_mean.pt \"float\" \"deltaUQ (mean)\" 0.0 1 128 \"hdf5\" \"./\"")
  ADD_API_UNIT_TEST(API_DIRECT_QUERY AMS::API::ModelDefine::DeltaUQMean::Single::DB::OnlyModel::HDF5::HOST "${CMAKE_CURRENT_BINARY_DIR}/ams_end_to_end 0 8 8 ${TORCH_MODEL_DIR}/single_cpu_duq_mean.pt \"float\" \"deltaUQ (mean)\" 1.0 1 128 \"hdf5\" \"./\"; python3 ${CMAKE_CURRENT_SOURCE_DIR}/verify_ete.py 0 8 8 ${TORCH_MODEL_DIR}/single_cpu_duq_mean.pt \"float\" \"deltaUQ (mean)\" 1.0 1 128 \"hdf5\" \"./\"")
  ADD_API_UNIT_TEST(API_DIRECT_QUERY AMS::API::ModelDefine::DeltaUQMean::Single::DB::HALF::HDF5::HOST "${CMAKE_CURRENT_BINARY_DIR}/ams_end_to_end 0 8 8 ${TORCH_MODEL_DIR}/single_cpu_duq_mean.pt \"float\" \"deltaUQ (mean)\" 0.5 1 128 \"hdf5\" \"./\"; python3 ${CMAKE_CURRENT_SOURCE_DIR}/verify_ete.py 0 8 8 ${TORCH_MODEL_DIR}/single_cpu_duq_mean.pt \"float\" \"deltaUQ (mean)\" 0.5 1 128 \"hdf5\" \"./\"")

  #######################################################
  # TEST: hdf5 output format 
  # UQ: deltauq-max
  # 3 Thresholds: 0, 0.5, 1.0
  # precision: double
  #######################################################
  ADD_API_UNIT_TEST(API_DIRECT_QUERY AMS::API::ModelDefine::DeltaUQMax::Double::DB::OnlyPhysics::HDF5::HOST "${CMAKE_CURRENT_BINARY_DIR}/ams_end_to_end 0 8 8 ${TORCH_MODEL_DIR}/double_cpu_duq_max.pt \"double\" \"deltaUQ (max)\" 0.0 1 128 \"hdf5\" \"./\"; python3 ${CMAKE_CURRENT_SOURCE_DIR}/verify_ete.py 0 8 8 ${TORCH_MODEL_DIR}/double_cpu_duq_max.pt \"double\" \"deltaUQ (max)\" 0.0 1 128 \"hdf5\" \"./\"")
  ADD_API_UNIT_TEST(API_DIRECT_QUERY AMS::API::ModelDefine::DeltaUQMax::Double::DB::OnlyModel::HDF5::HOST "${CMAKE_CURRENT_BINARY_DIR}/ams_end_to_end 0 8 8 ${TORCH_MODEL_DIR}/double_cpu_duq_max.pt \"double\" \"deltaUQ (max)\" 1.0 1 128 \"hdf5\" \"./\"; python3 ${CMAKE_CURRENT_SOURCE_DIR}/verify_ete.py 0 8 8 ${TORCH_MODEL_DIR}/double_cpu_duq_max.pt \"double\" \"deltaUQ (max)\" 1.0 1 128 \"hdf5\" \"./\"")
  ADD_API_UNIT_TEST(API_DIRECT_QUERY AMS::API::ModelDefine::DeltaUQMax::Double::DB::HALF::HDF5::HOST "${CMAKE_CURRENT_BINARY_DIR}/ams_end_to_end 0 8 8 ${TORCH_MODEL_DIR}/double_cpu_duq_max.pt \"double\" \"deltaUQ (max)\" 0.5 1 128 \"hdf5\" \"./\"; python3 ${CMAKE_CURRENT_SOURCE_DIR}/verify_ete.py 0 8 8 ${TORCH_MODEL_DIR}/double_cpu_duq_max.pt \"double\" \"deltaUQ (max)\" 0.5 1 128 \"hdf5\" \"./\"")

 #######################################################
  # TEST: hdf5 output format 
  # UQ: deltauq-max
  # 3 Thresholds: 0, 0.5, 1.0
  # precision: single
  #######################################################
  ADD_API_UNIT_TEST(API_DIRECT_QUERY AMS::API::ModelDefine::DeltaUQMax::Single::DB::OnlyPhysics::HDF5::HOST "${CMAKE_CURRENT_BINARY_DIR}/ams_end_to_end 0 8 8 ${TORCH_MODEL_DIR}/single_cpu_duq_max.pt \"float\" \"deltaUQ (max)\" 0.0 1 128 \"hdf5\" \"./\"; python3 ${CMAKE_CURRENT_SOURCE_DIR}/verify_ete.py 0 8 8 ${TORCH_MODEL_DIR}/single_cpu_duq_max.pt \"float\" \"deltaUQ (max)\" 0.0 1 128 \"hdf5\" \"./\"")
  ADD_API_UNIT_TEST(API_DIRECT_QUERY AMS::API::ModelDefine::DeltaUQMax::Single::DB::OnlyModel::HDF5::HOST "${CMAKE_CURRENT_BINARY_DIR}/ams_end_to_end 0 8 8 ${TORCH_MODEL_DIR}/single_cpu_duq_max.pt \"float\" \"deltaUQ (max)\" 1.0 1 128 \"hdf5\" \"./\"; python3 ${CMAKE_CURRENT_SOURCE_DIR}/verify_ete.py 0 8 8 ${TORCH_MODEL_DIR}/single_cpu_duq_max.pt \"float\" \"deltaUQ (max)\" 1.0 1 128 \"hdf5\" \"./\"")
  ADD_API_UNIT_TEST(API_DIRECT_QUERY AMS::API::ModelDefine::DeltaUQMax::Single::DB::HALF::HDF5::HOST "${CMAKE_CURRENT_BINARY_DIR}/ams_end_to_end 0 8 8 ${TORCH_MODEL_DIR}/single_cpu_duq_max.pt \"float\" \"deltaUQ (max)\" 0.5 1 128 \"hdf5\" \"./\"; python3 ${CMAKE_CURRENT_SOURCE_DIR}/verify_ete.py 0 8 8 ${TORCH_MODEL_DIR}/single_cpu_duq_max.pt \"float\" \"deltaUQ (max)\" 0.5 1 128 \"hdf5\" \"./\"")


  # Random model with inout arguments that represent 2D data.
  ADD_API_UNIT_TEST(API_DIRECT_QUERY AMS::API::ModelDefine::Inout2D::Random::Double::DB::OnlyPhysics::HDF5::HOST "${CMAKE_CURRENT_BINARY_DIR}/ams_inout_2d 0 8 8 ${TORCH_MODEL_DIR}/linear_scripted_single_cpu_random.pt \"double\" \"random\" 0.0 1 128 \"hdf5\" \"./\"; python3 ${CMAKE_CURRENT_SOURCE_DIR}/verify_ete.py 0 8 8 ${CMAKE_CURRENT_SOURCE_DIR}/linear_scripted_cpu.pt \"double\" \"random\" 0.0 1 128 \"hdf5\" \"./\"")
  ADD_API_UNIT_TEST(API_DIRECT_QUERY AMS::API::ModelDefine::Inout2D::Random::Double::DB::OnlyModel::HDF5::HOST "${CMAKE_CURRENT_BINARY_DIR}/ams_inout_2d 0 8 8 ${TORCH_MODEL_DIR}/linear_scripted_single_cpu_random.pt \"double\" \"random\" 1.0 1 128 \"hdf5\" \"./\"; python3 ${CMAKE_CURRENT_SOURCE_DIR}/verify_ete.py 0 8 8 ${CMAKE_CURRENT_SOURCE_DIR}/linear_scripted_cpu.pt \"double\" \"random\" 1.0 1 128 \"hdf5\" \"./\"")
  ADD_API_UNIT_TEST(API_DIRECT_QUERY AMS::API::ModelDefine::Inout2D::Random::Double::DB::HALF::HDF5::HOST "${CMAKE_CURRENT_BINARY_DIR}/ams_inout_2d 0 8 8 ${TORCH_MODEL_DIR}/linear_scripted_single_cpu_random.pt \"double\" \"random\" 0.5 1 128 \"hdf5\" \"./\"; python3 ${CMAKE_CURRENT_SOURCE_DIR}/verify_ete.py 0 8 8 ${CMAKE_CURRENT_SOURCE_DIR}/linear_scripted_cpu.pt \"double\" \"random\" 0.5 1 128 \"hdf5\" \"./\"")

# Random model with inout arguments that represent 2D data.
  ADD_API_UNIT_TEST(API_DIRECT_QUERY AMS::API::ModelDefine::Broadcast::Random::Double::DB::OnlyPhysics::HDF5::HOST "${CMAKE_CURRENT_BINARY_DIR}/ams_ete_broadcast 0 8 8 ${TORCH_MODEL_DIR}/linear_scripted_single_cpu_random.pt \"double\" \"random\" 0.0 1 128 \"hdf5\" \"./\"; python3 ${CMAKE_CURRENT_SOURCE_DIR}/verify_ete.py 0 8 8 ${CMAKE_CURRENT_SOURCE_DIR}/linear_scripted_cpu.pt \"double\" \"random\" 0.0 1 128 \"hdf5\" \"./\"")
  ADD_API_UNIT_TEST(API_DIRECT_QUERY AMS::API::ModelDefine::Broadcast::Random::Double::DB::OnlyModel::HDF5::HOST "${CMAKE_CURRENT_BINARY_DIR}/ams_ete_broadcast 0 8 8 ${TORCH_MODEL_DIR}/linear_scripted_single_cpu_random.pt \"double\" \"random\" 1.0 1 128 \"hdf5\" \"./\"; python3 ${CMAKE_CURRENT_SOURCE_DIR}/verify_ete.py 0 8 8 ${CMAKE_CURRENT_SOURCE_DIR}/linear_scripted_cpu.pt \"double\" \"random\" 1.0 1 128 \"hdf5\" \"./\"")
  ADD_API_UNIT_TEST(API_DIRECT_QUERY AMS::API::ModelDefine::Broadcast::Random::Double::DB::HALF::HDF5::HOST "${CMAKE_CURRENT_BINARY_DIR}/ams_ete_broadcast 0 8 8 ${TORCH_MODEL_DIR}/linear_scripted_single_cpu_random.pt \"double\" \"random\" 0.5 1 128 \"hdf5\" \"./\"; python3 ${CMAKE_CURRENT_SOURCE_DIR}/verify_ete.py 0 8 8 ${CMAKE_CURRENT_SOURCE_DIR}/linear_scripted_cpu.pt \"double\" \"random\" 0.5 1 128 \"hdf5\" \"./\"")

endfunction()

function(BUILD_TEST exe source)
  add_executable(${exe} ${source})
  target_include_directories(${exe} PRIVATE "${PROJECT_SOURCE_DIR}/src/AMSlib/" "${PROJECT_SOURCE_DIR}/src/AMSlib/include" ${caliper_INCLUDE_DIR} ${MPI_INCLUDE_PATH})
  target_link_libraries(${exe} PRIVATE AMS torch)
  if (WITH_HDF5)
    target_link_libraries(${exe} PRIVATE ${AMS_HDF5_TARGET})
  endif()
  target_compile_definitions(${exe} PRIVATE ${AMS_APP_DEFINES})
endfunction()

BUILD_TEST(ams_physics ams_physics.cpp)
BUILD_TEST(ams_end_to_end ams_ete.cpp)
BUILD_TEST(ams_inout_2d ams_ete_2d.cpp)
BUILD_TEST(ams_ete_broadcast ams_ete_broadcast.cpp)
INTEGRATION_TEST()
BUILD_TEST(ams_end_to_end_env ams_ete_env.cpp)
INTEGRATION_TEST_ENV()
BUILD_TEST(ams_multi_model_end_to_end ams_multi_model_ete.cpp)
ADD_API_UNIT_TEST(API_DIRECT_QUERY AMS::API::MultiModelDefine::Random::Double::DB::OnlyPhysics::None::HOST "${CMAKE_CURRENT_BINARY_DIR}/ams_multi_model_end_to_end 0 8 8 ${TORCH_MODEL_DIR}/linear_scripted_single_cpu_random.pt \"double\" \"random\" 0.0 1 128 \"none\" \"./\"; python3 ${CMAKE_CURRENT_SOURCE_DIR}/verify_ete.py 0 8 8 ${CMAKE_CURRENT_SOURCE_DIR}/linear_scripted_cpu.pt \"double\" \"random\" 0.0 1 128 \"none\" \"./\"")


BUILD_TEST(ams_multi_model_end_to_end_env ams_multi_model_ete_env.cpp)
ADD_API_UNIT_TEST(APIEnvAPI AMS::ENV::MultiModelDefine::Random10::Random50::Double::DB::${db_type}::HOST "AMS_OBJECTS=${JSON_FP} ${CMAKE_CURRENT_BINARY_DIR}/ams_end_to_end_env  0 8 8 \"double\" 1 128 app_random_10 app_random_50;AMS_OBJECTS=${JSON_FP} python3 ${CMAKE_CURRENT_SOURCE_DIR}/verify_ete.py 0 8 8 \"double\" 128 app_random_10 app_random_50")


#BUILD_TEST(ams_rmq ams_rmq_env.cpp)
#INTEGRATION_TEST_RMQ()
#
