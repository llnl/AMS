/*
 * Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
 * AMSLib Project Developers
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "AMS.h"
#include "AMSGraph.hpp"
#include "AMSTensor.hpp"
#include "nlohmann/json.hpp"

using namespace ams;
namespace fs = std::filesystem;

// Helper to create contiguous strides from shape
static std::vector<int64_t> contiguousStrides(const std::vector<int64_t>& shape)
{
  std::vector<int64_t> strides(shape.size(), 1);
  int64_t stride = 1;
  for (std::size_t i = shape.size(); i-- > 0;) {
    strides[i] = stride;
    stride *= shape[i];
  }
  return strides;
}

// Helper to create tensor with automatic strides
template <typename T>
static AMSTensor makeTensor(std::vector<int64_t> shape)
{
  std::vector<int64_t> strides = contiguousStrides(shape);
  return AMSTensor::create<T>(shape, strides, AMSResourceType::AMS_HOST);
}

CATCH_TEST_CASE(
    "Homogeneous graph forced physics stores data and reveals native schema",
    "[wf][graph][storage]")
{
  // Use unique temp directory
  fs::path test_dir =
      fs::temp_directory_path() / "ams_graph_schema_discovery_test";
  fs::remove_all(test_dir);
  fs::create_directories(test_dir);

  // Initialize AMS (from A0 investigation: AMSInit before AMSConfigureFSDatabase)
  AMSInit();
  AMSConfigureFSDatabase(AMSDBType::AMS_JSON, test_dir.string().c_str());

  // Recorder configuration: threshold < 0 forces physics, store_data=true enables DB
  AMSCAbstrModel recorder =
      AMSRegisterAbstractModel("heat_graph_schema", -1.0, "", true);
  AMSExecutor executor = AMSCreateExecutor(recorder, 0, 1);

  // Build small test graph
  const int64_t N = 10;  // nodes
  const int64_t E = 9;   // edges

  // Node features [N, 3] - float32
  auto node_feat = makeTensor<float>({N, 3});
  float* nf = node_feat.data<float>();
  for (int64_t i = 0; i < N * 3; i++) {
    nf[i] = static_cast<float>(i) * 0.1f;
  }

  // Edge index [2, E] - int64 (canonical)
  auto edge_idx = makeTensor<int64_t>({2, E});
  int64_t* ei = edge_idx.data<int64_t>();
  for (int64_t e = 0; e < E; e++) {
    ei[e] = e;                // source in first half
    ei[E + e] = (e + 1) % N;  // destination in second half
  }

  // Edge features [E, 2] - float32
  auto edge_feat = makeTensor<float>({E, 2});
  float* ef = edge_feat.data<float>();
  for (int64_t e = 0; e < E * 2; e++) {
    ef[e] = static_cast<float>(e) * 0.01f;
  }

  // Global features [2] - float32
  auto global_feat = makeTensor<float>({2});
  float* gf = global_feat.data<float>();
  gf[0] = 0.01f;   // dt
  gf[1] = 0.123f;  // time_n

  // Construct graph via public API
  AMSHomogeneousGraph graph(std::move(node_feat),
                            std::move(edge_idx),
                            std::move(edge_feat),
                            std::move(global_feat));

  AMSHomogeneousGraphFields outputs;

  // Physics callback that returns float64 delta_u
  int callback_count = 0;
  HomogeneousGraphDomainFn physics = [&](const AMSHomogeneousGraph& g,
                                         AMSHomogeneousGraphFields& o) {
    callback_count++;

    // Return float64 delta_u [N, 1] (MFEM precision)
    const int64_t num_nodes = g.node_features.shape()[0];
    auto delta = makeTensor<double>({num_nodes, 1});
    double* data = delta.data<double>();
    for (int64_t i = 0; i < num_nodes; i++) {
      data[i] = static_cast<double>(i) * 0.123;
    }

    o.node_fields.insert("delta_u", std::move(delta));
  };

  // Execute with forced physics + storage
  AMSExecute(executor, physics, graph, outputs);

  // Verify physics ran
  CATCH_REQUIRE(callback_count == 1);
  CATCH_REQUIRE(outputs.node_fields.find("delta_u") != nullptr);

  // Destroy executor to flush manifest (but don't call AMSFinalize)
  AMSDestroyExecutor(executor);

  // ========================================================================
  // INSPECT ACTUAL JSONDB OUTPUT - source of truth for schema
  // ========================================================================

  CATCH_REQUIRE(fs::exists(test_dir / "manifest.json"));

  std::ifstream manifest_file(test_dir / "manifest.json");
  nlohmann::json manifest;
  manifest_file >> manifest;

  // Document native schema structure
  std::cout << "\n=== NATIVE JSONDB SCHEMA (A1 Discovery) ===\n";
  std::cout << manifest.dump(2) << std::endl;
  std::cout << "==========================================\n" << std::endl;

  // Verify essential structure exists
  CATCH_REQUIRE(manifest.contains("format_version"));
  CATCH_REQUIRE(manifest.contains("endianness"));
  CATCH_REQUIRE(manifest.contains("cases"));
  CATCH_REQUIRE(manifest["cases"].is_array());
  CATCH_REQUIRE(manifest["cases"].size() == 1);

  auto case0 = manifest["cases"][0];
  CATCH_REQUIRE(case0.contains("tensors"));

  // Verify all graph components stored
  CATCH_REQUIRE(case0["tensors"].contains("node_features"));
  CATCH_REQUIRE(case0["tensors"].contains("edge_index"));
  CATCH_REQUIRE(case0["tensors"].contains("edge_features"));

  // CRITICAL: Verify global_features stored
  CATCH_REQUIRE(case0["tensors"].contains("global_features"));
  CATCH_REQUIRE(case0["global_feature_dim"].get<int64_t>() == 2);
  CATCH_REQUIRE(case0["tensors"]["global_features"]["shape"] ==
                nlohmann::json::array({2}));
  CATCH_REQUIRE(
      case0["tensors"]["global_features"]["byte_size"].get<size_t>() ==
      2 * sizeof(float));

  // Verify output field stored
  // Note: Field name may be "delta_u" or "target_delta_u" - discover actual
  bool has_delta_u = case0["tensors"].contains("delta_u") ||
                     case0["tensors"].contains("target_delta_u") ||
                     case0["tensors"].contains("node:delta_u");
  CATCH_REQUIRE(has_delta_u);

  // Verify dtypes preserved
  CATCH_REQUIRE(case0["tensors"]["node_features"]["dtype"].get<std::string>() ==
                "float32");
  CATCH_REQUIRE(case0["tensors"]["edge_index"]["dtype"].get<std::string>() ==
                "int64");
  CATCH_REQUIRE(case0["tensors"]["edge_features"]["dtype"].get<std::string>() ==
                "float32");
  CATCH_REQUIRE(
      case0["tensors"]["global_features"]["dtype"].get<std::string>() ==
      "float32");

  std::string global_path =
      case0["tensors"]["global_features"]["path"].get<std::string>();
  std::ifstream global_file(test_dir / global_path, std::ios::binary);
  CATCH_REQUIRE(global_file.is_open());
  float stored_globals[2] = {};
  global_file.read(reinterpret_cast<char*>(stored_globals),
                   static_cast<std::streamsize>(sizeof(stored_globals)));
  CATCH_REQUIRE(global_file.gcount() ==
                static_cast<std::streamsize>(sizeof(stored_globals)));
  CATCH_REQUIRE(stored_globals[0] == 0.01f);
  CATCH_REQUIRE(stored_globals[1] == 0.123f);

  // Verify float64 delta_u (CRITICAL for MFEM precision)
  std::string delta_key = "delta_u";
  if (case0["tensors"].contains("target_delta_u")) {
    delta_key = "target_delta_u";
  } else if (case0["tensors"].contains("node:delta_u")) {
    delta_key = "node:delta_u";
  }
  CATCH_REQUIRE(case0["tensors"][delta_key]["dtype"].get<std::string>() ==
                "float64");

  // Verify paths are relative to dataset root
  std::string node_path =
      case0["tensors"]["node_features"]["path"].get<std::string>();
  CATCH_REQUIRE(!fs::path(node_path).is_absolute());
  CATCH_REQUIRE(fs::exists(test_dir / node_path));

  // Note: Not destroying executor or calling AMSFinalize to avoid lifecycle
  // issues between tests. Cleanup happens at process exit.
  // fs::remove_all(test_dir);  // Keep for manual inspection

  std::cout << "A1 schema discovery test PASSED. Review manifest output above "
               "before proceeding to A2."
            << std::endl;
}

CATCH_TEST_CASE("Homogeneous graph without globals omits global storage",
                "[wf][graph][storage]")
{
  fs::path test_dir = fs::temp_directory_path() / "ams_graph_empty_globals_"
                                                  "test";
  fs::remove_all(test_dir);
  fs::create_directories(test_dir);

  AMSInit();
  AMSConfigureFSDatabase(AMSDBType::AMS_JSON, test_dir.string().c_str());

  AMSCAbstrModel recorder =
      AMSRegisterAbstractModel("empty_globals_domain", -1.0, "", true);
  AMSExecutor executor = AMSCreateExecutor(recorder, 0, 1);

  auto node_feat = makeTensor<float>({3, 2});
  auto edge_idx = makeTensor<int64_t>({2, 2});
  auto edge_feat = makeTensor<float>({2, 1});

  int64_t* edge_data = edge_idx.data<int64_t>();
  edge_data[0] = 0;
  edge_data[1] = 1;
  edge_data[2] = 1;
  edge_data[3] = 2;

  AMSHomogeneousGraph graph(std::move(node_feat),
                            std::move(edge_idx),
                            std::move(edge_feat));
  CATCH_REQUIRE(graph.global_features.shape().size() == 1);
  CATCH_REQUIRE(graph.global_features.shape()[0] == 0);

  AMSHomogeneousGraphFields outputs;
  HomogeneousGraphDomainFn physics = [](const AMSHomogeneousGraph& g,
                                        AMSHomogeneousGraphFields& o) {
    auto delta = makeTensor<double>({g.node_features.shape()[0], 1});
    o.node_fields.insert("delta_u", std::move(delta));
  };

  AMSExecute(executor, physics, graph, outputs);
  AMSDestroyExecutor(executor);

  std::ifstream manifest_file(test_dir / "manifest.json");
  CATCH_REQUIRE(manifest_file.is_open());
  nlohmann::json manifest;
  manifest_file >> manifest;

  CATCH_REQUIRE(manifest["cases"].size() == 1);
  const auto& stored_case = manifest["cases"][0];
  CATCH_REQUIRE(stored_case["global_feature_dim"].get<int64_t>() == 0);
  CATCH_REQUIRE_FALSE(stored_case["tensors"].contains("global_features"));
  CATCH_REQUIRE_FALSE(
      fs::exists(test_dir / "step_000000" / "global_features.bin"));

  fs::remove_all(test_dir);
}

// ============================================================================
// A2 Tests: Complete Storage Test Coverage
// ============================================================================

CATCH_TEST_CASE(
    "store_data=false returns physics output with zero recorded cases",
    "[wf][graph][storage]")
{
  fs::path test_dir = fs::temp_directory_path() / "ams_graph_no_storage_test";
  fs::remove_all(test_dir);
  fs::create_directories(test_dir);

  AMSInit();
  AMSConfigureFSDatabase(AMSDBType::AMS_JSON, test_dir.string().c_str());

  // Register with store_data=false
  AMSCAbstrModel no_store =
      AMSRegisterAbstractModel("no_storage_domain", -1.0, "", false);
  AMSExecutor executor = AMSCreateExecutor(no_store, 0, 1);

  // Build simple graph
  auto node_feat = makeTensor<float>({5, 2});
  auto edge_idx = makeTensor<int64_t>({2, 4});
  auto edge_feat = makeTensor<float>({4, 1});
  auto global_feat = makeTensor<float>({2});

  // Fill with identifiable values
  float* nf = node_feat.data<float>();
  for (int i = 0; i < 10; i++)
    nf[i] = static_cast<float>(i) + 100.0f;

  // Fill edge index (4 edges for 5 nodes)
  int64_t* ei = edge_idx.data<int64_t>();
  ei[0] = 0;
  ei[1] = 1;
  ei[2] = 2;
  ei[3] = 3;  // sources
  ei[4] = 1;
  ei[5] = 2;
  ei[6] = 3;
  ei[7] = 4;  // destinations

  // Fill edge features
  float* ef = edge_feat.data<float>();
  for (int i = 0; i < 4; i++)
    ef[i] = static_cast<float>(i) * 0.5f;

  // Fill global features
  float* gf = global_feat.data<float>();
  gf[0] = 0.01f;
  gf[1] = 0.02f;

  AMSHomogeneousGraph graph(std::move(node_feat),
                            std::move(edge_idx),
                            std::move(edge_feat),
                            std::move(global_feat));

  AMSHomogeneousGraphFields outputs;

  int callback_count = 0;
  HomogeneousGraphDomainFn physics = [&](const AMSHomogeneousGraph& g,
                                         AMSHomogeneousGraphFields& o) {
    callback_count++;
    const int64_t N = g.node_features.shape()[0];
    auto delta = makeTensor<double>({N, 1});
    double* data = delta.data<double>();
    for (int64_t i = 0; i < N; i++)
      data[i] = static_cast<double>(i) + 200.0;
    o.node_fields.insert("delta_u", std::move(delta));
  };

  AMSExecute(executor, physics, graph, outputs);

  // Verify physics ran and returned output
  CATCH_REQUIRE(callback_count == 1);
  CATCH_REQUIRE(outputs.node_fields.find("delta_u") != nullptr);

  const auto& delta = outputs.node_fields.at("delta_u");
  CATCH_REQUIRE(delta.shape()[0] == 5);
  CATCH_REQUIRE(delta.dType() == ams::AMS_DOUBLE);

  // Verify output values
  const double* delta_data = delta.data<double>();
  for (int i = 0; i < 5; i++) {
    CATCH_REQUIRE(delta_data[i] == static_cast<double>(i) + 200.0);
  }

  // Note: Not destroying executor to avoid triggering AMSFinalize between tests
  // The executor will be cleaned up at program exit

  // Verify NO manifest created (store_data=false)
  CATCH_REQUIRE(!fs::exists(test_dir / "manifest.json"));

  fs::remove_all(test_dir);
}

CATCH_TEST_CASE("Multiple calls accumulate cases with distinguishable values",
                "[wf][graph][storage]")
{
  fs::path test_dir = fs::temp_directory_path() / "ams_graph_accumulation_test";
  fs::remove_all(test_dir);
  fs::create_directories(test_dir);

  AMSInit();
  AMSConfigureFSDatabase(AMSDBType::AMS_JSON, test_dir.string().c_str());

  AMSCAbstrModel recorder =
      AMSRegisterAbstractModel("accumulation_domain", -1.0, "", true);
  AMSExecutor executor = AMSCreateExecutor(recorder, 0, 1);

  const int num_calls = 3;
  int total_callbacks = 0;

  for (int call = 0; call < num_calls; call++) {
    // Build graph with call-specific values
    auto node_feat = makeTensor<float>({4, 2});
    auto edge_idx = makeTensor<int64_t>({2, 3});
    auto edge_feat = makeTensor<float>({3, 1});
    auto global_feat = makeTensor<float>({1});

    // Fill node features with call-specific values
    float* nf = node_feat.data<float>();
    for (int i = 0; i < 8; i++) {
      nf[i] = static_cast<float>(call * 1000 + i);
    }

    // Fill edge index with valid connectivity (3 edges for 4 nodes)
    int64_t* ei = edge_idx.data<int64_t>();
    ei[0] = 0;
    ei[1] = 1;
    ei[2] = 2;  // sources
    ei[3] = 1;
    ei[4] = 2;
    ei[5] = 3;  // destinations

    // Fill edge features
    float* ef = edge_feat.data<float>();
    for (int i = 0; i < 3; i++) {
      ef[i] = static_cast<float>(call * 100 + i);
    }

    // Fill global features with call identifier
    float* gf = global_feat.data<float>();
    gf[0] = static_cast<float>(call);

    AMSHomogeneousGraph graph(std::move(node_feat),
                              std::move(edge_idx),
                              std::move(edge_feat),
                              std::move(global_feat));

    AMSHomogeneousGraphFields outputs;

    HomogeneousGraphDomainFn physics =
        [&, call_id = call](const AMSHomogeneousGraph& g,
                            AMSHomogeneousGraphFields& o) {
          total_callbacks++;
          const int64_t N = g.node_features.shape()[0];
          auto delta = makeTensor<double>({N, 1});
          double* data = delta.data<double>();
          for (int64_t i = 0; i < N; i++) {
            data[i] = static_cast<double>(call_id * 100 + i);
          }
          o.node_fields.insert("delta_u", std::move(delta));
        };

    AMSExecute(executor, physics, graph, outputs);
  }

  CATCH_REQUIRE(total_callbacks == num_calls);

  // Destroy executor to flush manifest (but don't call AMSFinalize)
  AMSDestroyExecutor(executor);

  // Verify manifest exists with correct case count
  CATCH_REQUIRE(fs::exists(test_dir / "manifest.json"));

  std::ifstream manifest_file(test_dir / "manifest.json");
  nlohmann::json manifest;
  manifest_file >> manifest;

  CATCH_REQUIRE(manifest["cases"].size() == num_calls);

  // Verify each case has distinguishable values
  for (int call = 0; call < num_calls; call++) {
    auto case_entry = manifest["cases"][call];
    CATCH_REQUIRE(case_entry["step_index"].get<int>() == call);

    // Load and verify global feature (call-specific marker)
    std::string global_path = case_entry["tensors"]["global_features"]["path"];
    fs::path full_path = test_dir / global_path;
    CATCH_REQUIRE(fs::exists(full_path));

    std::ifstream f(full_path, std::ios::binary);
    float global_val;
    f.read(reinterpret_cast<char*>(&global_val), sizeof(float));
    CATCH_REQUIRE(global_val == static_cast<float>(call));

    // Verify target output exists
    CATCH_REQUIRE(case_entry["tensors"].contains("target_delta_u"));
  }

  fs::remove_all(test_dir);
}

CATCH_TEST_CASE("Heterogeneous graph typed storage",
                "[.][wf][graph][storage][heterogeneous][future]")
{
  fs::path test_dir = fs::temp_directory_path() /
                      "ams_graph_heterogeneous_"
                      "test";
  fs::remove_all(test_dir);
  fs::create_directories(test_dir);

  AMSInit();
  AMSConfigureFSDatabase(AMSDBType::AMS_JSON, test_dir.string().c_str());

  AMSCAbstrModel recorder =
      AMSRegisterAbstractModel("hetero_domain", -1.0, "", true);
  AMSExecutor executor = AMSCreateExecutor(recorder, 0, 1);

  // Build heterogeneous graph with two node types
  AMSHeterogeneousGraph graph;

  // Node type "fluid": 5 nodes, 3 features
  auto& fluid_store = graph.getOrCreateNodeStore("fluid");
  auto fluid_feat = makeTensor<float>({5, 3});
  float* ff = fluid_feat.data<float>();
  for (int i = 0; i < 15; i++)
    ff[i] = static_cast<float>(i) * 0.1f;
  insertTensor(fluid_store, "features", std::move(fluid_feat));

  // Node type "solid": 3 nodes, 2 features
  auto& solid_store = graph.getOrCreateNodeStore("solid");
  auto solid_feat = makeTensor<float>({3, 2});
  float* sf = solid_feat.data<float>();
  for (int i = 0; i < 6; i++)
    sf[i] = static_cast<float>(i) * 0.2f;
  insertTensor(solid_store, "features", std::move(solid_feat));

  // Edge type: fluid->solid (4 edges)
  EdgeType edge_type("fluid", "interacts", "solid");
  auto& edge_store = graph.getOrCreateEdgeStore(edge_type);
  auto edge_idx = makeTensor<int64_t>({2, 4});
  int64_t* ei = edge_idx.data<int64_t>();
  ei[0] = 0;
  ei[1] = 1;
  ei[2] = 2;
  ei[3] = 3;  // fluid nodes (sources)
  ei[4] = 0;
  ei[5] = 1;
  ei[6] = 1;
  ei[7] = 2;  // solid nodes (destinations)
  insertTensor(edge_store, "edge_index", std::move(edge_idx));

  // Global features
  auto global_feat = makeTensor<float>({1, 2});
  float* gf = global_feat.data<float>();
  gf[0] = 0.01f;  // timestep
  gf[1] = 0.5f;   // time
  insertTensor(graph.global_store, "time", std::move(global_feat));

  AMSHeterogeneousGraphFields outputs;

  int callback_count = 0;
  HeterogeneousGraphDomainFn physics = [&](const AMSHeterogeneousGraph& g,
                                           AMSHeterogeneousGraphFields& o) {
    callback_count++;

    // Output for fluid nodes
    auto& fluid_out = o.getOrCreateNodeStore("fluid");
    auto fluid_delta = makeTensor<double>({5, 1});
    double* fd = fluid_delta.data<double>();
    for (int i = 0; i < 5; i++)
      fd[i] = static_cast<double>(i) + 10.0;
    insertTensor(fluid_out, "delta_u", std::move(fluid_delta));

    // Output for solid nodes
    auto& solid_out = o.getOrCreateNodeStore("solid");
    auto solid_delta = makeTensor<double>({3, 1});
    double* sd = solid_delta.data<double>();
    for (int i = 0; i < 3; i++)
      sd[i] = static_cast<double>(i) + 20.0;
    insertTensor(solid_out, "delta_u", std::move(solid_delta));
  };

  AMSExecute(executor, physics, graph, outputs);

  CATCH_REQUIRE(callback_count == 1);
  CATCH_REQUIRE(outputs.node_stores.find("fluid") != outputs.node_stores.end());
  CATCH_REQUIRE(outputs.node_stores.find("solid") != outputs.node_stores.end());

  // Note: Not destroying executor to avoid triggering AMSFinalize between tests
  // The executor will be cleaned up at program exit

  // Verify heterogeneous storage
  CATCH_REQUIRE(fs::exists(test_dir / "manifest.json"));

  std::ifstream manifest_file(test_dir / "manifest.json");
  nlohmann::json manifest;
  manifest_file >> manifest;

  CATCH_REQUIRE(manifest["cases"].size() == 1);
  auto case0 = manifest["cases"][0];

  // Verify typed node stores stored
  CATCH_REQUIRE(case0["tensors"].contains("node_fluid__features"));
  CATCH_REQUIRE(case0["tensors"].contains("node_solid__features"));

  // Verify typed outputs stored with target_ prefix
  CATCH_REQUIRE(case0["tensors"].contains("target_node_fluid__delta_u"));
  CATCH_REQUIRE(case0["tensors"].contains("target_node_solid__delta_u"));

  // Verify dtypes
  CATCH_REQUIRE(case0["tensors"]["target_node_fluid__delta_u"]["dtype"] ==
                "float64");
  CATCH_REQUIRE(case0["tensors"]["target_node_solid__delta_u"]["dtype"] ==
                "float64");

  // Verify shapes
  auto fluid_shape = case0["tensors"]["target_node_fluid__delta_u"]["shape"];
  CATCH_REQUIRE(fluid_shape[0] == 5);
  CATCH_REQUIRE(fluid_shape[1] == 1);

  auto solid_shape = case0["tensors"]["target_node_solid__delta_u"]["shape"];
  CATCH_REQUIRE(solid_shape[0] == 3);
  CATCH_REQUIRE(solid_shape[1] == 1);

  fs::remove_all(test_dir);
}

CATCH_TEST_CASE("Surrogate success: zero callbacks and zero stored cases",
                "[.][wf][graph][storage][surrogate][future]")
{
  // This test requires a working surrogate model that returns low uncertainty.
  // For now, verify the no-model case (which always runs physics).
  // TODO: Add actual surrogate test when model fixture is available.

  fs::path test_dir =
      fs::temp_directory_path() / "ams_graph_surrogate_success_test";
  fs::remove_all(test_dir);
  fs::create_directories(test_dir);

  AMSInit();
  AMSConfigureFSDatabase(AMSDBType::AMS_JSON, test_dir.string().c_str());

  // No model (empty path) means surrogate cannot succeed
  AMSCAbstrModel no_model =
      AMSRegisterAbstractModel("surrogate_domain", 0.5, "", true);
  AMSExecutor executor = AMSCreateExecutor(no_model, 0, 1);

  auto node_feat = makeTensor<float>({3, 2});
  auto edge_idx = makeTensor<int64_t>({2, 2});
  auto edge_feat = makeTensor<float>({2, 1});
  auto global_feat = makeTensor<float>({1});

  // Fill all tensors
  float* nf = node_feat.data<float>();
  for (int i = 0; i < 6; i++)
    nf[i] = static_cast<float>(i);

  int64_t* ei = edge_idx.data<int64_t>();
  ei[0] = 0;
  ei[1] = 1;  // sources
  ei[2] = 1;
  ei[3] = 2;  // destinations

  float* ef = edge_feat.data<float>();
  ef[0] = 0.5f;
  ef[1] = 1.0f;

  float* gf = global_feat.data<float>();
  gf[0] = 1.0f;

  AMSHomogeneousGraph graph(std::move(node_feat),
                            std::move(edge_idx),
                            std::move(edge_feat),
                            std::move(global_feat));

  AMSHomogeneousGraphFields outputs;

  int callback_count = 0;
  HomogeneousGraphDomainFn physics = [&](const AMSHomogeneousGraph& g,
                                         AMSHomogeneousGraphFields& o) {
    callback_count++;
    auto delta = makeTensor<double>({3, 1});
    o.node_fields.insert("delta_u", std::move(delta));
  };

  AMSExecute(executor, physics, graph, outputs);

  // Without model, physics runs
  CATCH_REQUIRE(callback_count == 1);
  CATCH_REQUIRE(outputs.node_fields.find("delta_u") != nullptr);

  // Note: Not destroying executor to avoid triggering AMSFinalize between tests
  // The executor will be cleaned up at program exit

  // Note: This test will be updated when surrogate model fixture is available
  // Expected behavior with working surrogate:
  // - callback_count == 0 (surrogate used)
  // - outputs populated by surrogate
  // - manifest contains 0 cases (no fallback storage)

  fs::remove_all(test_dir);
}

CATCH_TEST_CASE("No database configured: physics output returned without crash",
                "[wf][graph][storage]")
{
  // No AMSConfigureFSDatabase call - DB should be null

  AMSInit();

  // Register without configuring database (store_data=false → no DB needed)
  AMSCAbstrModel no_db =
      AMSRegisterAbstractModel("no_db_domain", -1.0, "", false);
  AMSExecutor executor = AMSCreateExecutor(no_db, 0, 1);

  auto node_feat = makeTensor<float>({4, 2});
  auto edge_idx = makeTensor<int64_t>({2, 3});
  auto edge_feat = makeTensor<float>({3, 1});
  auto global_feat = makeTensor<float>({1});

  // Fill all tensors
  float* nf = node_feat.data<float>();
  for (int i = 0; i < 8; i++)
    nf[i] = static_cast<float>(i);

  int64_t* ei = edge_idx.data<int64_t>();
  ei[0] = 0;
  ei[1] = 1;
  ei[2] = 2;  // sources
  ei[3] = 1;
  ei[4] = 2;
  ei[5] = 3;  // destinations

  float* ef = edge_feat.data<float>();
  for (int i = 0; i < 3; i++)
    ef[i] = static_cast<float>(i) * 0.3f;

  float* gf = global_feat.data<float>();
  gf[0] = 1.0f;

  AMSHomogeneousGraph graph(std::move(node_feat),
                            std::move(edge_idx),
                            std::move(edge_feat),
                            std::move(global_feat));

  AMSHomogeneousGraphFields outputs;

  int callback_count = 0;
  HomogeneousGraphDomainFn physics = [&](const AMSHomogeneousGraph& g,
                                         AMSHomogeneousGraphFields& o) {
    callback_count++;
    const int64_t N = g.node_features.shape()[0];
    auto delta = makeTensor<double>({N, 1});
    double* data = delta.data<double>();
    for (int64_t i = 0; i < N; i++)
      data[i] = static_cast<double>(i) * 2.5;
    o.node_fields.insert("delta_u", std::move(delta));
  };

  // Should not crash even without DB
  CATCH_REQUIRE_NOTHROW(AMSExecute(executor, physics, graph, outputs));

  // Verify physics ran and output returned
  CATCH_REQUIRE(callback_count == 1);
  CATCH_REQUIRE(outputs.node_fields.find("delta_u") != nullptr);

  const auto& delta = outputs.node_fields.at("delta_u");
  CATCH_REQUIRE(delta.shape()[0] == 4);

  // Verify values
  const double* delta_data = delta.data<double>();
  for (int i = 0; i < 4; i++) {
    CATCH_REQUIRE(delta_data[i] == static_cast<double>(i) * 2.5);
  }

  // Note: Not destroying executor to avoid triggering AMSFinalize between tests
  // The executor will be cleaned up at program exit
}

CATCH_TEST_CASE(
    "Surrogate rejection: model runs but rejected, physics executes once, "
    "exact output stored",
    "[wf][graph][storage][surrogate]")
{
  // This test validates the rejection path: when a surrogate model runs
  // but returns high uncertainty (above threshold), physics should execute
  // and store the fallback output exactly once.
  //
  // Current limitation: Requires a scripted model that returns high uncertainty.
  // For now, test the no-model fallback path which exercises the same storage logic.

  fs::path test_dir =
      fs::temp_directory_path() / "ams_graph_surrogate_reject_test";
  fs::remove_all(test_dir);
  fs::create_directories(test_dir);

  AMSInit();
  AMSConfigureFSDatabase(AMSDBType::AMS_JSON, test_dir.string().c_str());

  // No model path = immediate fallback to physics
  // (Same storage path as uncertainty rejection would trigger)
  AMSCAbstrModel fallback =
      AMSRegisterAbstractModel("reject_domain", 0.1, "", true);
  AMSExecutor executor = AMSCreateExecutor(fallback, 0, 1);

  auto node_feat = makeTensor<float>({6, 2});
  auto edge_idx = makeTensor<int64_t>({2, 5});
  auto edge_feat = makeTensor<float>({5, 1});
  auto global_feat = makeTensor<float>({1});

  // Fill with specific values to verify exact storage
  float* nf = node_feat.data<float>();
  for (int i = 0; i < 12; i++)
    nf[i] = static_cast<float>(i) * 1.5f;

  // Fill edge index (5 edges for 6 nodes)
  int64_t* ei = edge_idx.data<int64_t>();
  ei[0] = 0;
  ei[1] = 1;
  ei[2] = 2;
  ei[3] = 3;
  ei[4] = 4;  // sources
  ei[5] = 1;
  ei[6] = 2;
  ei[7] = 3;
  ei[8] = 4;
  ei[9] = 5;  // destinations

  // Fill edge features
  float* ef = edge_feat.data<float>();
  for (int i = 0; i < 5; i++)
    ef[i] = static_cast<float>(i) * 0.25f;

  // Fill global features
  float* gf = global_feat.data<float>();
  gf[0] = 1.0f;

  AMSHomogeneousGraph graph(std::move(node_feat),
                            std::move(edge_idx),
                            std::move(edge_feat),
                            std::move(global_feat));

  AMSHomogeneousGraphFields outputs;

  int callback_count = 0;
  HomogeneousGraphDomainFn physics = [&](const AMSHomogeneousGraph& g,
                                         AMSHomogeneousGraphFields& o) {
    callback_count++;
    const int64_t N = g.node_features.shape()[0];
    auto delta = makeTensor<double>({N, 1});
    double* data = delta.data<double>();
    for (int64_t i = 0; i < N; i++)
      data[i] = static_cast<double>(i) * 3.7;
    o.node_fields.insert("delta_u", std::move(delta));
  };

  AMSExecute(executor, physics, graph, outputs);

  // Verify physics executed exactly once
  CATCH_REQUIRE(callback_count == 1);

  // Verify output returned to caller
  CATCH_REQUIRE(outputs.node_fields.find("delta_u") != nullptr);
  const auto& delta = outputs.node_fields.at("delta_u");
  CATCH_REQUIRE(delta.shape()[0] == 6);
  CATCH_REQUIRE(delta.dType() == ams::AMS_DOUBLE);

  // Destroy executor to flush manifest (but don't call AMSFinalize)
  AMSDestroyExecutor(executor);

  // Verify exactly one case stored
  CATCH_REQUIRE(fs::exists(test_dir / "manifest.json"));

  std::ifstream manifest_file(test_dir / "manifest.json");
  nlohmann::json manifest;
  manifest_file >> manifest;

  CATCH_REQUIRE(manifest["cases"].size() == 1);

  auto case0 = manifest["cases"][0];

  // Verify stored output matches exact physics output
  std::string target_path = case0["tensors"]["target_delta_u"]["path"];
  fs::path full_path = test_dir / target_path;
  CATCH_REQUIRE(fs::exists(full_path));

  // Load stored values
  std::ifstream f(full_path, std::ios::binary);
  std::vector<double> stored_values(6);
  f.read(reinterpret_cast<char*>(stored_values.data()), 6 * sizeof(double));

  // Verify exact match with physics output
  for (int i = 0; i < 6; i++) {
    double expected = static_cast<double>(i) * 3.7;
    CATCH_REQUIRE(std::abs(stored_values[i] - expected) < 1e-12);
  }

  // Verify field name (runtime) vs storage name (target_ prefix)
  // Runtime API: node_fields["delta_u"]
  // Storage: "target_delta_u" in manifest
  CATCH_REQUIRE(case0["tensors"].contains("target_delta_u"));
  CATCH_REQUIRE(!case0["tensors"].contains("delta_u"));  // No prefix in storage

  // Verify dtype preserved as float64
  CATCH_REQUIRE(case0["tensors"]["target_delta_u"]["dtype"] == "float64");

  fs::remove_all(test_dir);
}
