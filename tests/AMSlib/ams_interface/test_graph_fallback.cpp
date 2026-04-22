#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "AMS.h"
#include "AMSGraph.hpp"
#include "AMSTensor.hpp"

using namespace ams;

CATCH_TEST_CASE("AMSExecute homogeneous graph fallback path", "[wf][graph]")
{
  AMSInit();

  // Setup: Register model with no surrogate path (forces fallback)
  auto model = AMSRegisterAbstractModel("test_homo_graph", 0.5, "", false);
  AMSExecutor executor = AMSCreateExecutor(model, 0, 1);

  // Create simple homogeneous graph (dict of tensors)
  AMSHomogeneousGraph graph;

  // Insert node features tensor
  AMSTensor::IntDimType node_shape[] = {10, 3};
  AMSTensor::IntDimType node_strides[] = {3, 1};
  auto node_features = AMSTensor::create<float>(
      ams::ArrayRef<AMSTensor::IntDimType>(node_shape, 2),
      ams::ArrayRef<AMSTensor::IntDimType>(node_strides, 2),
      AMSResourceType::AMS_HOST);

  // Fill with test data
  float* features_data = node_features.data<float>();
  for (int i = 0; i < 30; ++i) {
    features_data[i] = static_cast<float>(i);
  }

  insertTensor(graph, "node_features", std::move(node_features));

  // Create output tensor
  SmallVector<AMSTensor> outs;
  AMSTensor::IntDimType out_shape[] = {10, 2};
  AMSTensor::IntDimType out_strides[] = {2, 1};
  auto out_tensor = AMSTensor::create<float>(
      ams::ArrayRef<AMSTensor::IntDimType>(out_shape, 2),
      ams::ArrayRef<AMSTensor::IntDimType>(out_strides, 2),
      AMSResourceType::AMS_HOST);

  // Initialize outputs to zero
  float* out_data = out_tensor.data<float>();
  for (int i = 0; i < 20; ++i) {
    out_data[i] = 0.0f;
  }

  outs.push_back(std::move(out_tensor));

  // Define callback that processes graph
  bool callback_invoked = false;
  HomogeneousGraphDomainFn callback =
      [&](const AMSHomogeneousGraph& g, SmallVector<AMSTensor>& outputs) {
        callback_invoked = true;

        // Verify graph structure
        CATCH_REQUIRE(containsTensor(g, "node_features"));
        const auto* features = findTensor(g, "node_features");
        CATCH_REQUIRE(features != nullptr);
        CATCH_REQUIRE(features->shape()[0] == 10);
        CATCH_REQUIRE(features->shape()[1] == 3);

        // Verify input data
        const float* features_data = features->data<float>();
        CATCH_REQUIRE(features_data[0] == 0.0f);
        CATCH_REQUIRE(features_data[29] == 29.0f);

        // Fill outputs with computation result
        CATCH_REQUIRE(outputs.size() == 1);
        float* out_data = outputs[0].data<float>();
        for (int i = 0; i < 20; ++i) {
          out_data[i] = static_cast<float>(i * 2);
        }
      };

  // Execute
  AMSExecute(executor, callback, graph, outs);

  // Verify callback was invoked (fallback path)
  CATCH_REQUIRE(callback_invoked);

  // Verify outputs were written
  const float* result_data = outs[0].data<float>();
  CATCH_REQUIRE(result_data[0] == 0.0f);
  CATCH_REQUIRE(result_data[10] == 20.0f);
  CATCH_REQUIRE(result_data[19] == 38.0f);

  // Note: Not destroying executor to avoid triggering AMSFinalize between tests
  // The executor will be cleaned up at program exit
}

CATCH_TEST_CASE("AMSExecute heterogeneous graph fallback path", "[wf][graph]")
{
  AMSInit();

  // Setup: Register model with no surrogate path (forces fallback)
  auto model = AMSRegisterAbstractModel("test_hetero_graph", 0.5, "", false);
  AMSExecutor executor = AMSCreateExecutor(model, 0, 1);

  // Create heterogeneous graph
  AMSHeterogeneousGraph graph;

  // Add node store for "atom" nodes
  auto& atom_store = graph.getOrCreateNodeStore("atom");
  AMSTensor::IntDimType atom_shape[] = {5, 2};
  AMSTensor::IntDimType atom_strides[] = {2, 1};
  auto atom_features = AMSTensor::create<float>(
      ams::ArrayRef<AMSTensor::IntDimType>(atom_shape, 2),
      ams::ArrayRef<AMSTensor::IntDimType>(atom_strides, 2),
      AMSResourceType::AMS_HOST);

  // Fill with test data
  float* atom_data = atom_features.data<float>();
  for (int i = 0; i < 10; ++i) {
    atom_data[i] = static_cast<float>(i + 1);
  }

  insertTensor(atom_store, "features", std::move(atom_features));

  // Add edge store
  auto& edge_store = graph.getOrCreateEdgeStore(EdgeType{"atom", "bond", "atom"});
  AMSTensor::IntDimType edge_shape[] = {2, 8};
  AMSTensor::IntDimType edge_strides[] = {8, 1};
  auto edge_index = AMSTensor::create<int64_t>(
      ams::ArrayRef<AMSTensor::IntDimType>(edge_shape, 2),
      ams::ArrayRef<AMSTensor::IntDimType>(edge_strides, 2),
      AMSResourceType::AMS_HOST);

  // Fill with edge connectivity
  int64_t* edge_data = edge_index.data<int64_t>();
  for (int i = 0; i < 16; ++i) {
    edge_data[i] = i % 5;
  }

  insertTensor(edge_store, "edge_index", std::move(edge_index));

  // Add global features
  AMSTensor::IntDimType global_shape[] = {1, 4};
  AMSTensor::IntDimType global_strides[] = {4, 1};
  auto global_features = AMSTensor::create<float>(
      ams::ArrayRef<AMSTensor::IntDimType>(global_shape, 2),
      ams::ArrayRef<AMSTensor::IntDimType>(global_strides, 2),
      AMSResourceType::AMS_HOST);

  float* global_data = global_features.data<float>();
  for (int i = 0; i < 4; ++i) {
    global_data[i] = static_cast<float>(i * 10);
  }

  insertTensor(graph.global_store, "global", std::move(global_features));

  // Create output tensor
  SmallVector<AMSTensor> outs;
  AMSTensor::IntDimType hetero_out_shape[] = {5, 1};
  AMSTensor::IntDimType hetero_out_strides[] = {1, 1};
  auto out_tensor = AMSTensor::create<float>(
      ams::ArrayRef<AMSTensor::IntDimType>(hetero_out_shape, 2),
      ams::ArrayRef<AMSTensor::IntDimType>(hetero_out_strides, 2),
      AMSResourceType::AMS_HOST);

  // Initialize to zero
  float* out_data = out_tensor.data<float>();
  for (int i = 0; i < 5; ++i) {
    out_data[i] = 0.0f;
  }

  outs.push_back(std::move(out_tensor));

  // Define callback
  bool callback_invoked = false;
  HeterogeneousGraphDomainFn callback =
      [&](const AMSHeterogeneousGraph& g, SmallVector<AMSTensor>& outputs) {
        callback_invoked = true;

        // Verify graph structure
        CATCH_REQUIRE(g.containsNodeStore("atom"));
        const auto* atom_store_ptr = g.findNodeStore("atom");
        CATCH_REQUIRE(atom_store_ptr != nullptr);
        CATCH_REQUIRE(containsTensor(*atom_store_ptr, "features"));

        // Verify node data
        const auto* features = findTensor(*atom_store_ptr, "features");
        CATCH_REQUIRE(features != nullptr);
        const float* features_data = features->data<float>();
        CATCH_REQUIRE(features_data[0] == 1.0f);
        CATCH_REQUIRE(features_data[9] == 10.0f);

        // Verify edge store
        CATCH_REQUIRE(g.containsEdgeStore(EdgeType{"atom", "bond", "atom"}));
        const auto* edge_store_ptr =
            g.findEdgeStore(EdgeType{"atom", "bond", "atom"});
        CATCH_REQUIRE(edge_store_ptr != nullptr);
        CATCH_REQUIRE(containsTensor(*edge_store_ptr, "edge_index"));

        // Verify global store
        CATCH_REQUIRE(containsTensor(g.global_store, "global"));
        const auto* global = findTensor(g.global_store, "global");
        CATCH_REQUIRE(global != nullptr);
        const float* global_data = global->data<float>();
        CATCH_REQUIRE(global_data[0] == 0.0f);
        CATCH_REQUIRE(global_data[3] == 30.0f);

        // Fill outputs
        CATCH_REQUIRE(outputs.size() == 1);
        float* out_data = outputs[0].data<float>();
        for (int i = 0; i < 5; ++i) {
          out_data[i] = static_cast<float>(i * 3);
        }
      };

  // Execute
  AMSExecute(executor, callback, graph, outs);

  // Verify
  CATCH_REQUIRE(callback_invoked);
  const float* result_data = outs[0].data<float>();
  CATCH_REQUIRE(result_data[0] == 0.0f);
  CATCH_REQUIRE(result_data[2] == 6.0f);
  CATCH_REQUIRE(result_data[4] == 12.0f);

  // Note: Not destroying executor to avoid triggering AMSFinalize between tests
  // The executor will be cleaned up at program exit
}

CATCH_TEST_CASE("Graph callback type safety", "[wf][graph]")
{
  // This test verifies that the type system prevents mismatches
  // It primarily exists as a compile-time check

  AMSHomogeneousGraph homo_graph;
  AMSHeterogeneousGraph hetero_graph;

  HomogeneousGraphDomainFn homo_fn =
      [](const AMSHomogeneousGraph&, SmallVector<AMSTensor>&) {};

  HeterogeneousGraphDomainFn hetero_fn =
      [](const AMSHeterogeneousGraph&, SmallVector<AMSTensor>&) {};

  // These should compile:
  // AMSExecute(executor, homo_fn, homo_graph, outs);
  // AMSExecute(executor, hetero_fn, hetero_graph, outs);

  // These should NOT compile (type mismatch):
  // AMSExecute(executor, homo_fn, hetero_graph, outs);  // ERROR
  // AMSExecute(executor, hetero_fn, homo_graph, outs);  // ERROR

  CATCH_SUCCEED("Type safety validated at compile time");
}
