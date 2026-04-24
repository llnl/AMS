#include <catch2/catch_test_macros.hpp>

#include "AMS.h"
#include "AMSGraph.hpp"
#include "AMSTensor.hpp"

using namespace ams;

// Paths to graph test models (relative to test executable)
static const char* HOMOGENEOUS_GRAPH_MODEL_PATH =
    "../models/homogeneous_graph.pt";
static const char* HETEROGENEOUS_GRAPH_MODEL_PATH =
    "../models/heterogeneous_graph.pt";

CATCH_TEST_CASE("AMSExecute homogeneous graph surrogate execution",
                "[wf][graph][surrogate]")
{
  AMSInit();

  // Setup: Register model with actual generated model path
  auto model = AMSRegisterAbstractModel("test_homo_surrogate",
                                        0.5,
                                        HOMOGENEOUS_GRAPH_MODEL_PATH,
                                        false);
  AMSExecutor executor = AMSCreateExecutor(model, 0, 1);

  // Create simple homogeneous graph with 'x' field (expected by test model)
  AMSHomogeneousGraph graph;

  // Insert node features tensor named 'x'
  AMSTensor::IntDimType node_shape[] = {10, 16};  // 10 nodes, 16 features
  AMSTensor::IntDimType node_strides[] = {16, 1};
  auto node_features = AMSTensor::create<float>(
      ams::ArrayRef<AMSTensor::IntDimType>(node_shape, 2),
      ams::ArrayRef<AMSTensor::IntDimType>(node_strides, 2),
      AMSResourceType::AMS_HOST);

  // Fill with test data
  float* features_data = node_features.data<float>();
  for (int i = 0; i < 160; ++i) {
    features_data[i] = static_cast<float>(i) * 0.1f;
  }

  insertTensor(graph, "x", std::move(node_features));

  // Define callback (should NOT be called if surrogate succeeds)
  bool callback_invoked = false;
  HomogeneousGraphDomainFn callback = [&](const AMSHomogeneousGraph& g,
                                          SmallVector<AMSTensor>& outputs) {
    callback_invoked = true;

    // Verify graph structure
    CATCH_REQUIRE(containsTensor(g, "x"));
    const auto* x = findTensor(g, "x");
    CATCH_REQUIRE(x != nullptr);
    CATCH_REQUIRE(x->shape()[0] == 10);
    CATCH_REQUIRE(x->shape()[1] == 16);

    // Create output tensor (8 features per node, matching model output)
    AMSTensor::IntDimType out_shape[] = {10, 8};
    AMSTensor::IntDimType out_strides[] = {8, 1};
    auto out_tensor = AMSTensor::create<float>(
        ams::ArrayRef<AMSTensor::IntDimType>(out_shape, 2),
        ams::ArrayRef<AMSTensor::IntDimType>(out_strides, 2),
        AMSResourceType::AMS_HOST);

    // Fill with physics computation result
    float* out_data = out_tensor.data<float>();
    for (int i = 0; i < 80; ++i) {
      out_data[i] = static_cast<float>(i);
    }

    outputs.clear();
    outputs.push_back(std::move(out_tensor));
  };

  // Execute
  SmallVector<AMSTensor> outs;
  AMSExecute(executor, callback, graph, outs);

  // Model is available, so surrogate should be used (callback NOT invoked)
  CATCH_REQUIRE_FALSE(callback_invoked);
  CATCH_REQUIRE(outs.size() == 1);
  CATCH_REQUIRE(outs[0].shape()[0] == 10);
  CATCH_REQUIRE(outs[0].shape()[1] == 8);
}

CATCH_TEST_CASE("AMSExecute heterogeneous graph surrogate execution",
                "[wf][graph][surrogate]")
{
  AMSInit();

  // Setup: Register model with actual generated model path
  auto model = AMSRegisterAbstractModel("test_hetero_surrogate",
                                        0.5,
                                        HETEROGENEOUS_GRAPH_MODEL_PATH,
                                        false);
  AMSExecutor executor = AMSCreateExecutor(model, 0, 1);

  // Create heterogeneous graph
  AMSHeterogeneousGraph graph;

  // Add node store for "node" type with 'x' features
  // Note: Using fixed "node" name to match test fixture model expectation
  AMSTensorMap node_store;
  AMSTensor::IntDimType node_shape[] = {10, 16};
  AMSTensor::IntDimType node_strides[] = {16, 1};
  auto node_features = AMSTensor::create<float>(
      ams::ArrayRef<AMSTensor::IntDimType>(node_shape, 2),
      ams::ArrayRef<AMSTensor::IntDimType>(node_strides, 2),
      AMSResourceType::AMS_HOST);

  float* node_data = node_features.data<float>();
  for (int i = 0; i < 160; ++i) {
    node_data[i] = static_cast<float>(i) * 0.1f;
  }

  insertTensor(node_store, "x", std::move(node_features));
  graph.node_stores["node"] = std::move(node_store);

  // Add edge store with dummy data (empty dicts can cause TorchScript issues)
  AMSTensorMap edge_store;
  AMSTensor::IntDimType dummy_shape[] = {1, 1};
  AMSTensor::IntDimType dummy_strides[] = {1, 1};
  auto dummy_edge = AMSTensor::create<float>(
      ams::ArrayRef<AMSTensor::IntDimType>(dummy_shape, 2),
      ams::ArrayRef<AMSTensor::IntDimType>(dummy_strides, 2),
      AMSResourceType::AMS_HOST);
  dummy_edge.data<float>()[0] = 0.0f;
  insertTensor(edge_store, "dummy", std::move(dummy_edge));
  EdgeType edge_type{"node", "edge", "node"};
  graph.edge_stores[edge_type] = std::move(edge_store);

  // Add global store with dummy data
  AMSTensor::IntDimType global_shape[] = {1, 1};
  AMSTensor::IntDimType global_strides[] = {1, 1};
  auto dummy_global = AMSTensor::create<float>(
      ams::ArrayRef<AMSTensor::IntDimType>(global_shape, 2),
      ams::ArrayRef<AMSTensor::IntDimType>(global_strides, 2),
      AMSResourceType::AMS_HOST);
  dummy_global.data<float>()[0] = 0.0f;
  insertTensor(graph.global_store, "dummy", std::move(dummy_global));

  // Define callback (should NOT be called if surrogate succeeds)
  bool callback_invoked = false;
  HeterogeneousGraphDomainFn callback = [&](const AMSHeterogeneousGraph& g,
                                            SmallVector<AMSTensor>& outputs) {
    callback_invoked = true;

    // Verify graph structure
    CATCH_REQUIRE(g.containsNodeStore("node"));
    const auto* node_store = g.findNodeStore("node");
    CATCH_REQUIRE(node_store != nullptr);
    CATCH_REQUIRE(containsTensor(*node_store, "x"));

    // Create output tensor
    AMSTensor::IntDimType out_shape[] = {10, 8};
    AMSTensor::IntDimType out_strides[] = {8, 1};
    auto out_tensor = AMSTensor::create<float>(
        ams::ArrayRef<AMSTensor::IntDimType>(out_shape, 2),
        ams::ArrayRef<AMSTensor::IntDimType>(out_strides, 2),
        AMSResourceType::AMS_HOST);

    float* out_data = out_tensor.data<float>();
    for (int i = 0; i < 80; ++i) {
      out_data[i] = static_cast<float>(i);
    }

    outputs.clear();
    outputs.push_back(std::move(out_tensor));
  };

  // Execute
  SmallVector<AMSTensor> outs;
  AMSExecute(executor, callback, graph, outs);

  // Model is available, so surrogate should be used (callback NOT invoked)
  CATCH_REQUIRE_FALSE(callback_invoked);
  CATCH_REQUIRE(outs.size() == 1);
  CATCH_REQUIRE(outs[0].shape()[0] == 10);
  CATCH_REQUIRE(outs[0].shape()[1] == 8);
}

CATCH_TEST_CASE("Graph surrogate with no model triggers fallback",
                "[wf][graph][surrogate][fallback]")
{
  AMSInit();

  // Setup: Register model with empty path (no model available)
  auto model = AMSRegisterAbstractModel("test_no_model", 0.5, "", false);
  AMSExecutor executor = AMSCreateExecutor(model, 0, 1);

  // Create simple homogeneous graph
  AMSHomogeneousGraph graph;
  AMSTensor::IntDimType shape[] = {5, 16};
  AMSTensor::IntDimType strides[] = {16, 1};
  auto features =
      AMSTensor::create<float>(ams::ArrayRef<AMSTensor::IntDimType>(shape, 2),
                               ams::ArrayRef<AMSTensor::IntDimType>(strides, 2),
                               AMSResourceType::AMS_HOST);

  float* data = features.data<float>();
  for (int i = 0; i < 80; ++i) {
    data[i] = 1.0f;
  }

  insertTensor(graph, "x", std::move(features));

  // Define callback
  bool callback_invoked = false;
  HomogeneousGraphDomainFn callback = [&](const AMSHomogeneousGraph& g,
                                          SmallVector<AMSTensor>& outputs) {
    callback_invoked = true;

    AMSTensor::IntDimType out_shape[] = {5, 8};
    AMSTensor::IntDimType out_strides[] = {8, 1};
    auto out = AMSTensor::create<float>(
        ams::ArrayRef<AMSTensor::IntDimType>(out_shape, 2),
        ams::ArrayRef<AMSTensor::IntDimType>(out_strides, 2),
        AMSResourceType::AMS_HOST);

    outputs.clear();
    outputs.push_back(std::move(out));
  };

  // Execute
  SmallVector<AMSTensor> outs;
  AMSExecute(executor, callback, graph, outs);

  // Invalid model should trigger fallback
  CATCH_REQUIRE(callback_invoked);
  CATCH_REQUIRE(outs.size() == 1);
}
