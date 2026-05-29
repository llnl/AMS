#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "AMS.h"
#include "AMSGraph.hpp"
#include "AMSTensor.hpp"
#include "models/mgn_graph_fixtures.hpp"

using namespace ams;

using Dim = AMSTensor::IntDimType;

static const char* MGN_GRAPH_MODEL_PATH = "../models/mgn_graph_diffusion.pt";

static std::vector<Dim> contiguousStrides(const std::vector<Dim>& shape)
{
  std::vector<Dim> strides(shape.size(), 1);
  Dim stride = 1;
  for (std::size_t i = shape.size(); i-- > 0;) {
    strides[i] = stride;
    stride *= shape[i];
  }
  return strides;
}

template <typename T>
static AMSTensor makeTensor(std::vector<Dim> shape, const T* values)
{
  std::vector<Dim> strides = contiguousStrides(shape);
  auto tensor = AMSTensor::create<T>(shape, strides, AMSResourceType::AMS_HOST);
  std::copy(values, values + tensor.elements(), tensor.template data<T>());
  return tensor;
}

static Dim toDim(std::int64_t value) { return static_cast<Dim>(value); }

static AMSHomogeneousGraph makeGraph(const ams::test::mgn::GraphFixture& f)
{
  return AMSHomogeneousGraph(
      makeTensor<float>({toDim(f.num_nodes), toDim(f.node_feature_dim)},
                        f.node_features),
      makeTensor<std::int64_t>({2, toDim(f.num_edges)}, f.edge_index),
      makeTensor<float>({toDim(f.num_edges), toDim(f.edge_feature_dim)},
                        f.edge_features),
      makeTensor<float>({1, toDim(f.global_feature_dim)}, f.global_features));
}

static void verifyDeltaU(const ams::test::mgn::GraphFixture& f,
                         const AMSHomogeneousGraphFields& outputs)
{
  CATCH_REQUIRE(outputs.node_fields.contains("delta_u"));
  const auto& delta_u = outputs.node_fields.at("delta_u");
  CATCH_REQUIRE(delta_u.shape()[0] == f.num_nodes);
  CATCH_REQUIRE(delta_u.shape()[1] == f.reference_output_dim);

  const float* actual = delta_u.data<float>();
  const std::int64_t count = f.num_nodes * f.reference_output_dim;
  for (std::int64_t i = 0; i < count; ++i) {
    CATCH_REQUIRE(actual[i] == Catch::Approx(f.reference_delta_u[i])
                                      .epsilon(ams::test::mgn::kComparisonRtol)
                                      .margin(ams::test::mgn::kComparisonAtol));
  }
}

CATCH_TEST_CASE("AMSExecute homogeneous graph MGN diffusion surrogate",
                "[wf][graph][surrogate][mgn]")
{
  AMSInit();

  auto model = AMSRegisterAbstractModel("test_mgn_graph_diffusion_surrogate",
                                        0.5,
                                        MGN_GRAPH_MODEL_PATH,
                                        false);
  AMSExecutor executor = AMSCreateExecutor(model, 0, 1);

  for (std::size_t i = 0; i < ams::test::mgn::kNumGraphFixtures; ++i) {
    const auto& fixture = ams::test::mgn::kGraphFixtures[i];
    CATCH_DYNAMIC_SECTION("fixture " << fixture.name)
    {
      AMSHomogeneousGraph graph = makeGraph(fixture);

      bool callback_invoked = false;
      HomogeneousGraphDomainFn callback =
          [&](const AMSHomogeneousGraph&, AMSHomogeneousGraphFields& outputs) {
            callback_invoked = true;
            outputs.node_fields.set(
                "delta_u",
                makeTensor<float>({toDim(fixture.num_nodes),
                                   toDim(fixture.reference_output_dim)},
                                  fixture.reference_delta_u));
          };

      AMSHomogeneousGraphFields outputs;
      AMSExecute(executor, callback, graph, outputs);

      CATCH_REQUIRE_FALSE(callback_invoked);
      verifyDeltaU(fixture, outputs);
    }
  }
}
