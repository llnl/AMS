#include <algorithm>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "AMS.h"
#include "AMSGraph.hpp"
#include "AMSTensor.hpp"

// This test is the C++ half of the MGN diffusion example.
//
// The Python generator does the learned-model work:
//
//   1. create synthetic graph diffusion data;
//   2. train a small pure-Torch MeshGraphNet-like model;
//   3. export that model as an AMS-wrapped TorchScript file;
//   4. write "fixtures": fixed graph inputs plus Python TorchScript reference
//      outputs.
//
// This C++ test does not know the diffusion formula and does not try to judge
// model quality. Its job is deployment parity: load the exact TorchScript model
// and graph tensors from the generated fixture directory, run them through AMS,
// and verify that the AMS output field node:delta_u matches the Python
// TorchScript reference output.
//
// Said differently: Python proves "the model learned something"; this test
// proves "AMS runs the exported model the same way Python does."

using namespace ams;
using json = nlohmann::json;

using Dim = AMSTensor::IntDimType;

// The MGN diffusion workflow writes fixtures into the build tree at CTest
// runtime. CMake compiles that build-tree directory into this focused parity
// test so the test can be launched like any other Catch executable while still
// keeping generated tensors out of the source tree.
#ifndef AMS_MGN_DIFFUSION_FIXTURE_DIR
#define AMS_MGN_DIFFUSION_FIXTURE_DIR ""
#endif

namespace
{

constexpr int kFixtureFormatVersion = 1;
constexpr std::int64_t kNodeFeatureDim = 4;
constexpr std::int64_t kEdgeFeatureDim = 4;
constexpr std::int64_t kGlobalFeatureDim = 1;
constexpr std::int64_t kReferenceOutputDim = 1;
constexpr double kUnusedGraphSurrogateThreshold = 1.0;
constexpr int kSingleRankProcessId = 0;
constexpr int kSingleRankWorldSize = 1;
constexpr bool kStoreTrainingData = false;
constexpr const char* kDomainName = "test_mgn_graph_diffusion_surrogate";

// Glossary for readers new to this path:
//
// manifest: fixtures.json, the table of contents for generated fixture files.
// fixture directory: build-tree directory containing fixtures.json, the
//   TorchScript model, and raw tensor binaries.
// raw tensor binary: contiguous bytes for one tensor; shape/dtype live in JSON.
// AMSHomogeneousGraph: AMS C++ container for node_features, edge_index,
//   edge_features, and optional global_features.
// node:delta_u: TorchScript output key. AMS parses this into the node field
//   named "delta_u".
// fallback callback: application callback AMS calls only when surrogate
//   inference is unavailable or rejected.
// parity: equality of AMS/LibTorch inference and Python TorchScript inference.

struct TensorMetadata {
  std::filesystem::path path;
  std::string dtype;
  std::vector<std::int64_t> shape;
  std::string endianness;
  std::uintmax_t byte_size;
};

static std::vector<Dim> contiguousStrides(const std::vector<Dim>& shape)
{
  // AMSTensor stores both shape and strides. These fixtures are written as
  // simple row-major contiguous arrays, so the last dimension has stride 1 and
  // each earlier stride is the product of dimensions to its right.
  std::vector<Dim> strides(shape.size(), 1);
  Dim stride = 1;
  for (std::size_t i = shape.size(); i-- > 0;) {
    strides[i] = stride;
    stride *= shape[i];
  }
  return strides;
}

template <typename T>
static AMSTensor makeTensor(std::vector<Dim> shape,
                            const std::vector<T>& values)
{
  // Convert ordinary C++ vectors read from fixture binaries into owned
  // AMSTensors. From this point on, the graph looks like application-provided
  // AMS input rather than test-specific storage.
  std::vector<Dim> strides = contiguousStrides(shape);
  auto tensor = AMSTensor::create<T>(shape, strides, AMSResourceType::AMS_HOST);
  CATCH_REQUIRE(values.size() == static_cast<std::size_t>(tensor.elements()));
  std::copy(values.begin(), values.end(), tensor.template data<T>());
  return tensor;
}

static Dim toDim(std::int64_t value)
{
  CATCH_REQUIRE(value >= 0);
  CATCH_REQUIRE(value <= std::numeric_limits<Dim>::max());
  return static_cast<Dim>(value);
}

static bool hostIsLittleEndian()
{
  // Fixture binaries are written little-endian. AMS currently reads them by
  // copying bytes directly into native scalar arrays, so running this test on a
  // big-endian host should fail explicitly instead of silently byte-swapping
  // wrong values.
  const std::uint16_t value = 1;
  return *reinterpret_cast<const std::uint8_t*>(&value) == 1;
}

static std::uintmax_t dtypeByteWidth(const std::string& dtype)
{
  // The generator only writes float32 tensors for real-valued graph data and
  // int64 tensors for edge_index. Supporting more dtypes would require extending
  // both the manifest contract and the AMSTensor construction below.
  if (dtype == "float32") {
    return sizeof(float);
  }
  if (dtype == "int64") {
    return sizeof(std::int64_t);
  }
  CATCH_FAIL(
      "Unsupported MGN graph diffusion tensor dtype in manifest: " << dtype);
  return 0;
}

static std::uintmax_t shapeElementCount(const std::vector<std::int64_t>& shape)
{
  std::uintmax_t count = 1;
  for (std::int64_t dim : shape) {
    CATCH_REQUIRE(dim >= 0);
    count *= static_cast<std::uintmax_t>(dim);
  }
  return count;
}

static TensorMetadata parseTensorMetadata(const json& tensor)
{
  // Pull the small per-tensor manifest record into a typed C++ struct. This is
  // intentionally separate from validation so missing fields, unsupported
  // dtypes, and wrong shapes fail at the most helpful point in the test.
  CATCH_REQUIRE(tensor.contains("path"));
  CATCH_REQUIRE(tensor.contains("dtype"));
  CATCH_REQUIRE(tensor.contains("shape"));
  CATCH_REQUIRE(tensor.contains("endianness"));
  CATCH_REQUIRE(tensor.contains("byte_size"));

  TensorMetadata metadata;
  metadata.path = tensor.at("path").get<std::string>();
  metadata.dtype = tensor.at("dtype").get<std::string>();
  metadata.shape = tensor.at("shape").get<std::vector<std::int64_t>>();
  metadata.endianness = tensor.at("endianness").get<std::string>();
  metadata.byte_size = tensor.at("byte_size").get<std::uintmax_t>();
  return metadata;
}

static void validateTensorMetadata(
    const TensorMetadata& metadata,
    const std::string& expected_dtype,
    const std::vector<std::int64_t>& expected_shape)
{
  // Validate the manifest before allocating AMSTensors.
  //
  // The binary files deliberately contain no headers: no magic number, no shape,
  // no dtype tag. That keeps them easy for C++ to read without Python or NumPy,
  // but it means fixtures.json is the source of truth. If the manifest says a
  // tensor is [N, 4] float32, this check makes sure the file metadata agrees
  // with exactly the tensor shape this test is about to construct.
  CATCH_REQUIRE_FALSE(metadata.path.empty());
  CATCH_REQUIRE_FALSE(metadata.path.is_absolute());
  for (const auto& part : metadata.path) {
    CATCH_REQUIRE(part.string() != "..");
  }
  CATCH_REQUIRE(metadata.dtype == expected_dtype);
  CATCH_REQUIRE(metadata.shape == expected_shape);

  // The generator writes little-endian bytes. The host-endianness check makes
  // the limitation visible if this test ever runs on a different architecture.
  CATCH_REQUIRE(metadata.endianness == "little");
  CATCH_REQUIRE(hostIsLittleEndian());

  const std::uintmax_t expected_size =
      dtypeByteWidth(metadata.dtype) * shapeElementCount(metadata.shape);
  CATCH_REQUIRE(metadata.byte_size == expected_size);
}

template <typename T>
static std::vector<T> readTensorBinary(
    const std::filesystem::path& manifest_dir,
    const json& tensor,
    const std::string& expected_dtype,
    const std::vector<std::int64_t>& expected_shape)
{
  TensorMetadata metadata = parseTensorMetadata(tensor);
  validateTensorMetadata(metadata, expected_dtype, expected_shape);

  // All tensor paths are relative to fixtures.json. That keeps the generated
  // fixture directory relocatable: moving the build directory as a unit does not
  // invalidate absolute paths baked into the manifest.
  const std::filesystem::path tensor_path = manifest_dir / metadata.path;
  CATCH_REQUIRE(std::filesystem::exists(tensor_path));
  CATCH_REQUIRE(std::filesystem::is_regular_file(tensor_path));
  CATCH_REQUIRE(std::filesystem::file_size(tensor_path) == metadata.byte_size);

  const std::uintmax_t element_count = shapeElementCount(metadata.shape);
  CATCH_REQUIRE(element_count <= std::numeric_limits<std::size_t>::max());
  std::vector<T> values(static_cast<std::size_t>(element_count));

  std::ifstream input(tensor_path, std::ios::binary);
  CATCH_REQUIRE(input);
  if (metadata.byte_size > 0) {
    CATCH_REQUIRE(metadata.byte_size <=
                  static_cast<std::uintmax_t>(
                      std::numeric_limits<std::streamsize>::max()));
    input.read(reinterpret_cast<char*>(values.data()),
               static_cast<std::streamsize>(metadata.byte_size));
    // gcount verifies we actually read the promised number of bytes. Combined
    // with file_size above, this catches truncated or stale fixture files before
    // they can become misleading numerical comparisons.
    CATCH_REQUIRE(static_cast<std::uintmax_t>(input.gcount()) ==
                  metadata.byte_size);
  }
  return values;
}

static json loadManifest(const std::filesystem::path& fixture_dir)
{
  // fixtures.json is generated by the Python fixtures mode. If this file is
  // missing, the C++ test cannot make progress because it does not know which
  // model or tensors to load.
  const std::filesystem::path manifest_path = fixture_dir / "fixtures.json";
  if (!std::filesystem::exists(manifest_path)) {
    CATCH_FAIL("Missing MGN graph diffusion fixtures at "
               << manifest_path
               << ". Run `ctest -R MGN_DIFFUSION_FIXTURES` or the full "
                  "`ctest -R MGN_DIFFUSION` chain.");
  }

  std::ifstream input(manifest_path);
  CATCH_REQUIRE(input);
  json manifest = json::parse(input);

  // Keep the manifest version check close to parsing so future fixture format
  // changes fail loudly instead of being interpreted as the current raw-binary
  // contract.
  CATCH_REQUIRE(manifest.contains("format_version"));
  CATCH_REQUIRE(manifest.contains("endianness"));
  CATCH_REQUIRE(manifest.at("format_version").get<int>() ==
                kFixtureFormatVersion);
  CATCH_REQUIRE(manifest.at("endianness").get<std::string>() == "little");
  CATCH_REQUIRE(manifest.contains("model"));
  CATCH_REQUIRE(manifest.contains("cases"));
  CATCH_REQUIRE(manifest.contains("comparison"));
  CATCH_REQUIRE(manifest.at("cases").is_array());
  return manifest;
}

static std::filesystem::path resolveManifestRelativePath(
    const std::filesystem::path& manifest_dir,
    const json& object)
{
  // Model and tensor paths in the manifest are relative by design. Avoiding
  // absolute paths makes fixtures reusable if the whole build directory moves.
  const std::filesystem::path relative_path =
      object.at("path").get<std::string>();
  CATCH_REQUIRE_FALSE(relative_path.empty());
  CATCH_REQUIRE_FALSE(relative_path.is_absolute());
  for (const auto& part : relative_path) {
    CATCH_REQUIRE(part.string() != "..");
  }
  return manifest_dir / relative_path;
}

static AMSHomogeneousGraph makeGraph(const std::filesystem::path& manifest_dir,
                                     const json& graph_case)
{
  // Runtime fixture binaries become the same AMSTensor-backed homogeneous graph
  // that an application would pass to AMS:
  //
  //   node_features   [N, 4]  -> x, y, u, kappa
  //   edge_index      [2, E]  -> source row, destination row
  //   edge_features   [E, 4]  -> dx, dy, distance, message
  //   global_features [1]     -> dt
  //
  // Keeping this conversion here, instead of compiling arrays into the test,
  // makes the test closer to a real AMS deployment path: model and tensors are
  // runtime files.
  const std::int64_t num_nodes = graph_case.at("num_nodes").get<std::int64_t>();
  const std::int64_t num_edges = graph_case.at("num_edges").get<std::int64_t>();
  const std::int64_t node_dim =
      graph_case.at("node_feature_dim").get<std::int64_t>();
  const std::int64_t edge_dim =
      graph_case.at("edge_feature_dim").get<std::int64_t>();
  const std::int64_t global_dim =
      graph_case.at("global_feature_dim").get<std::int64_t>();

  // These feature dimensions are part of this particular learned example. If
  // the generator changes the graph contract, the C++ parity test should fail
  // and be updated deliberately.
  CATCH_REQUIRE(node_dim == kNodeFeatureDim);
  CATCH_REQUIRE(edge_dim == kEdgeFeatureDim);
  CATCH_REQUIRE(global_dim == kGlobalFeatureDim);

  const json& tensors = graph_case.at("tensors");

  // edge_index is the only integer tensor. It must remain int64 because the
  // TorchScript model canonicalizes and indexes with 64-bit node ids.
  auto node_features = readTensorBinary<float>(manifest_dir,
                                               tensors.at("node_features"),
                                               "float32",
                                               {num_nodes, node_dim});
  auto edge_index = readTensorBinary<std::int64_t>(manifest_dir,
                                                   tensors.at("edge_index"),
                                                   "int64",
                                                   {2, num_edges});
  auto edge_features = readTensorBinary<float>(manifest_dir,
                                               tensors.at("edge_features"),
                                               "float32",
                                               {num_edges, edge_dim});
  auto global_features = readTensorBinary<float>(manifest_dir,
                                                 tensors.at("global_features"),
                                                 "float32",
                                                 {global_dim});

  return AMSHomogeneousGraph(
      makeTensor<float>({toDim(num_nodes), toDim(node_dim)}, node_features),
      makeTensor<std::int64_t>({2, toDim(num_edges)}, edge_index),
      makeTensor<float>({toDim(num_edges), toDim(edge_dim)}, edge_features),
      makeTensor<float>({toDim(global_dim)}, global_features));
}

static std::vector<float> loadReferenceDeltaU(
    const std::filesystem::path& manifest_dir,
    const json& graph_case)
{
  // Reference output is one scalar delta_u per node. It was produced in Python
  // by reloading the exported TorchScript model, so it is the closest available
  // representation of what LibTorch should compute in AMS.
  const std::int64_t num_nodes = graph_case.at("num_nodes").get<std::int64_t>();
  const std::int64_t output_dim =
      graph_case.at("reference_output_dim").get<std::int64_t>();
  CATCH_REQUIRE(output_dim == kReferenceOutputDim);
  return readTensorBinary<float>(manifest_dir,
                                 graph_case.at("tensors").at("reference_delta_"
                                                             "u"),
                                 "float32",
                                 {num_nodes, output_dim});
}

static void verifyDeltaU(const json& graph_case,
                         const std::vector<float>& reference_delta_u,
                         const AMSHomogeneousGraphFields& outputs,
                         double rtol,
                         double atol)
{
  // The reference is the Python TorchScript output, not the exact synthetic
  // diffusion target. Training mode already checks whether the model learned
  // the synthetic formula. This test is about AMS/LibTorch deployment parity.
  const std::int64_t num_nodes = graph_case.at("num_nodes").get<std::int64_t>();
  const std::int64_t output_dim =
      graph_case.at("reference_output_dim").get<std::int64_t>();
  CATCH_REQUIRE(output_dim == kReferenceOutputDim);

  CATCH_REQUIRE(outputs.node_fields.contains("delta_u"));

  // The Python model returns "node:delta_u". AMS strips the "node:" namespace
  // and stores the tensor under outputs.node_fields.at("delta_u").
  const auto& delta_u = outputs.node_fields.at("delta_u");
  CATCH_REQUIRE(delta_u.shape()[0] == num_nodes);
  CATCH_REQUIRE(delta_u.shape()[1] == output_dim);
  CATCH_REQUIRE(reference_delta_u.size() ==
                static_cast<std::size_t>(num_nodes * output_dim));

  const float* actual = delta_u.data<float>();
  const std::int64_t count = num_nodes * output_dim;
  for (std::int64_t i = 0; i < count; ++i) {
    CATCH_REQUIRE(
        actual[i] ==
        Catch::Approx(reference_delta_u[i]).epsilon(rtol).margin(atol));
  }
}

}  // namespace

CATCH_TEST_CASE("AMSExecute homogeneous graph MGN diffusion surrogate",
                "[wf][graph][surrogate][mgn]")
{
  // CMake sets AMS_MGN_DIFFUSION_FIXTURE_DIR to the generated build-tree
  // fixture directory. The manifest inside that directory tells us where the
  // model and per-case tensor files live.
  const std::filesystem::path fixture_dir = AMS_MGN_DIFFUSION_FIXTURE_DIR;
  CATCH_REQUIRE_FALSE(fixture_dir.empty());

  const json manifest = loadManifest(fixture_dir);
  const std::filesystem::path manifest_dir = fixture_dir;

  // The model path also comes from the manifest rather than being hard-coded in
  // the test. That keeps the C++ loader tied to the same artifact table of
  // contents as the tensor files.
  const std::filesystem::path model_path =
      resolveManifestRelativePath(manifest_dir, manifest.at("model"));
  CATCH_REQUIRE(std::filesystem::exists(model_path));
  CATCH_REQUIRE(std::filesystem::is_regular_file(model_path));

  const double rtol = manifest.at("comparison").at("rtol").get<double>();
  const double atol = manifest.at("comparison").at("atol").get<double>();
  CATCH_REQUIRE(manifest.at("cases").size() == 2);

  // Register the exported TorchScript model with AMS and create an executor
  // exactly as an application would before calling AMSExecute.
  //
  // This test runs as a single-process executor, so process_id/world_size are
  // named constants rather than bare 0/1 literals.
  //
  // The threshold argument has no acceptance/rejection meaning for homogeneous
  // graph surrogate execution today: the graph path either runs the model or
  // falls back if the model cannot be used. AMSRegisterAbstractModel still
  // requires a threshold because the API is shared with pointwise surrogates.
  // Use 1.0 here as a readable "accept everything" value for that unused
  // graph-path parameter.
  //
  // The store_data argument is also inherited from the shared pointwise
  // workflow API. It controls whether an AMSWorkflow opens a training-data DB,
  // but this graph parity test is inference-only and requires the fallback
  // callback to remain unused. Current graph fallback data storage is also not
  // implemented, so false keeps the test from opening unrelated DB state.
  AMSInit();
  const std::string model_path_string = model_path.string();
  auto model = AMSRegisterAbstractModel(kDomainName,
                                        kUnusedGraphSurrogateThreshold,
                                        model_path_string.c_str(),
                                        kStoreTrainingData);
  AMSExecutor executor =
      AMSCreateExecutor(model, kSingleRankProcessId, kSingleRankWorldSize);

  // The two fixed fixture sizes intentionally have different N and E values.
  // Running both cases catches accidental static-shape assumptions in the
  // exported TorchScript model or in AMS graph tensor handling.
  //
  // N is checked explicitly here because this test is meant to prove the model
  // handles both fixture graph sizes, not just "whatever happened to be in the
  // manifest".
  const std::vector<std::int64_t> expected_node_counts = {24, 73};
  std::size_t case_index = 0;
  for (const json& graph_case : manifest.at("cases")) {
    CATCH_REQUIRE(graph_case.at("num_nodes").get<std::int64_t>() ==
                  expected_node_counts.at(case_index));
    CATCH_DYNAMIC_SECTION("fixture "
                          << graph_case.at("name").get<std::string>())
    {
      // Load one graph case from runtime files, then run it through AMS. Each
      // case has its own N, E, input tensors, and reference output.
      AMSHomogeneousGraph graph = makeGraph(manifest_dir, graph_case);
      std::vector<float> reference_delta_u =
          loadReferenceDeltaU(manifest_dir, graph_case);

      bool callback_invoked = false;
      // If the surrogate path fails, AMS would call the domain fallback. For
      // this parity test that would hide a deployment failure, so the callback
      // records whether it was invoked and supplies the reference only as a
      // valid fallback value.
      //
      // The assertion below requires callback_invoked == false. That means AMS
      // accepted the graph surrogate output instead of falling back to this
      // lambda.
      HomogeneousGraphDomainFn callback =
          [&](const AMSHomogeneousGraph&, AMSHomogeneousGraphFields& outputs) {
            callback_invoked = true;
            const std::int64_t num_nodes =
                graph_case.at("num_nodes").get<std::int64_t>();
            const std::int64_t output_dim =
                graph_case.at("reference_output_dim").get<std::int64_t>();
            outputs.node_fields.set("delta_u",
                                    makeTensor<float>({toDim(num_nodes),
                                                       toDim(output_dim)},
                                                      reference_delta_u));
          };

      AMSHomogeneousGraphFields outputs;
      AMSExecute(executor, callback, graph, outputs);

      CATCH_REQUIRE_FALSE(callback_invoked);
      verifyDeltaU(graph_case, reference_delta_u, outputs, rtol, atol);
    }
    ++case_index;
  }
}
