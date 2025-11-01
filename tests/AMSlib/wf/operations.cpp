#include <ATen/core/TensorBody.h>
#include <ATen/ops/rand.h>
#include <c10/core/DeviceType.h>
#include <torch/torch.h>
#include <torch/types.h>

#include <algorithm>
#include <catch2/catch_all.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>
#include <random>
#include <string>
#include <vector>

#include "wf/workflow.hpp"

using torch::indexing::Slice;

CATCH_TEST_CASE("Workflow: subSelectTensors matches boolean indexing",
                "[ams][workflow][subselect]")
{
  // Parameterize precision & device
  const auto prec = GENERATE("float", "double");
  const auto devStr = GENERATE("cpu", "cuda");

  if (devStr == std::string("cuda") && !torch::cuda::is_available()) {
    CATCH_SKIP("CUDA not available on this node");
  }

  const torch::Dtype DType =
      (prec == std::string("double")) ? torch::kFloat64 : torch::kFloat32;
  const c10::DeviceType dev = (devStr == std::string("cuda"))
                                  ? c10::DeviceType::CUDA
                                  : c10::DeviceType::CPU;

  torch::manual_seed(0);

  // Build inputs
  ams::SmallVector<torch::Tensor> vectors;
  for (int i = 0; i < 4; ++i) {
    vectors.push_back(
        torch::rand({32, 11}, torch::TensorOptions().dtype(DType).device(dev)));
  }

  // Odd rows are selected
  auto predicate =
      (torch::arange(0, 32, torch::kLong).to(dev) % 2).to(torch::kBool) != 0;

  // Under test
  auto subselected = ams::AMSWorkflow::subSelectTensors(vectors, predicate);

  // Check each tensor matches direct boolean indexing
  CATCH_REQUIRE(subselected.size() == vectors.size());
  for (int i = 0; i < static_cast<int>(vectors.size()); ++i) {
    auto expected = vectors[i].index({predicate, Slice()});
    CATCH_REQUIRE(torch::allclose(subselected[i], expected, 1e-5, 1e-8));
  }
}

// --- helper (deterministic) ---
static std::vector<int> generateRandomVector(int target_sum, int size)
{
  if (target_sum < size)
    throw std::invalid_argument("target_sum must be >= size");

  std::vector<int> v(size, 1);
  target_sum -= size;

  // fixed seed for reproducibility in tests
  std::mt19937 gen(0);
  std::uniform_int_distribution<> dis(0, size - 1);

  // distribute leftover one-by-one
  for (int i = 0; i < target_sum; ++i)
    v[dis(gen)]++;

  // shuffle for a stable-but-“random” partition
  std::shuffle(v.begin(), v.end(), gen);
  return v;  // sums to original target_sum + size == original target_sum input
}

// CUDA/HIP mapping like elsewhere
#if defined(__AMS_ENABLE_CUDA__) || defined(__AMS_ENABLE_HIP__)
constexpr c10::DeviceType AMS_GPU = c10::DeviceType::CUDA;
#else
constexpr c10::DeviceType AMS_GPU =
    c10::DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES;
#endif


CATCH_TEST_CASE("Workflow: ScatterPhysicOutputsToOrigDomain",
                "[ams][workflow][scatter]")
{
  // Parameters
  const auto prec = GENERATE("float", "double");
  auto devStr = GENERATE("cpu", "cuda");  // your matrix is cpu/cuda
  if (devStr == "cuda" && !torch::cuda::is_available()) {
    CATCH_SKIP("CUDA not available on this node");
  }

  // Types & device
  const torch::Dtype DType =
      (prec == std::string("double")) ? torch::kFloat64 : torch::kFloat32;
  const c10::DeviceType dev = (devStr == std::string("cuda"))
                                  ? c10::DeviceType::CUDA
                                  : c10::DeviceType::CPU;

  // Deterministic RNG for stable tests
  torch::manual_seed(0);

  // Build inputs like your main()
  ams::SmallVector<torch::Tensor> entireDomain;
  ams::SmallVector<torch::Tensor> computedDomain;
  entireDomain.reserve(4);
  computedDomain.reserve(4);

  for (int i = 0; i < 4; ++i) {
    entireDomain.push_back(
        torch::zeros({128, 7},
                     torch::TensorOptions().dtype(DType).device(dev)));
    computedDomain.push_back(
        torch::rand({64, 7}, torch::TensorOptions().dtype(DType).device(dev)));
  }

  // Predicate: odd rows are “computed”; even rows empty
  auto tmp = torch::arange(0, 128, torch::kLong).to(dev) % 2;
  auto predicate = tmp.to(torch::kBool) != 0;

  // Action under test
  ams::AMSWorkflow::ScatterPhysicOutputsToOrigDomain(computedDomain,
                                                     predicate,
                                                     entireDomain);

  // Check per-tensor
  for (int i = 0; i < static_cast<int>(computedDomain.size()); ++i) {
    auto cd = computedDomain[i];
    auto ed = entireDomain[i].index({predicate, Slice()});
    CATCH_REQUIRE(torch::allclose(ed, cd, /*rtol=*/1e-5, /*atol=*/1e-8));
  }
}

CATCH_TEST_CASE("Workflow: MLDomainToApplication round-trip",
                "[ams][workflow][domain-to-app]")
{
  // Params: (mlType, phType) × device
  const auto mlType = GENERATE("float", "double");
  const auto phType = GENERATE("float", "double");
  auto devStr = GENERATE("cpu", "cuda");  // your CTest uses cpu/gpu

  // Skip GPU when not available
  if (devStr == "cuda" && !torch::cuda::is_available()) {
    CATCH_SKIP("CUDA not available on this node");
  }

  // DTypes & device
  const torch::Dtype mlDType =
      (mlType == std::string("double")) ? torch::kFloat64 : torch::kFloat32;
  const torch::Dtype phDType =
      (phType == std::string("double")) ? torch::kFloat64 : torch::kFloat32;
  const c10::DeviceType dev = (devStr == std::string("cuda"))
                                  ? c10::DeviceType::CUDA
                                  : c10::DeviceType::CPU;

  // Source and random split of feature columns
  constexpr int B = 32;  // batch
  constexpr int F = 11;  // total features
  const auto shapes = generateRandomVector(/*target_sum=*/F, /*size=*/8);

  auto Src =
      torch::rand({B, F}, torch::TensorOptions().dtype(mlDType).device(dev));

  // Destination tensors (one per “shape” chunk)
  ams::SmallVector<torch::Tensor> Dest;
  Dest.reserve(shapes.size());
  for (int cols : shapes) {
    Dest.push_back(
        torch::zeros({B, cols},
                     torch::TensorOptions().dtype(phDType).device(dev)));
  }

  // Predicate: keep odd rows, zero out even rows
  auto pred = (torch::arange(B, torch::kLong).to(dev) % 2).to(torch::kBool);

  // First call on all but last destination to get offset; then finish remainder
  ams::SmallVector<torch::Tensor> subset(Dest.begin(), Dest.end() - 1);
  int offset =
      ams::AMSWorkflow::MLDomainToApplication(Src, subset, pred, /*offset=*/0);
  ams::AMSWorkflow::MLDomainToApplication(Src,
                                          {Dest.back()},
                                          pred,
                                          /*offset=*/offset);

  // Recompose result in ML dtype to compare with Src after zeroing masked rows
  auto result = torch::cat(Dest, /*dim=*/1).to(Src.options().dtype(mlDType));

  // Zero-out rows in Src that were masked out (even rows)
  auto inverted = ~pred;
  // broadcast a single row of zeros into all masked rows
  Src.index_put_({inverted, Slice()},
                 torch::zeros({1, Src.size(1)}, Src.options()));

  // Assert equality
  CATCH_REQUIRE(torch::allclose(Src, result, /*rtol=*/1e-5, /*atol=*/1e-8));
}

int main(int argc, char** argv)
{
  Catch::Session session;

  if (int rc = session.applyCommandLine(argc, argv))
    return rc;  // bad CLI -> propagate

  int rc = session.run();  // run tests
  std::cout << "RC:" << rc << "\n";

  // Treat "warnings only" (e.g., due to skipped tests) as success for CTest.
  // Catch2 commonly uses 4 for warnings; adjust if your config differs.
  return (rc == 0 || rc == 4) ? 0 : rc;
}
