#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <filesystem>
#include <nlohmann/json.hpp>

#include "AMSError.hpp"
#include "ml/AbstractModel.hpp"
#include "ml/Model.hpp"
#include "models/GeneratedBaseModels.hpp"

using namespace ams;
using namespace ams::ml;

//
// Test helpers from BaseModel tests
//
#define REQUIRE_AMS_OK(expr)                                      \
  do {                                                            \
    auto&& _res = (expr);                                         \
    if (!_res) {                                                  \
      CATCH_INFO("AMS error from `" #expr "`: " << _res.error()); \
    }                                                             \
    CATCH_REQUIRE(_res.has_value());                              \
  } while (false)

template <typename T>
T REQUIRE_AMS_VALUE(ams::AMSExpected<T>&& exp, const char* expr_str)
{
  if (!exp) {
    CATCH_INFO("AMS error from `" << expr_str << "`: " << exp.error());
    CATCH_REQUIRE(exp.has_value());
  }
  return std::move(exp.value());
}
#define REQUIRE_AMS_VALUE_EXPR(expr) REQUIRE_AMS_VALUE((expr), #expr)

//
// -----------------------------------------------------------------------------
//  TEST 1: InferenceModel type traits (move-only semantics)
// -----------------------------------------------------------------------------
//
CATCH_TEST_CASE("InferenceModel type traits", "[InferenceModel][traits]")
{
  CATCH_STATIC_REQUIRE(!std::is_copy_constructible_v<InferenceModel>);
  CATCH_STATIC_REQUIRE(!std::is_copy_assignable_v<InferenceModel>);
  CATCH_STATIC_REQUIRE(std::is_move_constructible_v<InferenceModel>);
  CATCH_STATIC_REQUIRE(std::is_move_assignable_v<InferenceModel>);
}

//
// -----------------------------------------------------------------------------
//  TEST 2: InferenceModel::load works (mirrors BaseModel behavior)
// -----------------------------------------------------------------------------
//
CATCH_TEST_CASE("InferenceModel::load succeeds for simple models",
                "[InferenceModel][load]")
{
  for (const auto& tm : simple_models) {
    CATCH_DYNAMIC_SECTION("Inference model: " << tm)
    {
      // Skip GPU models if CUDA unavailable
      if (tm.ModelDevice == "gpu" && !torch::cuda::is_available()) {
        CATCH_WARN("Skipping GPU model (CUDA unavailable): " << tm);
        continue;
      }

      AbstractModel desc(tm.ModelPath,
                         tm.ModelPrecision + "_" + tm.ModelDevice,
                         0);

      std::unique_ptr<InferenceModel> modelUPtr =
          REQUIRE_AMS_VALUE_EXPR(InferenceModel::load(desc));
      CATCH_REQUIRE(modelUPtr);
      auto& M = *modelUPtr;

      // dtype
      if (tm.ModelPrecision == "single")
        CATCH_CHECK(M.getDType() == c10::ScalarType::Float);
      else
        CATCH_CHECK(M.getDType() == c10::ScalarType::Double);

      // device
      if (tm.ModelDevice == "cpu")
        CATCH_CHECK(M.getDevice().is_cpu());
      else
        CATCH_CHECK(!M.getDevice().is_cpu());
    }
  }
}

//
// -----------------------------------------------------------------------------
//  TEST 3: operator()(Tensor)
// -----------------------------------------------------------------------------
//
CATCH_TEST_CASE("InferenceModel single-tensor operator()",
                "[InferenceModel][operator()]")
{
  for (const auto& tm : simple_models) {
    if (tm.ModelDevice == "gpu" && !torch::cuda::is_available()) continue;

    CATCH_DYNAMIC_SECTION("Model: " << tm)
    {
      AbstractModel desc(tm.ModelPath,
                         tm.ModelPrecision + "_" + tm.ModelDevice,
                         0);

      auto IMExp = InferenceModel::load(desc);
      REQUIRE_AMS_OK(IMExp);
      auto& M = *IMExp.value();

      torch::Tensor X = torch::ones({4}, M.getDType()).to(M.getDevice());

      auto outExp = M(X);
      REQUIRE_AMS_OK(outExp);

      auto out = outExp.value().toTensor();
      CATCH_CHECK(out.numel() == 2);
    }
  }
}

//
// -----------------------------------------------------------------------------
//  TEST 4: operator()(vector<IValue>)
// -----------------------------------------------------------------------------
//
CATCH_TEST_CASE("InferenceModel vector<IValue> operator()",
                "[InferenceModel][operator()][vector]")
{
  for (const auto& tm : simple_models) {
    if (tm.ModelDevice == "gpu" && !torch::cuda::is_available()) continue;

    CATCH_DYNAMIC_SECTION("Model: " << tm)
    {
      AbstractModel desc(tm.ModelPath,
                         tm.ModelPrecision + "_" + tm.ModelDevice,
                         0);

      auto IMExp = InferenceModel::load(desc);
      REQUIRE_AMS_OK(IMExp);
      auto& M = *IMExp.value();

      torch::Tensor X = torch::rand({4}, M.getDType()).to(M.getDevice());
      std::vector<torch::jit::IValue> inputs{X};

      auto outExp = M(std::move(inputs));
      REQUIRE_AMS_OK(outExp);

      auto out = outExp.value().toTensor();
      CATCH_CHECK(out.numel() == 2);
    }
  }
}

//
// -----------------------------------------------------------------------------
//  TEST 5: variadic operator()(t1, t2, ...)
//
//  If your test TorchScript models support only 1 input, this still tests the
//  variadic version with one argument.
// -----------------------------------------------------------------------------
//
CATCH_TEST_CASE("InferenceModel variadic operator()",
                "[InferenceModel][variadic]")
{
  for (const auto& tm : simple_models) {
    if (tm.ModelDevice == "gpu" && !torch::cuda::is_available()) continue;

    CATCH_DYNAMIC_SECTION("Model: " << tm)
    {
      AbstractModel desc(tm.ModelPath,
                         tm.ModelPrecision + "_" + tm.ModelDevice,
                         0);

      auto IMExp = InferenceModel::load(desc);
      REQUIRE_AMS_OK(IMExp);
      auto& M = *IMExp.value();

      torch::Tensor X = torch::rand({4}, M.getDType()).to(M.getDevice());

      auto outExp = M(X);  // uses the variadic operator()
      REQUIRE_AMS_OK(outExp);

      auto out = outExp.value().toTensor();
      // Our tests produce 2 outputs
      CATCH_CHECK(out.numel() == 2);
    }
  }
}

//
// -----------------------------------------------------------------------------
//  TEST 6: Error propagation (TorchInternal when inference fails)
// -----------------------------------------------------------------------------
CATCH_TEST_CASE("InferenceModel error propagation", "[InferenceModel][error]")
{
  const auto& tm = simple_models.front();
  if (tm.ModelDevice == "gpu" && !torch::cuda::is_available()) return;

  CATCH_DYNAMIC_SECTION("Testing errors for model: " << tm)
  {
    AbstractModel desc(tm.ModelPath,
                       tm.ModelPrecision + "_" + tm.ModelDevice,
                       0);

    auto IMExp = InferenceModel::load(desc);
    REQUIRE_AMS_OK(IMExp);
    auto& M = *IMExp.value();

    // invalid size tensor (likely triggers TorchScript exception)
    torch::Tensor bad = torch::rand({999, 999});

    auto outExp = M(bad);
    CATCH_REQUIRE_FALSE(outExp.has_value());
    CATCH_CHECK(outExp.error().getType() == AMSErrorType::TorchInternal);
  }
}

//
// -----------------------------------------------------------------------------
//  TEST 7: Eval-mode consistency (model must stay in eval mode)
// -----------------------------------------------------------------------------
CATCH_TEST_CASE("InferenceModel stays in eval mode", "[InferenceModel][eval]")
{
  const auto& tm = simple_models.front();
  if (tm.ModelDevice == "gpu" && !torch::cuda::is_available()) return;

  AbstractModel desc(tm.ModelPath, tm.ModelPrecision + "_" + tm.ModelDevice, 0);

  auto IMExp = InferenceModel::load(desc);
  REQUIRE_AMS_OK(IMExp);
  auto& M = *IMExp.value();

  torch::Tensor X = torch::ones({4}, M.getDType()).to(M.getDevice());

  auto outExp = M(X);
  REQUIRE_AMS_OK(outExp);

  CATCH_CHECK(M.isTraining() == false);
}
