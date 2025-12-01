#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <type_traits>

#include "AMSError.hpp"
#include "ml/AbstractModel.hpp"
#include "ml/Model.hpp"
#include "models/GeneratedBaseModels.hpp"

using ams::AMSErrorType;
using ams::AMSExpected;
using ams::ml::AbstractModel;
using ams::ml::BaseModel;


using json = nlohmann::json;
using namespace ams::ml;


// Helper macro: check an AMSExpected<T>, print the error if it fails.
#define REQUIRE_AMS_OK(expr)                                      \
  do {                                                            \
    auto _res = (expr);                                           \
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
    CATCH_REQUIRE(exp.has_value());  // will fail and show INFO
  }
  return std::move(exp.value());
}

#define REQUIRE_AMS_VALUE_EXPR(expr) REQUIRE_AMS_VALUE((expr), #expr)

// Helper: map string → AMSErrorType for failing_models
static ams::AMSErrorType parseErrorType(const std::string& s)
{
  using ams::AMSErrorType;

  if (s == "FileDoesNotExist") return AMSErrorType::FileDoesNotExist;
  if (s == "TorchInternal") return AMSErrorType::TorchInternal;
  if (s == "InvalidModel") return AMSErrorType::InvalidModel;
  if (s == "Generic") return AMSErrorType::Generic;

  // Fallback — useful when header and test get out of sync
  return AMSErrorType::Generic;
}

CATCH_TEST_CASE("AbstractModel parses path and name from JSON")
{
  auto tmp = std::filesystem::temp_directory_path() / "dummy_model.pt";
  std::ofstream(tmp.string()) << "dummy";  // ensure it exists

  json j;
  j["model_path"] = tmp.string();
  j["model_name"] = "test-model";

  AbstractModel m(j);

  CATCH_CHECK(m.getPath() == tmp);
  CATCH_REQUIRE(m.getName().has_value());
  CATCH_CHECK(m.getName().value() == "test-model");
}

CATCH_TEST_CASE("AbstractModel handles missing name and path")
{
  json j = json::object();  // no model_path, no model_name

  AbstractModel m(j);
  CATCH_CHECK(m.getPath().empty());
  CATCH_CHECK_FALSE(m.getName().has_value());
}

CATCH_TEST_CASE("BaseModel type traits")
{
  CATCH_STATIC_REQUIRE(!std::is_copy_constructible_v<ams::ml::BaseModel>);
  CATCH_STATIC_REQUIRE(!std::is_copy_assignable_v<ams::ml::BaseModel>);
  CATCH_STATIC_REQUIRE(std::is_move_constructible_v<ams::ml::BaseModel>);
  CATCH_STATIC_REQUIRE(std::is_move_assignable_v<ams::ml::BaseModel>);
}


CATCH_TEST_CASE("BaseModel::load succeeds for simple models",
                "[BaseModel][load]")
{
  for (const auto& tm : simple_models) {
    CATCH_DYNAMIC_SECTION("Simple model: " << tm)
    {
      // Skip GPU models if CUDA is not available.
      if (tm.ModelDevice == "gpu" && !torch::cuda::is_available()) {
        CATCH_WARN("Skipping GPU model (CUDA not available): " << tm);
        continue;
      }

      // Build descriptor (name = precision/device combo, version = 0)
      AbstractModel desc(tm.ModelPath,
                         tm.ModelPrecision + "_" + tm.ModelDevice,
                         /*Version=*/0);

      std::unique_ptr<BaseModel> modelUPtr =
          REQUIRE_AMS_VALUE_EXPR(BaseModel::load(desc));
      CATCH_REQUIRE(modelUPtr);
      auto& model = *modelUPtr;


      // ---- Check dtype ----
      if (tm.ModelPrecision == "single") {
        CATCH_CHECK(model.getDType() == c10::ScalarType::Float);
        CATCH_CHECK(model.isType<float>());
        CATCH_CHECK_FALSE(model.isType<double>());
      } else if (tm.ModelPrecision == "double") {
        CATCH_CHECK(model.getDType() == c10::ScalarType::Double);
        CATCH_CHECK(model.isType<double>());
        CATCH_CHECK_FALSE(model.isType<float>());
      } else {
        CATCH_FAIL("Unexpected ModelPrecision value in TestModel: "
                   << tm.ModelPrecision);
      }

      // ---- Check device ----
      if (tm.ModelDevice == "cpu") {
        CATCH_CHECK(model.getDevice().is_cpu());
        CATCH_CHECK_FALSE(
            model.isDevice());  // BaseModel::isDevice() == !is_cpu()
      } else if (tm.ModelDevice == "gpu") {
        CATCH_CHECK_FALSE(model.getDevice().is_cpu());
        CHECK(model.isDevice());
      } else {
        CATCH_FAIL(
            "Unexpected ModelDevice value in TestModel: " << tm.ModelDevice);
      }
    }
  }
}

CATCH_TEST_CASE("BaseModel::convertTo changes dtype correctly",
                "[BaseModel][convertTo]")
{
  for (const auto& tm : simple_models) {
    CATCH_DYNAMIC_SECTION("Convert model: " << tm)
    {
      if (tm.ModelDevice == "gpu" && !torch::cuda::is_available()) {
        CATCH_WARN("Skipping GPU model (CUDA not available): " << tm);
        continue;
      }

      AbstractModel desc(tm.ModelPath,
                         tm.ModelPrecision + "_" + tm.ModelDevice,
                         /*Version=*/0);

      auto result = BaseModel::load(desc);
      if (!result) {
        CATCH_INFO("BaseModel::load failed with error: " << result.error());
      }
      CATCH_REQUIRE(result);
      auto& model = *result.value();

      // Convert everything to double on same device
      auto status = model.convertTo<double>();
      if (!status) {
        CATCH_INFO("BaseModel::load failed with error: " << status.error());
      }
      CATCH_REQUIRE(status.has_value());  // AMSStatus is AMSExpected<void>

      CATCH_CHECK(model.getDType() == c10::ScalarType::Double);
      CATCH_CHECK(model.isType<double>());
      CATCH_CHECK_FALSE(model.isType<float>());

      // Now convert to float on same device
      status = model.convertTo<float>();
      CATCH_REQUIRE(status.has_value());

      CATCH_CHECK(model.getDType() == c10::ScalarType::Float);
      CATCH_CHECK(model.isType<float>());
      CATCH_CHECK_FALSE(model.isType<double>());
    }
  }
}

CATCH_TEST_CASE("BaseModel::load reports expected errors for failing models",
                "[BaseModel][load][error]")
{
  for (const auto& fm : failing_models) {
    CATCH_DYNAMIC_SECTION("Failing model: " << fm)
    {
      AbstractModel desc(fm.ModelPath, /*Name=*/std::nullopt, /*Version=*/0);

      auto result = BaseModel::load(desc);

      CATCH_REQUIRE_FALSE(result.has_value());  // all failing_models must fail

      const auto& err = result.error();
      auto expectedType = parseErrorType(fm.ExpectedErrorType);

      CATCH_CHECK(err.getType() == expectedType);
    }
  }
}
