#include <torch/torch.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <iostream>
#include <stdexcept>
#include <string>

#include "AMS.h"
#include "ml/surrogate.hpp"
#include "models/simple_models.hpp"

static void verify(torch::Tensor& input,
                   torch::Tensor& output,
                   torch::Tensor& predicate,
                   float threshold)
{

  CATCH_REQUIRE(torch::equal(input, output));
  const double probability = predicate.to(torch::kDouble).mean().item<double>();
  CATCH_INFO("probability=" << probability << " threshold=" << threshold);
  CATCH_REQUIRE(probability == Catch::Approx(threshold).margin(0.05));
}


CATCH_TEST_CASE(
    "SurrogateModel echoes input and predicate mean matches threshold",
    "[ams][surrogate]")
{
  // Pick one model entry from your generated list
  // This is simillar to iterating over the vector and executing one test at the time
  auto model_desc = GENERATE_COPY(from_range(simple_models));

  // Load model; if UQ needs to be passed, do it here (depends on your API)
  auto model = SurrogateModel::getInstance(model_desc.ModelPath);
  CATCH_REQUIRE(model != nullptr);

  auto model_type = model->getModelDataType();
  auto model_device = model->getModelResourceType();

  // Build a deterministic input (rand is fine too)
  static const std::vector<int64_t> kInputShape = {2000, 8};
  CATCH_SECTION("Model precision matches declaration: " +
                Catch::StringMaker<test_models>::convert(model_desc))
  {
    if (model_desc.ModelPrecision == "single") {
      CATCH_REQUIRE(model->is_float());
    } else if (model_desc.ModelPrecision == "double") {
      CATCH_REQUIRE(model->is_double());
    } else {
      CATCH_FAIL("Unsupported precision string: " << model_desc.ModelPrecision);
    }
  }


  CATCH_SECTION("Model device matches declaration: " +
                Catch::StringMaker<test_models>::convert(model_desc))
  {
    if (model_desc.ModelDevice == "gpu") {
      CATCH_REQUIRE(model->is_gpu());
    } else if (model_desc.ModelDevice == "cpu") {
      CATCH_REQUIRE(model->is_cpu());
    } else {
      CATCH_FAIL("Unsupported device type" << model_desc.ModelPrecision);
    }
  }

  torch::Tensor input = torch::rand(kInputShape,
                                    torch::TensorOptions()
                                        .dtype(std::get<1>(model_type))
                                        .device(std::get<1>(model_device)));


  CATCH_DYNAMIC_SECTION(model_desc)
  {
    CATCH_SECTION("threshold=0.0")
    {
      auto [out, pred] = model->_evaluate(input, 0.0);
      verify(input, out, pred, 0.0);
    }
    CATCH_SECTION("threshold=0.5")
    {
      auto [out, pred] = model->_evaluate(input, 0.5);
      verify(input, out, pred, 0.5);
    }
    CATCH_SECTION("threshold=1.0")
    {
      auto [out, pred] = model->_evaluate(input, 1.0);
      verify(input, out, pred, 1.0);
    }
  }
}
