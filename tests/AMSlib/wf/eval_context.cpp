#include "wf/eval_context.hpp"

#include <ATen/ATen.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "wf/tensor_bundle.hpp"

CATCH_TEST_CASE("EvalContext default construction", "[evalcontext]")
{
  ams::EvalContext ctx;

  // Bundles should start empty
  CATCH_REQUIRE(ctx.Inputs.empty());
  CATCH_REQUIRE(ctx.Inouts.empty());
  CATCH_REQUIRE(ctx.Outputs.empty());

  // Optional threshold should be disengaged
  CATCH_REQUIRE_FALSE(ctx.Threshold.has_value());

  // Optional uncertainties should be disengaged
  CATCH_REQUIRE_FALSE(ctx.Uncertainties.has_value());

  // Model and layout pointers should be nullptr
  CATCH_REQUIRE(ctx.Model == nullptr);
  CATCH_REQUIRE(ctx.Layout == nullptr);

  // Intermediate tensors should be empty
  CATCH_REQUIRE(ctx.ModelInput.numel() == 0);
  CATCH_REQUIRE(ctx.ModelOutput.numel() == 0);

  // No fallback indices yet
  CATCH_REQUIRE(ctx.FallbackIndices.empty());
}

CATCH_TEST_CASE("EvalContext parameterized construction", "[evalcontext]")
{
  // Prepare bundles
  ams::TensorBundle ins;
  ams::TensorBundle ios;
  ams::TensorBundle outs;

  ins.add("a", at::ones({2}));
  ios.add("b", at::zeros({3}));
  outs.add("c", at::full({1}, 42));

  // Dummy pointers
  ams::ml::InferenceModel* modelPtr = nullptr;
  ams::LayoutTransform* layoutPtr = nullptr;

  // Construct context with threshold
  ams::EvalContext ctx(std::move(ins),
                       std::move(ios),
                       std::move(outs),
                       modelPtr,
                       layoutPtr,
                       0.75f);

  // Bundles moved correctly
  CATCH_REQUIRE(ctx.Inputs.size() == 1);
  CATCH_REQUIRE(ctx.Inouts.size() == 1);
  CATCH_REQUIRE(ctx.Outputs.size() == 1);

  // Threshold exists and is correct
  CATCH_REQUIRE(ctx.Threshold.has_value());
  CATCH_REQUIRE_THAT(ctx.Threshold.value(),
                     Catch::Matchers::WithinAbs(0.75f, 1e-6));

  // Model + layout pointers preserved
  CATCH_REQUIRE(ctx.Model == modelPtr);
  CATCH_REQUIRE(ctx.Layout == layoutPtr);

  // Intermediate tensors must still be empty
  CATCH_REQUIRE(ctx.ModelInput.numel() == 0);
  CATCH_REQUIRE(ctx.ModelOutput.numel() == 0);

  // Uncertainties should be disengaged on construction
  CATCH_REQUIRE_FALSE(ctx.Uncertainties.has_value());

  // No fallback indices yet
  CATCH_REQUIRE(ctx.FallbackIndices.empty());
}

CATCH_TEST_CASE("EvalContext optional uncertainties usage", "[evalcontext]")
{
  ams::EvalContext ctx;

  // Initially no uncertainties
  CATCH_REQUIRE_FALSE(ctx.Uncertainties.has_value());

  // Assign uncertainties tensor
  ctx.Uncertainties = at::full({4}, 0.123f);

  CATCH_REQUIRE(ctx.Uncertainties.has_value());
  CATCH_REQUIRE(ctx.Uncertainties->sizes() == at::IntArrayRef({4}));
  CATCH_REQUIRE(
      at::allclose(*ctx.Uncertainties, at::full({4}, 0.123f), 1e-6, 1e-6));
}
