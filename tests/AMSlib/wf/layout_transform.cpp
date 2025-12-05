#include "wf/layout_transform.hpp"

#include <ATen/ATen.h>
#include <torch/script.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "wf/tensor_bundle.hpp"

using Catch::Matchers::WithinAbs;

namespace
{

/// Dummy transform that:
///   - pack() returns a constant tensor {9, 9}
///   - unpack() expects an IValue tuple: (prediction, uncertainty)
///   - writes prediction → Outputs
///   - writes uncertainty → Uncertainties
class DummyLayoutTransform : public ams::LayoutTransform
{
public:
  at::Tensor pack(const ams::TensorBundle& Inputs,
                  const ams::TensorBundle& Inouts) override
  {
    // A predictable packed tensor for test
    return at::full({2}, 9.0f);
  }

  void unpack(const torch::jit::IValue& iv,
              ams::TensorBundle& Outputs,
              ams::TensorBundle& Inouts,
              std::optional<at::Tensor>& Uncertainties) override
  {
    // Expect a tuple of 2 tensors
    auto tup = iv.toTuple();
    auto pred = tup->elements()[0].toTensor();
    auto uncrt = tup->elements()[1].toTensor();

    // Output bundle receives the prediction
    Outputs.add("pred", pred);

    // Uncertainties receives the uncertainty tensor
    Uncertainties = uncrt;
  }

  const char* name() const noexcept override { return "DummyLayoutTransform"; }
};

}  // namespace

// -----------------------------------------------------------------------------
// TESTS
// -----------------------------------------------------------------------------

CATCH_TEST_CASE("LayoutTransform pack() returns model input tensor", "[layout]")
{
  DummyLayoutTransform lt;

  ams::TensorBundle ins;
  ams::TensorBundle ios;

  ins.add("a", at::ones({1}));
  ios.add("b", at::zeros({1}));

  at::Tensor packed = lt.pack(ins, ios);

  CATCH_REQUIRE(packed.sizes() == at::IntArrayRef({2}));
  CATCH_REQUIRE_THAT(packed[0].item<float>(), WithinAbs(9.0f, 1e-6f));
  CATCH_REQUIRE_THAT(packed[1].item<float>(), WithinAbs(9.0f, 1e-6f));
}

CATCH_TEST_CASE("LayoutTransform unpack() populates Outputs + Uncertainties",
                "[layout]")
{
  DummyLayoutTransform lt;

  // Dummy prediction + uncertainty tensors
  at::Tensor pred = at::full({3}, 42.0f);
  at::Tensor uncrt = at::full({3}, 0.5f);

  // Construct an IValue tuple: (prediction, uncertainty)
  auto tup = c10::ivalue::Tuple::create({pred, uncrt});
  torch::jit::IValue iv(tup);

  ams::TensorBundle outs;
  ams::TensorBundle ios;  // not modified by dummy
  std::optional<at::Tensor> uncertainties;

  lt.unpack(iv, outs, ios, uncertainties);

  // Output extracted correctly
  CATCH_REQUIRE(outs.size() == 1);
  CATCH_REQUIRE(outs[0].name == "pred");
  CATCH_REQUIRE(outs[0].tensor.sizes() == at::IntArrayRef({3}));
  CATCH_REQUIRE_THAT(outs[0].tensor[0].item<float>(), WithinAbs(42.0f, 1e-6f));

  // Uncertainty extracted correctly
  CATCH_REQUIRE(uncertainties.has_value());
  CATCH_REQUIRE(at::allclose(*uncertainties, at::full({3}, 0.5f)));
}

CATCH_TEST_CASE("LayoutTransform name() returns identifier", "[layout]")
{
  DummyLayoutTransform lt;
  CATCH_REQUIRE(std::string(lt.name()) == "DummyLayoutTransform");
}
