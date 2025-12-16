#include "wf/pointwise_layout_transform.hpp"

#include <ATen/ATen.h>
#include <torch/script.h>
#include <torch/torch.h>
using namespace torch::indexing;

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::WithinAbs;

// -----------------------------------------------------------------------------
// TEST 1: pack() builds correct mapping and concatenation
// -----------------------------------------------------------------------------
CATCH_TEST_CASE("PointwiseConcatTransform pack()", "[layout][concat]")
{
  ams::PointwiseConcatTransform lt;

  ams::TensorBundle ins;
  ams::TensorBundle ios;

  // Shapes: a -> [4], b -> [4,3], c -> [4]
  ins.add("a", at::ones({4}));           // 1 column
  ins.add("b", at::full({4, 3}, 2.0f));  // 3 columns
  ios.add("c", at::full({4}, 3.0f));     // 1 column

  at::Tensor model_input;
  auto MapOrErr = lt.pack(ins, ios, model_input);
  if (!MapOrErr) std::cout << MapOrErr.error() << "\n";
  CATCH_REQUIRE(MapOrErr);
  ams::IndexMap map = std::move(*MapOrErr);
  auto A = model_input.index({Slice(), Slice(0, 1)});
  auto B = model_input.index({Slice(), Slice(1, 4)});
  auto C = model_input.index({Slice(), Slice(4, 5)});
  CATCH_REQUIRE(model_input.sizes() == at::IntArrayRef({4, 5}));

  // Expected IndexMap:
  //  a: offset 0, cols 1
  //  b: offset 1, cols 3
  //  c: offset 4, cols 1
  CATCH_REQUIRE(map.Fields.size() == 3);

  CATCH_REQUIRE(map.Fields[0].Name == "a");
  CATCH_REQUIRE(map.Fields[0].Offset == 0);
  CATCH_REQUIRE(map.Fields[0].Cols == 1);

  CATCH_REQUIRE(map.Fields[1].Name == "b");
  CATCH_REQUIRE(map.Fields[1].Offset == 1);
  CATCH_REQUIRE(map.Fields[1].Cols == 3);

  CATCH_REQUIRE(map.Fields[2].Name == "c");
  CATCH_REQUIRE(map.Fields[2].Offset == 4);
  CATCH_REQUIRE(map.Fields[2].Cols == 1);
  CATCH_REQUIRE(torch::equal(A.squeeze(), ins[0].tensor));
  CATCH_REQUIRE(torch::equal(B, ins[1].tensor));
  CATCH_REQUIRE(torch::equal(C.squeeze(), ios[0].tensor));
}

// -----------------------------------------------------------------------------
// TEST 2: unpack() with only predictions
// -----------------------------------------------------------------------------
CATCH_TEST_CASE("PointwiseConcatTransform unpack() predictions only",
                "[layout][concat]")
{

  ams::PointwiseConcatTransform lt;
  // Prepare pack
  at::Tensor pred = at::full({6, 6}, 7.0f);

  ams::TensorBundle outs, inouts;
  outs.add("a", torch::Tensor());
  outs.add("b", torch::Tensor());
  outs.add("c", torch::Tensor());
  outs.add("d", torch::Tensor());
  inouts.add("e", at::full({1, 6}, 1.0f));
  inouts.add("f", at::full({1, 6}, 1.0f));

  std::optional<at::Tensor> uncrt_out;

  auto res = lt.unpack(pred, outs, inouts, uncrt_out);
  CATCH_REQUIRE(res.has_value());

  at::Tensor corr = at::full({1, 6}, 7.0f);
  for (auto& V : outs) {
    CATCH_REQUIRE(at::allclose(V.tensor, corr));
  }

  for (auto& V : inouts) {
    CATCH_REQUIRE(at::allclose(V.tensor, corr));
  }

  CATCH_REQUIRE_FALSE(uncrt_out.has_value());
}

// -----------------------------------------------------------------------------
// TEST 3: unpack() with uncertainty tuple
// -----------------------------------------------------------------------------
CATCH_TEST_CASE("PointwiseConcatTransform unpack() with uncertainty",
                "[layout][concat][uncertainty]")
{

  ams::PointwiseConcatTransform lt;

  // Prepare pack
  at::Tensor pred = at::full({6, 6}, 7.0f);
  at::Tensor uncrt = at::full({6}, 0.25f);

  torch::jit::IValue iv{c10::ivalue::Tuple::create({pred, uncrt})};

  ams::TensorBundle outs, inouts;
  outs.add("a", torch::Tensor());
  outs.add("b", torch::Tensor());
  outs.add("c", torch::Tensor());
  outs.add("d", torch::Tensor());
  inouts.add("e", at::full({1, 6}, 1.0f));
  inouts.add("f", at::full({1, 6}, 1.0f));

  std::optional<at::Tensor> uncrt_out;

  auto res = lt.unpack(iv, outs, inouts, uncrt_out);
  CATCH_REQUIRE(res.has_value());

  at::Tensor corr = at::full({1, 6}, 7.0f);
  for (auto& V : outs) {
    CATCH_REQUIRE(at::allclose(V.tensor, corr));
  }

  for (auto& V : inouts) {
    CATCH_REQUIRE(at::allclose(V.tensor, corr));
  }

  CATCH_REQUIRE(uncrt_out.has_value());
  CATCH_REQUIRE(at::allclose(*uncrt_out, uncrt));
}

// -----------------------------------------------------------------------------
// TEST 4: pack() errors on 0-dim tensors (dim < 1)
// -----------------------------------------------------------------------------
CATCH_TEST_CASE("PointwiseConcatTransform pack() rejects scalar 0-dim tensor",
                "[layout][concat][error]")
{
  ams::PointwiseConcatTransform lt;

  ams::TensorBundle ins, ios;

  // 0-dim tensor: dim() == 0  -> should error
  ins.add("bad", at::scalar_tensor(1.0f));

  at::Tensor model_input;
  auto res = lt.pack(ins, ios, model_input);

  CATCH_REQUIRE_FALSE(res);
  CATCH_REQUIRE(res.error().getType() == ams::AMSErrorType::InvalidShapes);
  CATCH_REQUIRE_THAT(res.error().getMessage(),
                     ContainsSubstring("must have at least 1 dimension"));
}

// -----------------------------------------------------------------------------
// TEST 5: pack() errors on empty Inputs + InOuts (no columns)
// -----------------------------------------------------------------------------
CATCH_TEST_CASE("PointwiseConcatTransform pack() rejects empty bundles",
                "[layout][concat][error]")
{
  ams::PointwiseConcatTransform lt;

  ams::TensorBundle ins, ios;
  at::Tensor model_input;

  auto res = lt.pack(ins, ios, model_input);

  CATCH_REQUIRE_FALSE(res);
  CATCH_REQUIRE(res.error().getType() == ams::AMSErrorType::InvalidShapes);
  CATCH_REQUIRE_THAT(res.error().getMessage(),
                     ContainsSubstring("expected at least a single dimension"));
}

// -----------------------------------------------------------------------------
// TEST 6: unpack() errors when ModelOutput is not tensor or tuple
// -----------------------------------------------------------------------------
CATCH_TEST_CASE(
    "PointwiseConcatTransform unpack() rejects non-tensor non-tuple",
    "[layout][concat][error]")
{
  ams::PointwiseConcatTransform lt;

  // e.g., int IValue
  torch::jit::IValue iv(123);

  ams::TensorBundle outs, inouts;
  outs.add("a", torch::Tensor());
  inouts.add("b", torch::Tensor());

  std::optional<at::Tensor> uncrt_out;

  auto st = lt.unpack(iv, outs, inouts, uncrt_out);

  CATCH_REQUIRE_FALSE(st);
  CATCH_REQUIRE(st.error().getType() == ams::AMSErrorType::InvalidShapes);
  CATCH_REQUIRE_THAT(st.error().getMessage(),
                     ContainsSubstring("ModelOutput must be"));
}

// -----------------------------------------------------------------------------
// TEST 7: unpack() errors when tuple size != 2
// -----------------------------------------------------------------------------
CATCH_TEST_CASE("PointwiseConcatTransform unpack() rejects tuple size != 2",
                "[layout][concat][error]")
{
  ams::PointwiseConcatTransform lt;

  at::Tensor pred = at::full({2, 2}, 7.0f);
  at::Tensor uncrt = at::full({2}, 0.25f);
  at::Tensor extra = at::zeros({1});

  torch::jit::IValue iv{
      c10::ivalue::Tuple::create({pred, uncrt, extra})  // size 3
  };

  ams::TensorBundle outs, inouts;
  outs.add("a", torch::Tensor());
  outs.add("b", torch::Tensor());
  inouts.add("c", torch::Tensor());
  inouts.add("d", torch::Tensor());

  std::optional<at::Tensor> uncrt_out;

  auto st = lt.unpack(iv, outs, inouts, uncrt_out);

  CATCH_REQUIRE_FALSE(st);
  CATCH_REQUIRE(st.error().getType() == ams::AMSErrorType::InvalidShapes);
  CATCH_REQUIRE_THAT(st.error().getMessage(),
                     ContainsSubstring("expected tuple(pred,uncertainty)"));
}

// -----------------------------------------------------------------------------
// TEST 8: unpack() errors when output columns don't match Outs+InOuts
// -----------------------------------------------------------------------------
CATCH_TEST_CASE(
    "PointwiseConcatTransform unpack() rejects mismatched output width",
    "[layout][concat][error]")
{
  ams::PointwiseConcatTransform lt;

  // ModelOut has 3 columns
  at::Tensor pred = at::full({5, 3}, 7.0f);

  // But Outs+InOuts expects 4 fields -> mismatch
  ams::TensorBundle outs, inouts;
  outs.add("a", torch::Tensor());
  outs.add("b", torch::Tensor());
  inouts.add("c", torch::Tensor());
  inouts.add("d", torch::Tensor());

  std::optional<at::Tensor> uncrt_out;

  auto st = lt.unpack(pred, outs, inouts, uncrt_out);

  CATCH_REQUIRE_FALSE(st);
  CATCH_REQUIRE(st.error().getType() == ams::AMSErrorType::InvalidShapes);
  CATCH_REQUIRE_THAT(st.error().getMessage(),
                     ContainsSubstring("Expected the output size"));
}
