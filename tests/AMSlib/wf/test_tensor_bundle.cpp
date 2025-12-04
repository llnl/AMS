#include <ATen/ATen.h>

#include <catch2/catch_test_macros.hpp>

#include "wf/tensor_bundle.hpp"  // adjust include path accordingly

CATCH_TEST_CASE("TensorBundle basic construction", "[tensorbundle]")
{
  ams::TensorBundle tb;

  CATCH_REQUIRE(tb.size() == 0);
  CATCH_REQUIRE(tb.empty());
}

CATCH_TEST_CASE("TensorBundle add and access items", "[tensorbundle]")
{
  ams::TensorBundle tb;

  at::Tensor t1 = at::ones({3});
  at::Tensor t2 = at::zeros({2});

  tb.add("a", t1);
  tb.add("b", t2);

  CATCH_REQUIRE(tb.size() == 2);
  CATCH_REQUIRE_FALSE(tb.empty());

  CATCH_REQUIRE(tb[0].name == "a");
  CATCH_REQUIRE(tb[1].name == "b");

  CATCH_REQUIRE(tb[0].tensor.equal(t1));
  CATCH_REQUIRE(tb[1].tensor.equal(t2));
}

CATCH_TEST_CASE("TensorBundle iteration works", "[tensorbundle]")
{
  ams::TensorBundle tb;
  tb.add("x", at::full({1}, 42));
  tb.add("y", at::full({1}, 13));

  std::vector<std::string> names;
  for (auto& item : tb) {
    names.push_back(item.name);
  }

  CATCH_REQUIRE(names.size() == 2);
  CATCH_REQUIRE(names[0] == "x");
  CATCH_REQUIRE(names[1] == "y");
}

CATCH_TEST_CASE("TensorBundle copy semantics", "[tensorbundle]")
{
  ams::TensorBundle tb;
  tb.add("z", at::ones({5}));

  ams::TensorBundle tb2 = tb;  // copy

  CATCH_REQUIRE(tb2.size() == 1);
  CATCH_REQUIRE(tb2[0].name == "z");
  CATCH_REQUIRE(tb2[0].tensor.equal(tb[0].tensor));
}

CATCH_TEST_CASE("TensorBundle move semantics", "[tensorbundle]")
{
  ams::TensorBundle tb;
  tb.add("m", at::rand({4}));

  at::Tensor original = tb[0].tensor;

  ams::TensorBundle tb2 = std::move(tb);

  CATCH_REQUIRE(tb2.size() == 1);
  CATCH_REQUIRE(tb2[0].name == "m");
  CATCH_REQUIRE(tb2[0].tensor.equal(original));

  // moved-from tb should be valid but empty
  CATCH_REQUIRE(tb.size() == 0);
  CATCH_REQUIRE(tb.empty());
}

CATCH_TEST_CASE("TensorBundle clear()", "[tensorbundle]")
{
  ams::TensorBundle tb;

  tb.add("a", at::rand({1}));
  tb.add("b", at::rand({1}));

  CATCH_REQUIRE(tb.size() == 2);

  tb.clear();

  CATCH_REQUIRE(tb.size() == 0);
  CATCH_REQUIRE(tb.empty());
}
