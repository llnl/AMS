#include "wf/policy.hpp"

#include <catch2/catch_test_macros.hpp>
#include <memory>
#include <string>
#include <type_traits>

#include "ml/Model.hpp"
#include "wf/action.hpp"
#include "wf/eval_context.hpp"
#include "wf/layout_transform.hpp"
#include "wf/pipeline.hpp"

namespace ams
{

namespace
{

class IncAction final : public Action
{
public:
  const char* name() const noexcept override { return "IncAction"; }
  AMSStatus run(EvalContext& Ctx) override
  {
    Ctx.Threshold = Ctx.Threshold.value_or(0.0f) + 1.0f;
    return {};
  }
};

class FailAction final : public Action
{
public:
  const char* name() const noexcept override { return "FailAction"; }
  AMSStatus run(EvalContext&) override
  {
    return AMS_MAKE_ERROR(AMSErrorType::Generic, "FailAction triggered");
  }
};

class DummyLayout final : public LayoutTransform
{
public:
  const char* name() const noexcept override { return "DummyLayout"; }

  AMSExpected<IndexMap> pack(const TensorBundle&,
                             const TensorBundle&,
                             at::Tensor&) override
  {
    return IndexMap{};
  }
  AMSStatus unpack(const torch::jit::IValue&,
                   TensorBundle&,
                   TensorBundle&,
                   std::optional<at::Tensor>&) override
  {
    return {};
  }
};

class DirectLikePolicy final : public Policy
{
public:
  const char* name() const noexcept override { return "DirectLikePolicy"; }

  Pipeline makePipeline(const ml::InferenceModel* /*Model*/,
                        LayoutTransform& /*Layout*/) const override
  {
    Pipeline P;
    P.add(std::make_unique<IncAction>()).add(std::make_unique<IncAction>());
    return P;
  }
};

class FailingPolicy final : public Policy
{
public:
  const char* name() const noexcept override { return "FailingPolicy"; }

  Pipeline makePipeline(const ml::InferenceModel* /*Model*/,
                        LayoutTransform& /*Layout*/) const override
  {
    Pipeline P;
    P.add(std::make_unique<IncAction>())
        .add(std::make_unique<FailAction>())
        .add(std::make_unique<IncAction>());  // must not run
    return P;
  }
};

}  // namespace

CATCH_TEST_CASE("Policy is an abstract factory for Pipelines", "[wf][policy]")
{
  CATCH_STATIC_REQUIRE(std::is_abstract_v<Policy>);
  CATCH_STATIC_REQUIRE(std::has_virtual_destructor_v<Policy>);

  DummyLayout L;
  ml::InferenceModel* Model = nullptr;

  DirectLikePolicy Pol;
  CATCH_REQUIRE(std::string(Pol.name()) == "DirectLikePolicy");

  EvalContext Ctx{};
  auto P = Pol.makePipeline(Model, L);

  auto St = P.run(Ctx);
  CATCH_REQUIRE(St);
  CATCH_REQUIRE(Ctx.Threshold == 2.0f);
}

CATCH_TEST_CASE("Policy-built pipeline short-circuits on Action failure",
                "[wf][policy]")
{
  DummyLayout L;
  ml::InferenceModel* Model = nullptr;

  FailingPolicy Pol;
  EvalContext Ctx{};

  auto P = Pol.makePipeline(Model, L);
  auto St = P.run(Ctx);

  CATCH_REQUIRE_FALSE(St);
  CATCH_REQUIRE(St.error().getType() == AMSErrorType::Generic);
  CATCH_REQUIRE(Ctx.Threshold == 1.0f);
}

}  // namespace ams
