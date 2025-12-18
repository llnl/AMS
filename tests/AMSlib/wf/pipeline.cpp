#include "wf/pipeline.hpp"

#include <catch2/catch_test_macros.hpp>
#include <memory>
#include <string>

#include "wf/eval_context.hpp"

namespace ams
{

namespace
{

class IncAction final : public Action
{
public:
  const char* name() const noexcept override { return "IncAction"; }

  AMSStatus run(EvalContext& ctx) override
  {
    ctx.Threshold = ctx.Threshold.value_or(0.0f) + 1.0f;
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

}  // namespace

CATCH_TEST_CASE("Pipeline runs actions in order and short-circuits on error",
                "[wf][pipeline]")
{
  EvalContext Ctx{};
  Pipeline P;

  // Two increments -> Threshold becomes 2, then FailAction stops the pipeline.
  P.add(std::make_unique<IncAction>())
      .add(std::make_unique<IncAction>())
      .add(std::make_unique<FailAction>())
      .add(std::make_unique<IncAction>());  // must NOT execute

  Ctx.Threshold = 0.0f;

  auto St = P.run(Ctx);
  CATCH_REQUIRE_FALSE(St);
  CATCH_REQUIRE(St.error().getType() == AMSErrorType::Generic);

  // Only the first two IncAction should have run.
  CATCH_REQUIRE(Ctx.Threshold.value() == 2.0f);
}

CATCH_TEST_CASE("Pipeline succeeds when all actions succeed", "[wf][pipeline]")
{
  EvalContext Ctx{};
  Pipeline P;

  P.add(std::make_unique<IncAction>()).add(std::make_unique<IncAction>());

  Ctx.Threshold = 0.0f;
  auto St = P.run(Ctx);
  CATCH_REQUIRE(St);
  CATCH_REQUIRE(Ctx.Threshold.value() == 2.0f);
}

}  // namespace ams
