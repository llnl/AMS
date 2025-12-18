#include "wf/action.hpp"

#include <catch2/catch_test_macros.hpp>
#include <memory>
#include <type_traits>

// Prefer the real EvalContext if available.
// If your project uses a different header name, adjust accordingly.
#include "wf/eval_context.hpp"

namespace ams
{

namespace
{
class TestAction final : public Action
{
public:
  const char* name() const noexcept override { return "TestAction"; }

  AMSStatus run(EvalContext& ctx) override
  {
    ctx.Threshold = ctx.Threshold.value_or(0.0f) + 1.0f;
    return {};
  }
};
}  // namespace

CATCH_TEST_CASE("Action: abstract base class + virtual interface",
                "[wf][action]")
{
  CATCH_STATIC_REQUIRE(std::is_abstract_v<Action>);
  CATCH_STATIC_REQUIRE(std::has_virtual_destructor_v<Action>);

  EvalContext ctx{};
  ctx.Threshold = 0.0f;
  std::unique_ptr<Action> act = std::make_unique<TestAction>();

  CATCH_REQUIRE(act->name() == std::string("TestAction"));

  auto Err = act->run(ctx);
  CATCH_REQUIRE(Err);
  CATCH_REQUIRE(ctx.Threshold == 1.0f);

  auto Err1 = act->run(ctx);
  CATCH_REQUIRE(Err1);
  CATCH_REQUIRE(ctx.Threshold == 2.0f);
}

}  // namespace ams
