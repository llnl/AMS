#include <unistd.h>

#include <catch2/catch_test_macros.hpp>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>

#include "AMS.h"
#include "wf/logger.hpp"

namespace fs = std::filesystem;

CATCH_TEST_CASE("AMS_OBJECTS missing model follows the build's Torch policy",
                "[ams][environment][model]")
{
  const auto root = fs::temp_directory_path() /
                    ("ams_missing_model_" + std::to_string(getpid()));
  fs::create_directories(root);
  const auto model_path = root / "missing_model.pt";
  const auto config_path = root / "objects.json";
  const auto log_path = root / "ams.log";

  const nlohmann::json config = {{"domain_models",
                                  {{"test_domain", "test_model"}}},
                                 {"ml_models",
                                  {{"test_model",
                                    {{"model_path", model_path.string()},
                                     {"threshold", 0.5},
                                     {"store", false}}}}},
                                 {"db", {{"dbType", "none"}}}};
  {
    std::ofstream output(config_path);
    CATCH_REQUIRE(output.is_open());
    output << config;
  }

  CATCH_REQUIRE(unsetenv("AMS_LOG_LEVEL") == 0);
  CATCH_REQUIRE(setenv("AMS_OBJECTS", config_path.c_str(), 1) == 0);
  auto& logger = ams::Logger::get();
  CATCH_REQUIRE(logger.getLevel() == ams::LogLevel::Warning);
  logger.setOutputFile(log_path.string());

#ifdef __AMS_ENABLE_TORCH__
  bool caught = false;
  try {
    ams::AMSInit();
  } catch (const std::runtime_error& error) {
    caught = true;
    CATCH_CHECK(std::string(error.what()).find(model_path.string()) !=
                std::string::npos);
  }
  CATCH_REQUIRE(caught);
#else
  CATCH_REQUIRE_NOTHROW(ams::AMSInit());
  const auto model = ams::AMSQueryModel("test_domain");
  CATCH_REQUIRE(model >= 0);
  const auto executor = ams::AMSCreateExecutor(model, 0, 1);
  logger.flush();

  std::ifstream log(log_path);
  const std::string messages{std::istreambuf_iterator<char>(log),
                             std::istreambuf_iterator<char>()};
  CATCH_CHECK(messages.find("WARNING") != std::string::npos);
  CATCH_CHECK(messages.find(model_path.string()) != std::string::npos);

  ams::SmallVector<ams::AMSTensor> inputs;
  ams::SmallVector<ams::AMSTensor> inouts;
  ams::SmallVector<ams::AMSTensor> outputs;
  int physics_calls = 0;
  ams::DomainLambda physics = [&](const ams::SmallVector<ams::AMSTensor>&,
                                  ams::SmallVector<ams::AMSTensor>&,
                                  ams::SmallVector<ams::AMSTensor>&) {
    ++physics_calls;
  };
  ams::AMSExecute(executor, physics, inputs, inouts, outputs);
  CATCH_CHECK(physics_calls == 1);
  ams::AMSDestroyExecutor(executor);
#endif

  CATCH_REQUIRE(unsetenv("AMS_OBJECTS") == 0);
  fs::remove_all(root);
}
