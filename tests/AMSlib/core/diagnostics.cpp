#include <catch2/catch_test_macros.hpp>
#include <stdexcept>
#include <string>

#include "wf/debug.h"
#include "wf/logger.hpp"

CATCH_TEST_CASE("THROW logs and raises the requested exception",
                "[diagnostics][throw]")
{
  auto& logger = ams::Logger::get();
  logger.setLevel(ams::LogLevel::Error);

  bool caught = false;
  try {
    THROW(std::invalid_argument, "diagnostic THROW error")
  } catch (const std::invalid_argument& e) {
    caught = true;
    const std::string message = e.what();
    CATCH_CHECK(message.find("diagnostic THROW error") != std::string::npos);
    CATCH_CHECK(message.find("diagnostics.cpp:") != std::string::npos);
  }
  CATCH_CHECK(caught);
}

CATCH_TEST_CASE("AMS warnings obey the runtime level", "[diagnostics][warning]")
{
  auto& logger = ams::Logger::get();
  const auto original_level = logger.getLevel();
  logger.setLevel(ams::LogLevel::Warning);

  int evaluated = 0;
  AMS_CWARNING(Diagnostics,
               false,
               "diagnostic hidden conditional warning {}",
               ++evaluated)
  AMS_WARNING(Diagnostics, "diagnostic visible warning {}", ++evaluated)
  CATCH_CHECK(evaluated == 1);

  logger.setLevel(ams::LogLevel::Error);
  AMS_WARNING(Diagnostics, "diagnostic suppressed warning")
  logger.setLevel(original_level);
}
