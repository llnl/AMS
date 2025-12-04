/*
 * Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
 * AMSLib Project Developers
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "logger.hpp"

#include <unistd.h>

#include <cstdlib>
#include <stdexcept>

namespace ams
{

/// Detects the rank from common HPC environments.
static int detectRank()
{
  if (const char* flux = std::getenv("FLUX_TASK_RANK")) return std::stoi(flux);
  if (const char* s = std::getenv("SLURM_PROCID")) return std::stoi(s);
  if (const char* p = std::getenv("PMIX_RANK")) return std::stoi(p);
  if (const char* j = std::getenv("JSM_NAMESPACE_RANK")) return std::stoi(j);
  return 0;
}

Logger& Logger::get()
{
  static Logger Inst;
  return Inst;
}

Logger::Logger()
{
  initializeIds();
  auto Lvl = std::getenv("AMS_LOG_LEVEL");
  RuntimeLevel =
      Logger::fromString(Lvl ? std::optional<std::string>(Lvl) : std::nullopt);
}

Logger::~Logger()
{
  if (Output && Output != stdout && Output != stderr) {
    std::fclose(Output);
  }
}

void Logger::initializeIds()
{
  char buf[256];
  if (gethostname(buf, sizeof(buf)) == 0)
    Hostname = buf;
  else
    Hostname = "unknown-host";

  RankId = detectRank();
  Pid = static_cast<int>(getpid());
}

void Logger::setOutputFile(const std::string& path)
{
  FILE* f = std::fopen(path.c_str(), "a");
  if (!f) throw std::runtime_error("Unable to open log file: " + path);
  Output = f;
}

inline fmt::text_style style_for(bool UseColor, LogLevel Lvl)
{
  if (!UseColor) return {};  // default (no color)
  switch (Lvl) {
    case LogLevel::Debug:
      return fg(fmt::color::cyan);
    case LogLevel::Info:
      return fg(fmt::color::green);
    case LogLevel::Warning:
      return fg(fmt::color::yellow);
    case LogLevel::Error:
      return fg(fmt::color::red);
  }
  return {};  // default (no color)
}

void Logger::printPrefix(LogLevel Lvl)
{
  using fmt::color;
  using fmt::fg;
  auto Color = style_for(Config.UseColor, Lvl);
  fmt::print(Output, Color, "[AMS:{}:", toString(Lvl));

  if (Config.ShowHostname) fmt::print(Output, Color, "{}", Hostname);

  if (Config.ShowRank) fmt::print(Output, Color, ":{}", RankId);

  if (Config.ShowPid) fmt::print(Output, Color, ":{}", Pid);

  fmt::print(Output, Color, "]");
}

}  // namespace ams
