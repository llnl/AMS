/*
 * Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
 * AMSLib Project Developers
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#pragma once

#include <fmt/color.h>
#include <fmt/core.h>

#include <cstdio>
#include <mutex>
#include <string>

#include "wf/utils.hpp"

namespace ams
{

/// Verbosity levels for AMS logging.
enum class LogLevel {
  Unknown = -1,
  Error = 0,
  Warning,
  Info,
  Debug,
  NumLevels
};

/// Configuration controlling how log records are annotated.
struct LogConfig {
  bool UseColor = true;      ///< Enable colored log prefixes.
  bool ShowPid = true;       ///< Show process ID.
  bool ShowRank = true;      ///< Show Flux/Slurm/PMI rank.
  bool ShowHostname = true;  ///< Show hostname.
};

/// Central logging facility for AMS.
/// Provides thread-safe, fmt-based output with compile-time filtering.
class Logger
{
public:
  /// Returns the global AMS logger instance.
  static Logger& get();

  /// Sets the runtime logging verbosity.
  void setLevel(LogLevel lvl) noexcept { RuntimeLevel = lvl; }

  /// Returns the current runtime log level.
  LogLevel getLevel() const noexcept { return RuntimeLevel; }

  /// Applies a new log configuration.
  void setConfig(const LogConfig& cfg) noexcept { Config = cfg; }

  /// Redirects logging output to the given file path.
  void setOutputFile(const std::string& path);

  /// Redirects logging to stdout.
  void setOutputStdout() noexcept { Output = stdout; }

  /// Redirects logging to stderr.
  void setOutputStderr() noexcept { Output = stderr; }

  /// Logs a message using fmt formatting.
  template <typename... Args>
  inline void log(LogLevel lvl,
                  const char* Descr,
                  fmt::format_string<Args...> fmtstr,
                  Args&&... args)
  {
    if (!isEnabled(lvl)) return;

    std::lock_guard<std::mutex> lock(Mutex);
    printPrefix(lvl);
    fmt::print(Output,
               fg(fmt::color::light_gray) | fmt::emphasis::bold,
               "[{}] ",
               Descr);
    fmt::print(Output, fmtstr, std::forward<Args>(args)...);
    fmt::print(Output, "\n");

    if (AutoFlush) std::fflush(Output);
  }

  /// Returns true if the runtime level allows logging the given level.
  inline bool isEnabled(LogLevel lvl) const noexcept
  {
    return lvl <= RuntimeLevel;
  }

  /// Flushes the output stream.
  void flush() { std::fflush(Output); }

  /// Returns a string representation of the log level.
  static constexpr const char* toString(LogLevel lvl)
  {
    switch (lvl) {
      case LogLevel::Error:
        return "ERROR";
      case LogLevel::Warning:
        return "WARNING";
      case LogLevel::Info:
        return "INFO";
      case LogLevel::Debug:
        return "DEBUG";
      default:
        return "UNKNOWN";
    }
  }

  static LogLevel fromString(std::optional<std::string> LvlStr)
  {
    if (!LvlStr) return LogLevel::Error;
    auto& Lvl = LvlStr.value();
    std::transform(Lvl.begin(), Lvl.end(), Lvl.begin(), [](unsigned char c) {
      return std::tolower(c);
    });
    if (Lvl == "debug")
      return LogLevel::Debug;
    else if (Lvl == "info")
      return LogLevel::Info;
    else if (Lvl == "warning")
      return LogLevel::Warning;
    else if (Lvl == "error")
      return LogLevel::Error;
    return LogLevel::Unknown;
  }

private:
  Logger();
  ~Logger();

  /// Gathers hostname, PID, rank, etc.
  void initializeIds();

  /// Prints the prefix: `[AMS:<LEVEL>:host:rank:pid] `
  void printPrefix(LogLevel lvl);

private:
  std::mutex Mutex;
  LogLevel RuntimeLevel = LogLevel::Error;
  LogConfig Config;

  FILE* Output = stderr;
  bool AutoFlush = true;

  // Cached identifiers
  std::string Hostname;
  int RankId = 0;
  int Pid = 0;
};

}  // namespace ams
