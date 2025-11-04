/*
 * Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
 * AMSLib Project Developers
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */


#include "logger.hpp"

#include <limits.h>
#include <unistd.h>

#include <algorithm>  // for std::equal
#include <cctype>     // for std::toupper
#include <cstdlib>    // for getenv()
#include <experimental/filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <ostream>
#include <regex>
#include <string>

#include "debug.h"
#include "wf/debug.h"
#include "wf/logger.hpp"

static int get_rank_id()
{
  if (const char* flux_id = std::getenv("FLUX_TASK_RANK")) {
    return std::stoi(flux_id);
  } else if (const char* rid = std::getenv("SLURM_PROCID")) {
    return std::stoi(rid);
  } else if (const char* jsm = std::getenv("JSM_NAMESPACE_RANK")) {
    return std::stoi(jsm);
  } else if (const char* pmi = std::getenv("PMIX_RANK")) {
    return std::stoi(pmi);
  }
  return 0;
}


namespace ams
{
namespace util
{

static bool path_exists(std::string& path)
{
  namespace fs = std::experimental::filesystem;
  fs::path Path(path);
  std::error_code ec;

  if (!fs::exists(Path, ec)) {
    AMS_FATAL(AMS, "Path %s does not exist", path.c_str());
    return false;
  }
  return true;
}


// By default AMS prints only errors
static LogVerbosityLevel defaultLevel = LogVerbosityLevel::Error;

const char* Logger::MessageLevelName[LogVerbosityLevel::Num_Levels] = {"ERROR",
                                                                       "WARNIN"
                                                                       "G",
                                                                       "INFO",
                                                                       "DEBUG"};

static int case_insensitive_match(const std::string s1, const std::string s2)
{
  return (s1.size() == s2.size()) &&
         std::equal(s1.begin(), s1.end(), s2.begin(), [](char c1, char c2) {
           return (std::toupper(c1) == std::toupper(c2));
         });
}

Logger::Logger() noexcept
    :  // by default, all message streams are disabled
      m_is_enabled{false, false, false, false}
{
  LogVerbosityLevel level{defaultLevel};
  setLoggingMsgLevel(level);
}

LogVerbosityLevel getVerbosityLevel(const char* level_str)
{
  if (level_str == nullptr) return defaultLevel;

  for (int i = 0; i < LogVerbosityLevel::Num_Levels; ++i) {
    if (case_insensitive_match(level_str, Logger::MessageLevelName[i])) {
      return static_cast<LogVerbosityLevel>(i);
    }
  }

  return defaultLevel;
}

void Logger::setLoggingMsgLevel(LogVerbosityLevel level)
{
  for (int i = 0; i < LogVerbosityLevel::Num_Levels; ++i)
    m_is_enabled[i] = (i <= level);
}

Logger* Logger::getActiveLogger()
{
  static Logger logger;
  static std::once_flag _amsLogger;
  std::call_once(_amsLogger, [&]() { logger.setup_loggers(); });
  return &logger;
}

static inline std::string concat_file_name(const std::string& path,
                                           const std::string& prefix,
                                           const std::string& suffix)
{
  return path + "/" + prefix + "." + suffix;
}

void Logger::initialize_std_io_err(const bool enable_log,
                                   std::string& log_path,
                                   std::string log_fn)
{
  ams_out = nullptr;
  ams_err = stderr;

  AMS_CFATAL(AMS, !path_exists(log_path), "Log Directory does not exist");

  if (enable_log) {
    ams_out = stdout;
    // The case we want to just redirect to stdout
    if (!log_fn.empty()) {
      const std::string log_filename{concat_file_name(log_path, log_fn, "log")};
      ams_out = fopen(log_filename.c_str(), "a");
      AMS_CFATAL(Logger,
             ams_out == nullptr,
             "Could not open file for stdout redirection");
    }
  }
}


void Logger::setup_loggers()
{
  namespace fs = std::experimental::filesystem;
  const char* ams_logger_level = std::getenv("AMS_LOG_LEVEL");
  const char* ams_logger_dir = std::getenv("AMS_LOG_DIR");
  const char* ams_logger_prefix = std::getenv("AMS_LOG_PREFIX");
  std::string log_fn("");
  std::string log_path("./");

  bool enable_log = false;

  if (ams_logger_level) {
    auto log_lvl = ams::util::getVerbosityLevel(ams_logger_level);
    setLoggingMsgLevel(log_lvl);
    enable_log = true;
  }

  // In the case we specify a directory and we do not specify a file
  // by default we write to a file.
  if (ams_logger_dir && !ams_logger_prefix) {
    ams_logger_prefix = "ams";
  }

  if (ams_logger_prefix) {
    // We are going to redirect stdout to some file
    // By default we store to the current directory
    std::string pattern("");
    std::string log_prefix(ams_logger_prefix);

    if (ams_logger_dir) {
      log_path = std::string(ams_logger_dir);
    }

    char hostname[HOST_NAME_MAX];
    if (gethostname(hostname, HOST_NAME_MAX) != 0) {
      AMS_FATAL(AMS, "Get hostname returns error");
    }

    int id = 0;
    if (log_prefix.find("<RID>") != std::string::npos) {
      pattern = std::string("<RID>");
      id = get_rank_id();
    } else if (log_prefix.find("<PID>") != std::string::npos) {
      pattern = std::string("<PID>");
      id = getpid();
    }

    // Combine hostname and pid
    std::ostringstream combined;
    combined << "." << hostname << "." << id;

    if (!pattern.empty()) {
      log_path = fs::absolute(log_path).string();
      log_fn =
          std::regex_replace(log_prefix, std::regex(pattern), combined.str());
    } else {
      log_path = fs::absolute(log_path).string();
      log_fn = log_prefix + combined.str();
    }
  }
  initialize_std_io_err(enable_log, log_path, log_fn);

  return;
}


void Logger::flush()
{
  if (ams_out != nullptr && ams_out != stdout) fflush(ams_out);
  fflush(ams_err);
}


void Logger::close()
{

  if (ams_out != nullptr && ams_out != stdout) {
    fclose(ams_out);
    ams_out = nullptr;
  }
}

void close()
{
  auto logger = Logger::getActiveLogger();
  logger->flush();
  logger->close();
}

void flush_files()
{
  auto logger = Logger::getActiveLogger();
  logger->flush();
}

}  // namespace util
}  // namespace ams
