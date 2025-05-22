/*
 * Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
 * AMSLib Project Developers
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef __TEST_UTILS__
#define __TEST_UTILS__

#include <execinfo.h>
#include <unistd.h>

#include <csignal>

#include "AMS.h"

#ifdef __AMS_ENABLE_MPI__
#include <mpi.h>
#define MPI_CALL(stmt)                                                         \
  if (stmt != MPI_SUCCESS) {                                                   \
    fprintf(stderr, "Error in MPI-Call (File: %s, %d)\n", __FILE__, __LINE__); \
  }
#else
typedef void* MPI_Comm;
#define MPI_CALL(stm)
#endif

using namespace ams;

static inline AMSDType getDataType(const char* d_type)
{
  AMSDType dType = AMSDType::AMS_DOUBLE;
  if (std::strcmp(d_type, "float") == 0) {
    dType = AMSDType::AMS_SINGLE;
  } else if (std::strcmp(d_type, "double") == 0) {
    dType = AMSDType::AMS_DOUBLE;
  } else {
    assert(false && "Unknown data type (must be 'float' or 'double')");
  }
  return dType;
}

static inline AMSDType getDataType(std::string& d_type)
{
  AMSDType dType = AMSDType::AMS_DOUBLE;
  if (d_type == "float") {
    dType = AMSDType::AMS_SINGLE;
  } else if (d_type == "double") {
    dType = AMSDType::AMS_DOUBLE;
  } else {
    assert(false && "Unknown data type (must be 'float' or 'double')");
  }
  return dType;
}


static inline AMSDBType getDBType(const char* db_type)
{
  AMSDBType dbType = AMSDBType::AMS_NONE;
  if (std::strcmp(db_type, "hdf5") == 0) {
    dbType = AMSDBType::AMS_HDF5;
  } else if (std::strcmp(db_type, "rmq") == 0) {
    dbType = AMSDBType::AMS_RMQ;
  }
  return dbType;
}

static inline AMSDBType getDBType(std::string db_type)
{
  AMSDBType dbType = AMSDBType::AMS_NONE;
  if (db_type == "hdf5") {
    dbType = AMSDBType::AMS_HDF5;
  } else if (db_type == "rmq") {
    dbType = AMSDBType::AMS_RMQ;
  }
  return dbType;
}

// Signal handler to print the stack trace
static inline void signalHandler(int signum)
{
  const char* msg = "[signalHandler] Caught signal\n";
  write(STDERR_FILENO, msg, strlen(msg));

  // Obtain the backtrace
  const int maxFrames = 128;
  void* addrlist[maxFrames];

  // Get void*'s for all entries on the stack
  int addrlen = backtrace(addrlist, maxFrames);

  if (addrlen == 0) {
    const char* no_stack = "No stack trace available\n";
    write(STDERR_FILENO, no_stack, strlen(no_stack));
    _exit(1);  // _exit() Cannot be trap, interrupted
  }

  // Print out all the frames to stderr
  backtrace_symbols_fd(addrlist, addrlen, STDERR_FILENO);
  _exit(1);
}


static inline void installSignals()
{
  std::signal(SIGSEGV, signalHandler);  // segmentation fault
  std::signal(SIGABRT, signalHandler);  // abort()
  std::signal(SIGFPE, signalHandler);   // floating-point exception
  std::signal(SIGILL, signalHandler);   // illegal instruction
  std::signal(SIGINT, signalHandler);   // interrupt (e.g., Ctrl+C)
  std::signal(SIGTERM, signalHandler);  // termination request
  std::signal(SIGPIPE, signalHandler);  // broken pipe
}

class TestArgs
{
public:
  void PrintOptions() const
  {
    std::cout << "Available options:\n";
    for (const auto& opt : registered_) {
      std::cout << "  ";
      for (size_t i = 0; i < opt.keys.size(); ++i) {
        std::cout << opt.keys[i];
        if (i + 1 < opt.keys.size()) std::cout << ", ";
      }
      std::cout << "\n    " << opt.help << "\n";
    }
  }

  void Parse(int argc, char** argv)
  {
    for (int i = 1; i < argc; ++i) {
      std::string arg = argv[i];
      if (arg[0] != '-') continue;

      std::string key = arg;
      std::string value;

      // If the next item isn't an option, treat it as a value
      if (i + 1 < argc && argv[i + 1][0] != '-') {
        value = argv[++i];
      } else {
        value = "true";  // Boolean flag
      }

      options_[key] = value;
    }

    // Set parsed values into variables
    for (auto& opt : registered_) {
      const auto& keys = opt.keys;
      for (const auto& key : keys) {
        if (options_.count(key)) {
          opt.setter(options_[key]);
          opt.wasset = true;
          break;
        }
      }
    }
  }

  template <typename T>
  void AddOption(T* out,
                 std::string short_opt,
                 std::string long_opt,
                 std::string help,
                 bool required = true)
  {
    registered_.push_back(
        {{short_opt, long_opt},
         [out](const std::string& val) { parseValue(val, out); },
         [out]() { return toString(*out); },
         help,
         required,
         false});
  }

  bool Good() const
  {
    for (const auto& opt : registered_) {
      if (opt.required && !opt.wasset) {
        return false;
      }
    }
    return true;
  }

  void PrintUsage() const
  {
    std::cout << "Parsed arguments:\n";
    for (const auto& opt : registered_) {
      std::cout << "  ";
      for (size_t i = 0; i < opt.keys.size(); ++i) {
        std::cout << opt.keys[i];
        if (i + 1 < opt.keys.size()) std::cout << ", ";
      }
      std::cout << " = " << opt.getter() << "\n";
    }
  }

private:
  struct RegisteredOption {
    std::vector<std::string> keys;
    std::function<void(const std::string&)> setter;
    std::function<std::string()> getter;
    std::string help;
    bool required;
    bool wasset;
  };

  std::vector<RegisteredOption> registered_;
  std::unordered_map<std::string, std::string> options_;

  // Parser helper
  template <typename T>
  static void parseValue(const std::string& s, T* out);

  static void parseValue(const std::string& s, std::string* out) { *out = s; }

  static void parseValue(const std::string& s, int* out)
  {
    *out = std::stoi(s);
  }

  static void parseValue(const std::string& s, bool* out)
  {
    *out = (s == "true" || s == "1");
  }

  static void parseValue(const std::string& s, double* out)
  {
    *out = std::stod(s);
  }


  static std::string toString(const std::string& val) { return val; }
  static std::string toString(bool val) { return val ? "true" : "false"; }
  static std::string toString(int val) { return std::to_string(val); }
  static std::string toString(double val) { return std::to_string(val); }
};

#endif
