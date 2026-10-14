/*
 * Copyright 2021-2026 Lawrence Livermore National Security, LLC and other
 * AMSLib Project Developers
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#include "AMS.h"
#include "problems.hpp"

namespace
{
namespace fs = std::filesystem;

constexpr int kNumInputs = 3;
constexpr int kNumInouts = 3;
constexpr int kNumOutputs = 2;

struct Options {
  std::string backend;
  fs::path db_path;
  int num_elements = 0;
};

void printUsage(const char* executable)
{
  std::cerr << "Usage: " << executable
            << " --backend hdf5|json|rmq --db-path PATH --num-elements N\n";
}

Options parseOptions(int argc, char** argv)
{
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument == "--backend" && i + 1 < argc) {
      options.backend = argv[++i];
    } else if (argument == "--db-path" && i + 1 < argc) {
      options.db_path = argv[++i];
    } else if (argument == "--num-elements" && i + 1 < argc) {
      const std::string value = argv[++i];
      size_t parsed = 0;
      const unsigned long count = std::stoul(value, &parsed);
      if (parsed != value.size() || count == 0 ||
          count > static_cast<unsigned long>(std::numeric_limits<int>::max()))
        throw std::invalid_argument(
            "--num-elements must be a positive integer");
      options.num_elements = static_cast<int>(count);
    } else {
      throw std::invalid_argument("Unknown or incomplete argument: " +
                                  argument);
    }
  }

  if (options.backend != "hdf5" && options.backend != "json" &&
      options.backend != "rmq")
    throw std::invalid_argument("--backend must be one of: hdf5, json, rmq");
  if (options.num_elements == 0)
    throw std::invalid_argument("--num-elements is required");
  if (options.backend != "rmq" && options.db_path.empty())
    throw std::invalid_argument("--db-path is required for file backends");
  if (options.backend == "rmq" && std::getenv("AMS_OBJECTS") == nullptr)
    throw std::invalid_argument(
        "The rmq backend requires connection details through AMS_OBJECTS");

  return options;
}

template <typename ProblemType>
void runDomain(const std::string& domain,
               ProblemType& problem,
               int num_elements)
{
  const ams::AMSCAbstrModel model =
      ams::AMSRegisterAbstractModel(domain.c_str(), 0.5, "", true);
  if (model < 0) throw std::runtime_error("Failed to register " + domain);

  ams::AMSExecutor executor = ams::AMSCreateExecutor(model, 0, 1);
  if (executor < 0) throw std::runtime_error("Failed to create " + domain);

  problem.ams_run(executor, ams::AMSResourceType::AMS_HOST, 1, num_elements);
  ams::AMSDestroyExecutor(executor);
}

template <typename DType>
void runWorkloads(const std::string& type_name, int num_elements)
{
  Problem<DType> ordinary(kNumInputs, kNumOutputs);
  runDomain("no_torch_" + type_name + "_ordinary", ordinary, num_elements);

  Problem2D<DType> inout(kNumInputs - 1, kNumInouts, kNumOutputs);
  runDomain("no_torch_" + type_name + "_inout", inout, num_elements);

  ProblemBroadcast<DType> broadcast(kNumInputs, kNumOutputs);
  runDomain("no_torch_" + type_name + "_broadcast", broadcast, num_elements);
}

void removeExistingHdf5Files(const fs::path& db_path)
{
  for (const char* type_name : {"float", "double"}) {
    for (const char* workload : {"ordinary", "inout", "broadcast"}) {
      fs::remove(db_path / (std::string("no_torch_") + type_name + "_" +
                            workload + "_0.h5"));
    }
  }
}

void configureDatabase(const Options& options)
{
  if (options.backend == "rmq") return;

  fs::create_directories(options.db_path);
  if (!fs::is_directory(options.db_path))
    throw std::runtime_error("Database path is not a directory: " +
                             options.db_path.string());

  if (options.backend == "hdf5") {
    removeExistingHdf5Files(options.db_path);
    ams::AMSConfigureFSDatabase(ams::AMSDBType::AMS_HDF5,
                                options.db_path.c_str());
  } else {
    if (std::getenv("AMS_JSON_MODE") == nullptr)
      setenv("AMS_JSON_MODE", "json", 0);
    ams::AMSConfigureFSDatabase(ams::AMSDBType::AMS_JSON,
                                options.db_path.c_str());
  }
}
}  // namespace

int main(int argc, char** argv)
{
  bool initialized = false;
  try {
    const Options options = parseOptions(argc, argv);
    ams::AMSInit();
    initialized = true;
    configureDatabase(options);
    runWorkloads<float>("float", options.num_elements);
    runWorkloads<double>("double", options.num_elements);
    ams::AMSFinalize();
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "ams_ete_no_torch: " << error.what() << '\n';
    printUsage(argv[0]);
    if (initialized) ams::AMSFinalize();
    return EXIT_FAILURE;
  }
}
