#include <hdf5.h>
#include <torch/torch.h>
#include <unistd.h>

#include <atomic>
#include <catch2/catch_all.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <catch2/interfaces/catch_interfaces_reporter.hpp>
#include <catch2/reporters/catch_reporter_event_listener.hpp>
#include <catch2/reporters/catch_reporter_registrars.hpp>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <optional>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "AMS.h"
#include "models/linear_models.hpp"
#include "problems.hpp"

using namespace ams;
using namespace std;


template <torch::ScalarType TorchType>
static std::optional<torch::Tensor> readDatasetContents(
    const std::string& fileName,
    const std::string& datasetName,
    hid_t DataType)
{
  hid_t file_id = H5Fopen(fileName.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file_id < 0)
    throw std::runtime_error("Failed to open HDF5 file: " + fileName);

  hid_t dset_id = H5Dopen2(file_id, datasetName.c_str(), H5P_DEFAULT);
  if (dset_id < 0) {
    H5Fclose(file_id);
    return std::nullopt;
  }

  hid_t space_id = H5Dget_space(dset_id);
  if (space_id < 0) {
    H5Dclose(dset_id);
    H5Fclose(file_id);
    return std::nullopt;
  }

  int ndims = H5Sget_simple_extent_ndims(space_id);
  if (ndims < 0) {
    H5Sclose(space_id);
    H5Dclose(dset_id);
    H5Fclose(file_id);
    throw std::runtime_error("Bad ndims");
  }

  std::vector<hsize_t> dims(ndims);
  if (H5Sget_simple_extent_dims(space_id, dims.data(), nullptr) < 0) {
    H5Sclose(space_id);
    H5Dclose(dset_id);
    H5Fclose(file_id);
    return std::nullopt;
  }

  H5Sclose(space_id);

  std::vector<int64_t> shape(dims.begin(), dims.end());

  // Read as float32 on CPU
  torch::Tensor readTensor = torch::empty(shape, TorchType);
  herr_t st = H5Dread(
      dset_id, DataType, H5S_ALL, H5S_ALL, H5P_DEFAULT, readTensor.data_ptr());
  H5Dclose(dset_id);
  H5Fclose(file_id);
  return readTensor;
}


void verify_mean_data(const torch::Tensor& inputs,
                      const torch::Tensor& outputs,
                      double threshold,
                      int64_t num_inputs,
                      int64_t num_outputs)
{
  torch::Tensor verify_inputs = torch::zeros_like(inputs);

  int64_t step = (threshold == 0.5) ? 2 : 1;
  if (threshold == 0.5) {
    verify_inputs[0] = torch::ones({num_inputs}, inputs.options());
  }

  // Fill verify_inputs row-by-row (along first dimension)
  const int64_t outer_dim = inputs.size(0);
  for (int64_t i = 1; i < outer_dim; ++i)
    verify_inputs[i] = verify_inputs[i - 1] + step;
  // Compare inputs
  double diff_sum_inputs = (verify_inputs - inputs).abs().sum().item<double>();
  CATCH_REQUIRE(diff_sum_inputs <= 1e-6);

  // Sum over all dimensions except the first (axis=1 in NumPy)
  std::vector<int64_t> reduce_dims;
  for (int64_t d = 1; d < inputs.dim(); ++d)
    reduce_dims.push_back(d);

  torch::Tensor verify_output = inputs.sum(reduce_dims) * num_outputs;
  torch::Tensor outputs_reduced = outputs.sum(reduce_dims);


  double diff_sum_outputs =
      (verify_output - outputs_reduced).abs().sum().item<double>();
  CATCH_REQUIRE(diff_sum_outputs <= 1e-6);
}


static std::atomic<int> g_test_counter{0};

static std::string next_test_name()
{
  int id = g_test_counter.fetch_add(1, std::memory_order_relaxed);
  return "test_" + std::to_string(id);
}

namespace Catch
{
template <>
struct StringMaker<AMSDType> {
  static std::string convert(AMSDType t)
  {
    switch (t) {
      case AMSDType::AMS_SINGLE:
        return "single";
      case AMSDType::AMS_DOUBLE:
        return "double";
      default:
        return "unknown";
    }
  }
};
template <>
struct StringMaker<AMSResourceType> {
  static std::string convert(AMSResourceType r)
  {
    switch (r) {
      case AMSResourceType::AMS_HOST:
        return "host";
      case AMSResourceType::AMS_DEVICE:
        return "device";
      default:
        return "unknown";
    }
  }
};
}  // namespace Catch


CATCH_TEST_CASE("Evaluate AMS explicit Interface 1D in/out")
{
  // ----- Generators -----
  auto model_desc = GENERATE_COPY(from_range(simple_models));
  auto threshold = GENERATE(Catch::Generators::values({0.0, 0.5, 1.0}));

  auto phDTypes = GENERATE(
      Catch::Generators::values({AMSDType::AMS_SINGLE, AMSDType::AMS_DOUBLE}));
  auto resource =
      GENERATE(Catch::Generators::values({AMSResourceType::AMS_HOST}));

  constexpr int numElements = 4 * 1024;
  constexpr int numIterations = 1;

  CATCH_DYNAMIC_SECTION("model=" << model_desc << " | dtype=" << phDTypes
                                 << " | resource=" << resource
                                 << " | thr=" << threshold)
  {
    auto test_name = next_test_name();
    AMSCAbstrModel ams_model_descr =
        AMSRegisterAbstractModel(test_name.c_str(),
                                 threshold,
                                 model_desc.ModelPath.c_str(),
                                 test_name.c_str());

    CATCH_REQUIRE(ams_model_descr >= 0);

    AMSExecutor wf = AMSCreateExecutor(ams_model_descr, 0, 1);
    CATCH_REQUIRE(wf >= 0);

    if (phDTypes == AMSDType::AMS_SINGLE) {
      Problem<float> prob(model_desc.numIn, model_desc.numOut);
      prob.ams_run(wf, resource, numIterations, numElements);
    } else {
      Problem<double> prob(model_desc.numIn, model_desc.numOut);
      prob.ams_run(wf, resource, numIterations, numElements);
    }
    auto db_name = AMSGetDatabaseName(wf);
    AMSDestroyExecutor(wf);
    CATCH_DYNAMIC_SECTION("Processing file " << db_name)
    {
      auto fraction = 1.0 - threshold;
      auto DomainNameTensor =
          readDatasetContents<torch::kUInt8>(db_name,
                                             "domain_name",
                                             H5T_NATIVE_CHAR);
      CATCH_REQUIRE(DomainNameTensor.has_value());
      auto data_ptr = DomainNameTensor.value().data_ptr<uint8_t>();
      std::string name{reinterpret_cast<const char*>(data_ptr),
                       static_cast<std::string::size_type>(
                           DomainNameTensor.value().numel())};

      CATCH_REQUIRE(name == test_name);
      // WHen fraction is 0.0 we are not storing anything ...
      if (fraction != 0.0) {
        auto InData = readDatasetContents<torch::kFloat32>(db_name,
                                                           "input_data",
                                                           H5T_NATIVE_FLOAT);
        CATCH_REQUIRE(InData.has_value());

        auto OutData = readDatasetContents<torch::kFloat32>(db_name,
                                                            "output_data",
                                                            H5T_NATIVE_FLOAT);
        CATCH_REQUIRE(OutData.has_value());

        if (fraction == 1.0) {
          CATCH_REQUIRE(InData.value().sizes()[0] == fraction * numElements);
          CATCH_REQUIRE(OutData.value().sizes()[0] == fraction * numElements);
        } else if (fraction == 0.5) {
          CATCH_REQUIRE(InData.value().sizes()[0] ==
                        OutData.value().sizes()[0]);
          CATCH_REQUIRE_THAT(OutData.value().sizes()[0],
                             Catch::Matchers::WithinAbs(fraction * numElements,
                                                        0.05 * numElements));
          CATCH_REQUIRE_THAT(InData.value().sizes()[0],
                             Catch::Matchers::WithinAbs(fraction * numElements,
                                                        0.05 * numElements));
        }
        if (model_desc.UQType.find("duq") != std::string::npos &&
            fraction == 0.5) {
          verify_mean_data(InData.value(),
                           OutData.value(),
                           threshold,
                           model_desc.numIn,
                           model_desc.numOut);
        }
      }
    }
  }
}


CATCH_TEST_CASE("Evaluate AMS explicit Interface 2D in/inout/in")
{
  // ----- Generators -----
  auto model_desc = GENERATE_COPY(from_range(simple_models));
  auto threshold = GENERATE(Catch::Generators::values({0.0, 0.5, 1.0}));

  auto phDTypes = GENERATE(
      Catch::Generators::values({AMSDType::AMS_SINGLE, AMSDType::AMS_DOUBLE}));
  auto resource =
      GENERATE(Catch::Generators::values({AMSResourceType::AMS_HOST}));

  auto num_inouts = GENERATE(Catch::Generators::values({6}));


  constexpr int numElements = 4 * 1024;
  constexpr int numIterations = 1;

  CATCH_DYNAMIC_SECTION("model=" << model_desc << " | dtype=" << phDTypes
                                 << " | resource=" << resource << " | thr="
                                 << threshold << " | num_inouts=" << num_inouts)
  {
    auto test_name = next_test_name();
    AMSCAbstrModel ams_model_descr =
        AMSRegisterAbstractModel(test_name.c_str(),
                                 threshold,
                                 model_desc.ModelPath.c_str(),
                                 test_name.c_str());

    CATCH_REQUIRE(ams_model_descr >= 0);

    AMSExecutor wf = AMSCreateExecutor(ams_model_descr, 0, 1);
    CATCH_REQUIRE(wf >= 0);

    if (phDTypes == AMSDType::AMS_SINGLE) {
      Problem2D<float> prob(model_desc.numIn - num_inouts,
                            num_inouts,
                            model_desc.numOut - num_inouts);
      prob.ams_run(wf, resource, numIterations, numElements);
    } else {
      Problem2D<double> prob(model_desc.numIn - num_inouts,
                             num_inouts,
                             model_desc.numOut - num_inouts);
      prob.ams_run(wf, resource, numIterations, numElements);
    }
    auto db_name = AMSGetDatabaseName(wf);
    AMSDestroyExecutor(wf);
    CATCH_DYNAMIC_SECTION("Processing file " << db_name)
    {
      auto fraction = 1.0 - threshold;
      auto DomainNameTensor =
          readDatasetContents<torch::kUInt8>(db_name,
                                             "domain_name",
                                             H5T_NATIVE_CHAR);
      CATCH_REQUIRE(DomainNameTensor.has_value());
      auto data_ptr = DomainNameTensor.value().data_ptr<uint8_t>();
      std::string name{reinterpret_cast<const char*>(data_ptr),
                       static_cast<std::string::size_type>(
                           DomainNameTensor.value().numel())};

      CATCH_REQUIRE(name == test_name);
      // WHen fraction is 0.0 we are not storing anything ...
      if (fraction != 0.0) {
        auto InData = readDatasetContents<torch::kFloat32>(db_name,
                                                           "input_data",
                                                           H5T_NATIVE_FLOAT);
        CATCH_REQUIRE(InData.has_value());

        auto OutData = readDatasetContents<torch::kFloat32>(db_name,
                                                            "output_data",
                                                            H5T_NATIVE_FLOAT);
        CATCH_REQUIRE(OutData.has_value());

        if (fraction == 1.0) {
          CATCH_REQUIRE(InData.value().sizes()[0] == fraction * numElements);
          CATCH_REQUIRE(OutData.value().sizes()[0] == fraction * numElements);
        } else if (fraction == 0.5) {
          CATCH_REQUIRE(InData.value().sizes()[0] ==
                        OutData.value().sizes()[0]);
          CATCH_REQUIRE_THAT(OutData.value().sizes()[0],
                             Catch::Matchers::WithinAbs(fraction * numElements,
                                                        0.05 * numElements));
          CATCH_REQUIRE_THAT(InData.value().sizes()[0],
                             Catch::Matchers::WithinAbs(fraction * numElements,
                                                        0.05 * numElements));
        }
        if (model_desc.UQType.find("duq") != std::string::npos &&
            fraction == 0.5) {
          verify_mean_data(InData.value(),
                           OutData.value(),
                           threshold,
                           model_desc.numIn,
                           model_desc.numOut);
        }
      }
    }
  }
}

CATCH_TEST_CASE("Evaluate AMS explicit Interface Broadcast in/out")
{
  // ----- Generators -----
  auto model_desc = GENERATE_COPY(from_range(simple_models));
  auto threshold = GENERATE(Catch::Generators::values({0.0, 0.5, 1.0}));

  auto phDTypes = GENERATE(
      Catch::Generators::values({AMSDType::AMS_SINGLE, AMSDType::AMS_DOUBLE}));
  auto resource =
      GENERATE(Catch::Generators::values({AMSResourceType::AMS_HOST}));

  constexpr int numElements = 4 * 1024;
  constexpr int numIterations = 1;

  if (model_desc.ModelPath.find("linear") != std::string::npos) {
    CATCH_SKIP("Skipping cause verification applicable only to random");
  }


  CATCH_DYNAMIC_SECTION("model=" << model_desc << " | dtype=" << phDTypes
                                 << " | resource=" << resource
                                 << " | thr=" << threshold)
  {
    auto test_name = next_test_name();
    AMSCAbstrModel ams_model_descr =
        AMSRegisterAbstractModel(test_name.c_str(),
                                 threshold,
                                 model_desc.ModelPath.c_str(),
                                 test_name.c_str());

    CATCH_REQUIRE(ams_model_descr >= 0);

    AMSExecutor wf = AMSCreateExecutor(ams_model_descr, 0, 1);
    CATCH_REQUIRE(wf >= 0);

    if (phDTypes == AMSDType::AMS_SINGLE) {
      ProblemBroadcast<float> prob(model_desc.numIn, model_desc.numOut);
      prob.ams_run(wf, resource, numIterations, numElements);
    } else {
      ProblemBroadcast<double> prob(model_desc.numIn, model_desc.numOut);
      prob.ams_run(wf, resource, numIterations, numElements);
    }
    auto db_name = AMSGetDatabaseName(wf);
    AMSDestroyExecutor(wf);
    CATCH_DYNAMIC_SECTION("Processing file " << db_name)
    {
      auto fraction = 1.0 - threshold;
      auto DomainNameTensor =
          readDatasetContents<torch::kUInt8>(db_name,
                                             "domain_name",
                                             H5T_NATIVE_CHAR);
      CATCH_REQUIRE(DomainNameTensor.has_value());
      auto data_ptr = DomainNameTensor.value().data_ptr<uint8_t>();
      std::string name{reinterpret_cast<const char*>(data_ptr),
                       static_cast<std::string::size_type>(
                           DomainNameTensor.value().numel())};

      CATCH_REQUIRE(name == test_name);
      // WHen fraction is 0.0 we are not storing anything ...
      if (fraction != 0.0) {
        auto InData = readDatasetContents<torch::kFloat32>(db_name,
                                                           "input_data",
                                                           H5T_NATIVE_FLOAT);
        CATCH_REQUIRE(InData.has_value());

        auto OutData = readDatasetContents<torch::kFloat32>(db_name,
                                                            "output_data",
                                                            H5T_NATIVE_FLOAT);
        CATCH_REQUIRE(OutData.has_value());

        if (fraction == 1.0) {
          CATCH_REQUIRE(InData.value().sizes()[0] == fraction * numElements);
          CATCH_REQUIRE(OutData.value().sizes()[0] == fraction * numElements);
        } else if (fraction == 0.5) {
          CATCH_REQUIRE(InData.value().sizes()[0] ==
                        OutData.value().sizes()[0]);
          CATCH_REQUIRE_THAT(OutData.value().sizes()[0],
                             Catch::Matchers::WithinAbs(fraction * numElements,
                                                        0.05 * numElements));
          CATCH_REQUIRE_THAT(InData.value().sizes()[0],
                             Catch::Matchers::WithinAbs(fraction * numElements,
                                                        0.05 * numElements));
        }
        if (model_desc.UQType.find("duq") != std::string::npos &&
            fraction == 0.5) {
          verify_mean_data(InData.value(),
                           OutData.value(),
                           threshold,
                           model_desc.numIn,
                           model_desc.numOut);
        }
      }
    }
  }
}


int main(int argc, char** argv)
{
  auto db_dir = (std::filesystem::temp_directory_path() / "ams_workflow_tests");
  std::filesystem::create_directories(db_dir);

  std::string tmp_dir = db_dir / "ams-test-XXXXXX";
  std::vector<char> tmp(tmp_dir.begin(), tmp_dir.end());
  tmp.push_back('\0');
  char* dirname = mkdtemp(tmp.data());
  if (!dirname) {
    perror("mkdtemp");
  }

  db_dir = std::filesystem::path(dirname);
  std::filesystem::create_directories(db_dir);

  AMSInit();
  // Use a temp file in the build tree (std::filesystem temp)
  AMSConfigureFSDatabase(AMSDBType::AMS_HDF5, db_dir.c_str());

  Catch::Session session;

  if (int rc = session.applyCommandLine(argc, argv))
    return rc;  // bad CLI -> propagate

  int rc = session.run();  // run tests
  std::cout << "RC:" << rc << "\n";

  // Treat "warnings only" (e.g., due to skipped tests) as success for CTest.
  // Catch2 commonly uses 4 for warnings; adjust if your config differs.
  AMSFinalize();
  std::filesystem::remove_all(db_dir);
  return (rc == 0 || rc == 4) ? 0 : rc;
}
