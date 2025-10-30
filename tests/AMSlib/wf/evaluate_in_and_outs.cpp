#include <ATen/core/TensorBody.h>
#include <ATen/ops/matmul.h>
#include <c10/core/DeviceType.h>
#include <sys/types.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/torch.h>
#include <torch/types.h>

#include <cstdint>
#include <vector>

#include "AMS.h"
#include "wf/workflow.hpp"

#if defined(__AMS_ENABLE_CUDA__)
constexpr c10::DeviceType AMS_TEST_CTYPE = c10::DeviceType::CUDA;
#elif defined(__AMS_ENABLE_HIP__)
constexpr c10::DeviceType AMS_TEST_CTYPE = c10::DeviceType::CUDA;
#else
constexpr c10::DeviceType AMS_TEST_CTYPE =
    c10::DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES;
#endif

using namespace ams;

#define SIZE 32


std::vector<std::int64_t> getDims(const std::string input, char delimiter)
{
  std::vector<int64_t> tokens;
  std::stringstream ss(input);
  std::string token;

  while (std::getline(ss, token, delimiter)) {
    tokens.push_back(std::stoi(token));
  }

  return tokens;
}


// Function to read a dataset and compare it with the expected tensor
bool verifyDatasetContents(const std::string& fileName,
                           const std::string& datasetName,
                           torch::Tensor& expectedTensor)
{
  // Open the HDF5 file
  hid_t file_id = H5Fopen(fileName.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file_id < 0) {
    throw std::runtime_error("Failed to open HDF5 file.");
  }

  // Open the dataset
  hid_t dset_id = H5Dopen2(file_id, datasetName.c_str(), H5P_DEFAULT);
  if (dset_id < 0) {
    H5Fclose(file_id);
    throw std::runtime_error("Failed to open dataset.");
  }

  // Get the dataspace
  hid_t space_id = H5Dget_space(dset_id);
  if (space_id < 0) {
    H5Dclose(dset_id);
    H5Fclose(file_id);
    throw std::runtime_error("Failed to get dataspace.");
  }

  // Get the dataset dimensions
  int ndims = H5Sget_simple_extent_ndims(space_id);
  if (ndims < 0) {
    H5Sclose(space_id);
    H5Dclose(dset_id);
    H5Fclose(file_id);
    throw std::runtime_error("Failed to get number of dimensions.");
  }

  std::vector<hsize_t> dims(ndims);
  if (H5Sget_simple_extent_dims(space_id, dims.data(), NULL) < 0) {
    H5Sclose(space_id);
    H5Dclose(dset_id);
    H5Fclose(file_id);
    throw std::runtime_error("Failed to get dataset dimensions.");
  }

  // Close dataspace
  H5Sclose(space_id);

  // Flatten the dataset dimensions into a total size
  size_t totalSize = 1;
  for (const auto& dim : dims) {
    totalSize *= dim;
  }

  // Allocate a tensor to read the dataset
  auto readTensor =
      torch::empty({static_cast<int64_t>(totalSize)}, torch::kFloat);

  // Read the dataset into the tensor
  herr_t status = H5Dread(dset_id,
                          H5T_NATIVE_FLOAT,
                          H5S_ALL,
                          H5S_ALL,
                          H5P_DEFAULT,
                          readTensor.data_ptr());
  if (status < 0) {
    H5Dclose(dset_id);
    H5Fclose(file_id);
    throw std::runtime_error("Failed to read dataset.");
  }

  // Close dataset and file
  H5Dclose(dset_id);
  H5Fclose(file_id);

  // Concatenate all expected tensors into one
  expectedTensor = expectedTensor.flatten();

  // Compare the tensors
  if (!torch::allclose(readTensor, expectedTensor)) {
    throw std::runtime_error(
        "Dataset contents do not match the expected tensors.");
  }

  std::cout << "Dataset contents match the expected tensors!" << std::endl;
  return true;
}


template <typename T, torch::Dtype DType, torch::DeviceType DeviceType>
void compute(ams::AMSWorkflow& wf,
             std::vector<torch::Tensor>& orig_in,
             std::vector<torch::Tensor>& orig_inout,
             std::vector<torch::Tensor>& orig_out,
             T& broadcastVal,
             bool has_broadcast = false)
{

  auto callBack = [&](const ams::SmallVector<ams::AMSTensor>& pruned_ins,
                      ams::SmallVector<ams::AMSTensor>& pruned_inouts,
                      ams::SmallVector<ams::AMSTensor>& pruned_outs) {
    int numIn = pruned_ins.size();
    int numInOut = pruned_inouts.size();
    int numOut = pruned_outs.size();
    int numElements = 0;
    std::cout << "Num ins are " << numIn << "\n";
    std::cout << "Num inouts are " << numInOut << "\n";
    std::cout << "Num outs are " << numOut << "\n";
    if (pruned_ins.size() != 0) {
      numElements = pruned_ins[0].shape()[0];
    } else if (pruned_inouts.size() != 0) {
      numElements = pruned_inouts[0].shape()[0];
    } else {
      throw std::runtime_error(
          "call back should be called at least with some elements in batch "
          "axis");
    }

    // I am converthing all ams - tensors to torch - tensors. This is a conveniency for testing,
    // as I can execute arbitary GPU code.
    std::vector<torch::Tensor> in;
    for (auto& V : pruned_ins) {
      c10::IntArrayRef shape(V.shape().begin(), V.shape().size());
      std::cout << "Pointer of in " << V.data<float>() << "\n";
      in.push_back(torch::from_blob((void*)V.data<uint8_t>(),
                                    shape,
                                    torch::TensorOptions().dtype(DType).device(
                                        DeviceType)));
    }

    std::vector<torch::Tensor> inout;
    for (auto& V : pruned_inouts) {
      std::cout << "Pointer of inout " << V.data<float>() << "\n";
      c10::IntArrayRef shape(V.shape().begin(), V.shape().size());
      inout.push_back(torch::from_blob(
          (void*)V.data<uint8_t>(),
          shape,
          torch::TensorOptions().dtype(DType).device(DeviceType)));
    }

    std::vector<torch::Tensor> out;
    for (auto& V : pruned_outs) {
      c10::IntArrayRef shape(V.shape().begin(), V.shape().size());
      std::cout << "Pointer of out " << V.data<uint8_t>() << "\n";
      out.push_back(torch::from_blob((void*)V.data<uint8_t>(),
                                     shape,
                                     torch::TensorOptions().dtype(DType).device(
                                         DeviceType)));
    }


    torch::Tensor identity_matrix =
        torch::eye(out.size() + inout.size(),
                   torch::TensorOptions().dtype(DType).device(DeviceType));

    // Iterate over all elements
    for (int i = 0; i < numElements; i++) {
      // Create a tensor to aggregate input values
      torch::Tensor aggregate =
          torch::zeros({1, numIn + numInOut},
                       torch::TensorOptions().dtype(DType).device(DeviceType));

      // Fill aggregate with cumulative sums from `in` tensors
      for (int j = 0; j < numIn; j++) {
        aggregate[0][j] = in[j][i][0];
      }


      // Continue filling aggregate with cumulative sums from `inout` tensors
      for (int j = 0; j < numInOut; j++) {
        aggregate[0][numIn + j] = inout[j][i][0];
      }

      std::cout << "Aggr:" << aggregate << "\n";
      std::cout << "IDM" << identity_matrix << "\n";
      auto res = aggregate.matmul(identity_matrix) * 13.0;
      std::cout << "Res " << res << "\n";

      // Assign to `out` tensors using modulo indexing
      for (int j = 0; j < numOut; j++) {
        out[j][i][0] = res[0][j];
      }

      // Update `inout` tensors using modulo indexing
      for (int j = 0; j < numInOut; j++) {
        std::cout << "Setting in out for res" << res[0] << "\n";
        inout[j][i][0] = res[0][numOut + j];
      }
    }
  };
  wf.evaluate(callBack, orig_in, orig_inout, orig_out);
}


int main(int argc, char* argv[])
{

  if (argc != 10) {
    std::cout << "Wrong command line\n";
    std::cout << argv[0]
              << " <physics type (float|double)> <physics-device (cpu|cuda)> "
                 "<path-to-model> <duq-type> <threshold> <input-dim-shape "
                 "(1024,2) <output-dim-shape (1024, 2) > <path to db> "
                 "<num-in-outs\n";
    return -1;
  }
  torch::Dtype DType = torch::kFloat32;
  torch::DeviceType dev = c10::DeviceType::CPU;

  std::string Type(argv[1]);
  std::string device(argv[2]);
  std::string model_path(argv[3]);
  std::string duq_type(argv[4]);
  float threshold = std::atof(argv[5]);
  std::vector<int64_t> iShape(getDims(argv[6], ','));
  std::vector<int64_t> oShape(getDims(argv[7], ','));
  std::string db_path(argv[8]);
  int numInOuts = std::atoi(argv[9]);
  auto& db_instance = ams::db::DBManager::getInstance();
  db_instance.instantiate_fs_db(AMSDBType::AMS_HDF5, db_path);

  if (Type.compare("double") == 0) {
    DType = torch::kFloat64;
  }

  if (Type.compare("double") == 0) DType = torch::kFloat64;

  if (device.compare("cuda") == 0)
    dev = c10::DeviceType::CUDA;
  else if (device.compare("hip") == 0)
    dev = c10::DeviceType::CUDA;
  std::string domain_name("test");

  auto tOptions = torch::TensorOptions().dtype(DType).device(dev);
  std::string filename;

  {
    ams::AMSWorkflow wf =
        ams::AMSWorkflow(model_path, domain_name, threshold, 0, 1);

    filename = wf.getDBFilename();

    // How many numInOuts are we going to have in this test
    std::vector<torch::Tensor> in;
    std::vector<torch::Tensor> inout;
    std::vector<torch::Tensor> out;
    // Get the number of inputs for this test
    int numIn = iShape[iShape.size() - 1] - numInOuts;
    for (auto i = 0; i < numIn; i++) {
      in.push_back(torch::ones({SIZE, 1}, tOptions));
    }
    for (auto i = 0l; i < numInOuts; i++) {
      inout.push_back(torch::ones({SIZE, 1}, tOptions));
    }

    int numOut = oShape[oShape.size() - 1] - numInOuts;
    for (auto i = 0; i < numOut; i++) {
      out.push_back(torch::zeros({SIZE, 1}, tOptions));
    }

    // Call compute_torch
    float fbroadcastVal = 0.0;
    double dbroadcastVal = 0.0;
    std::cout << "Creating workflow with:\n";
    std::cout << "NumIn " << numIn << " " << in.size() << "\n";
    std::cout << "NumOut " << numOut << " " << out.size() << "\n";
    std::cout << "NumInOut " << numInOuts << " " << inout.size() << "\n";
    if (DType == torch::kFloat64 && dev == AMS_TEST_CTYPE)
      compute<double, torch::kFloat64, AMS_TEST_CTYPE>(
          wf, in, inout, out, dbroadcastVal, false);
    else if (DType == torch::kFloat32 && dev == AMS_TEST_CTYPE)
      compute<float, torch::kFloat32, AMS_TEST_CTYPE>(
          wf, in, inout, out, fbroadcastVal, false);
    else if (DType == torch::kFloat64 && dev == c10::DeviceType::CPU)
      compute<double, torch::kFloat64, c10::DeviceType::CPU>(
          wf, in, inout, out, dbroadcastVal, false);
    else if (DType == torch::kFloat32 && dev == c10::DeviceType::CPU)
      compute<float, torch::kFloat32, c10::DeviceType::CPU>(
          wf, in, inout, out, fbroadcastVal, false);

    // We do this, as AMS should ignore completely the threshold
    // value when it doesn't have a model
    if (model_path.empty()) threshold = 0.0;

    for (auto& V : {inout, out}) {
      for (auto i = 0; i < V.size(); i++) {
        auto data = V[i];
        if (threshold == 0.0) {
          auto correct = torch::ones(data.sizes(), data.options()) * 13;
          bool close = torch::allclose(correct, data, 1e-5, 1e-8);
          if (!close) {
            std::cout << "Values are not close\n";
            std::cout << data << "\n";
            std::cout << "Correct data are "
                      << "\n";
            std::cout << correct << "\n";
            return -1;
          }
        } else if (threshold == 0.5) {
          auto correct = torch::ones(data.sizes(), data.options());
          // Create a tensor with values [0, 1, 2, ..., size-1]
          auto indices = torch::arange(data.sizes()[0], data.options());

          auto alternating_tensor = (indices % 2) * 12;
          alternating_tensor = alternating_tensor.reshape({data.sizes()[0], 1});
          correct += alternating_tensor;
          // Use modulo operation to create alternating 0s and 1s
          bool close = torch::allclose(correct, data, 1e-5, 1e-8);
          if (!close) {
            std::cout << "Values are not close\n";
            std::cout << data << "\n";
            std::cout << "Correct data are "
                      << "\n";
            std::cout << correct << "\n";
            return -1;
          }
        } else if (threshold == 1.0) {
          auto correct = torch::ones(data.sizes(), data.options());
          bool close = torch::allclose(correct, data, 1e-5, 1e-8);
          if (!close) {
            std::cout << "Values are not close\n";
            std::cout << data << "\n";
            std::cout << "Correct data are "
                      << "\n";
            std::cout << correct << "\n";
            return -1;
          }
        } else {
          std::cout << "Unknown threshold value\n";
        }
      }
    }
    in.clear();
    inout.clear();
    out.clear();
  }

  // Reverse to compute how many physics we want.
  threshold = 1 - threshold;

  if (threshold > 0) {
    int numIn = iShape[iShape.size() - 1];
    auto expectedInput =
        torch::ones({(long)(SIZE * threshold), numIn},
                    torch::TensorOptions().dtype(torch::kFloat32));

    int numOut = iShape[oShape.size() - 1];
    auto expectedOutput =
        torch::ones({(long)(SIZE * threshold), numOut},
                    torch::TensorOptions().dtype(torch::kFloat32)) *
        13;

    std::cout << "Output size :\n" << expectedOutput.sizes() << "\n";
    std::cout << "Input size :\n" << expectedInput.sizes() << "\n";

    auto& dbg_mg = ams::db::DBManager::getInstance();
    dbg_mg.clean();
    if (threshold != 1.0)
      if (!verifyDatasetContents(filename, "input_data", expectedInput) ||
          !verifyDatasetContents(filename, "output_data", expectedOutput)) {
        std::cout << "Could not verify outputs\n";
        return -1;
      }
  }

  return 0;
}
