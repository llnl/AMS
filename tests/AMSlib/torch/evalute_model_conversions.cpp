#include <c10/core/DeviceType.h>
#include <c10/util/SmallVector.h>
#include <torch/torch.h>
#include <torch/types.h>

#include <algorithm>  // For std::shuffle
#include <iostream>
#include <random>  // For std::mt19937 and std::uniform_int_distribution
#include <stdexcept>
#include <string>

#include "AMS.h"
#include "ml/surrogate.hpp"

void printTensorShape(const torch::Tensor& tensor)
{
  std::cout << "Tensor shape: [";
  for (size_t i = 0; i < tensor.sizes().size(); ++i) {
    std::cout << tensor.size(i);
    if (i != tensor.sizes().size() - 1) {
      std::cout << ", ";
    }
  }
  std::cout << "]" << std::endl;
}

// Helper function to convert torch::Dtype to a string
std::string dtypeToString(torch::Dtype dtype)
{
  static const std::unordered_map<torch::Dtype, std::string> dtypeMap = {
      {torch::kFloat32, "float32"},
      {torch::kFloat, "float32"},  // Alias for float32
      {torch::kFloat64, "float64"},
      {torch::kDouble, "float64"},  // Alias for float64
      {torch::kInt32, "int32"},
      {torch::kInt64, "int64"},
      {torch::kBool, "bool"},
      {torch::kUInt8, "uint8"},
      {torch::kInt8, "int8"},
      {torch::kHalf, "float16"},
      {torch::kBFloat16, "bfloat16"}};
  return dtypeMap.count(dtype) ? dtypeMap.at(dtype) : "unknown dtype";
}

// Helper function to convert c10::DeviceType to a string
std::string deviceTypeToString(c10::DeviceType deviceType)
{
  static const std::unordered_map<c10::DeviceType, std::string> deviceMap = {
      {c10::DeviceType::CPU, "CPU"},
      {c10::DeviceType::CUDA, "CUDA"},
      {c10::DeviceType::HIP, "HIP"},
      {c10::DeviceType::FPGA, "FPGA"},
      {c10::DeviceType::XLA, "XLA"},
      {c10::DeviceType::Meta, "Meta"},
      {c10::DeviceType::ORT, "ORT"}};
  return deviceMap.count(deviceType) ? deviceMap.at(deviceType)
                                     : "unknown device";
}

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

std::vector<int> generateRandomVector(int target_sum, int size)
{
  if (target_sum < size) {
    throw std::invalid_argument(
        "Target sum must be at least equal to the size of the vector.");
  }

  std::vector<int> result(
      size, 1);  // Start with each element as 1 (minimum positive integer).
  target_sum -=
      size;  // Reduce the remaining sum by the size (since all elements are 1).

  std::random_device rd;
  std::mt19937 gen(0);
  std::uniform_int_distribution<> dis(0, target_sum);

  // Generate random values and distribute the remaining sum
  for (int i = 0; i < target_sum; ++i) {
    int index = dis(gen) % size;  // Pick a random index
    ++result[index];              // Increment the value at the chosen index
  }

  // Shuffle the vector for more randomness
  std::shuffle(result.begin(), result.end(), gen);

  return result;
}


bool verify(torch::Tensor& input,
            torch::Tensor& output,
            torch::Tensor& predicate,
            float threshold,
            torch::Dtype model_dtype)
{

  // our current 'interface' always return the dtype of the model.
  if (output.dtype() != model_dtype) {
    throw ::std::runtime_error(
        "Tensors should have the data type of the model");
  }

  if (input.dtype() != output.dtype()) {
    output = output.to(input.dtype());
  }

  bool close = torch::allclose(input, output, 1e-5, 1e-8);

  if (!close) throw std::runtime_error("Tensors are not identical");

  torch::Tensor float_tensor = predicate.to(torch::kDouble);
  // Calculate the probability (mean of the tensor)
  double probability = float_tensor.mean().item<double>();
  std::cout << "probability is " << probability << "\n";
  if (probability != threshold)
    throw std::runtime_error(
        "Expecing a probability of 0.0 in the case of threshold <0>");
  return true;
}

void test(SurrogateModel& model,
          std::vector<int64_t>& iDims,
          std::vector<int64_t>& oDims,
          ams::AMSUQPolicy policy)
{
  auto model_type = model.getModelDataType();
  auto model_device = model.getModelResourceType();
  std::vector<torch::Dtype> SupportedDTypes = {torch::kFloat32,
                                               torch::kFloat64};
  auto inputShapes = generateRandomVector(iDims[iDims.size() - 1], 3);
  std::vector<c10::DeviceType> SupportedDevices = {c10::DeviceType::CPU};
  if (torch::cuda::is_available() && torch::cuda::device_count() > 0) {
    SupportedDevices.push_back(c10::DeviceType::CUDA);
  }

  for (auto type : SupportedDTypes) {
    for (auto device : SupportedDevices) {
      ams::SmallVector<torch::Tensor> inputs;
      for (auto outer : inputShapes) {
        std::vector<int64_t> partialShape(iDims.begin(), iDims.end());
        partialShape[partialShape.size() - 1] = outer;
        auto inp =
            torch::rand(partialShape,
                        torch::TensorOptions().dtype(type).device(device));
        printTensorShape(inp);
        inputs.push_back(inp);
      }

      std::cout << "Testing with input tensor type: " << dtypeToString(type)
                << " on device: " << deviceTypeToString(device) << "\n";

      std::cout << "Testing with model parameter types : "
                << dtypeToString(std::get<1>(model_type)) << " on device: "
                << deviceTypeToString(std::get<1>(model_device)) << "\n";


      {
        std::cout << "Staring Test-1 with random-uq and threshold of 0.0\n";
        auto [out, predicate] = model.evaluate(inputs, policy, 0.0);
        c10::SmallVector<torch::Tensor> data(inputs.begin(), inputs.end());
        auto input = torch::cat(data, iDims.size() - 1);
        std::cout << "Output of model is of type "
                  << dtypeToString((torch::typeMetaToScalarType(out.dtype())))
                  << "\n";
        std::cout << "Input of model is of type "
                  << dtypeToString(
                         (torch::typeMetaToScalarType(inputs[0].dtype())))
                  << "\n";
        verify(input, out, predicate, 0.0, std::get<1>(model_type));
        std::cout << "SUCCESS\n";
      }
      {
        std::cout << "Staring Test-2 with random-uq and threshold of 0.5\n";
        auto [out, predicate] = model.evaluate(inputs, policy, 0.5);
        c10::SmallVector<torch::Tensor> data(inputs.begin(), inputs.end());
        auto input = torch::cat(data, iDims.size() - 1);
        verify(input, out, predicate, 0.5, std::get<1>(model_type));
        std::cout << "SUCCESS\n";
      }
      {
        std::cout << "Staring Test-3 with random-uq and threshold of 1.0\n";
        auto [out, predicate] = model.evaluate(inputs, policy, 1.0);
        c10::SmallVector<torch::Tensor> data(inputs.begin(), inputs.end());
        auto input = torch::cat(data, iDims.size() - 1);
        verify(input, out, predicate, 1.0, std::get<1>(model_type));
        std::cout << "SUCCESS\n";
      }
    }
  }
}

int main(int argc, char* argv[])
{
  if (argc != 5) {
    std::cerr << "Wrong command line, expecting , "
                 "<input-dim-shape (1024,2,4)> <output-dim-shape> (1024, 2, "
                 "6) <model-path> <duq_type (mean|max)>\n";
    return -1;
  }

  std::vector<int64_t> iShape(getDims(argv[1], ','));
  std::vector<int64_t> oShape(getDims(argv[2], ','));
  std::string model_path(argv[3]);
  std::string uq(argv[4]);
  bool isDeltaUQ = true;
  if (uq.compare("random") == 0) isDeltaUQ = false;
  auto model = SurrogateModel::getInstance(model_path, isDeltaUQ);
  if (std::string(argv[4]).compare("duq_mean") == 0) {
    test(*model, iShape, oShape, ams::AMSUQPolicy::AMS_DELTAUQ_MEAN);
  } else if (std::string(argv[4]).compare("duq_max") == 0) {
    test(*model, iShape, oShape, ams::AMSUQPolicy::AMS_DELTAUQ_MAX);
  } else if (std::string(argv[4]).compare("random") == 0) {
    test(*model, iShape, oShape, ams::AMSUQPolicy::AMS_RANDOM);
  } else {
    std::cout << "Unknown dUQ \n";
    return 1;
  }
}
