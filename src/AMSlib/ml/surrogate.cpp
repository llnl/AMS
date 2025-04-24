
#include <c10/core/ScalarTypeToTypeMeta.h>
#include <torch/script.h>

#include <experimental/filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>

#include "AMS.h"
#include "surrogate.hpp"
#include "wf/debug.h"
#include "wf/utils.hpp"

using namespace ams;
static std::string getDTypeAsString(torch::Dtype dtype)
{
  if (dtype == torch::kFloat32) return "float32";
  if (dtype == torch::kFloat64) return "float64";
  if (dtype == torch::kInt32) return "int32";
  if (dtype == torch::kInt64) return "int64";
  if (dtype == torch::kBool) return "bool";
  if (dtype == torch::kUInt8) return "uint8";
  if (dtype == torch::kInt8) return "int8";

  // Add other types as needed
  return "unknown";
}

static std::string getAMSDTypeAsString(AMSDType dType)
{
  if (dType == AMS_SINGLE)
    return "float32";
  else if (dType == AMS_DOUBLE)
    return "float64";
  return "unknown";
}

static std::string getAMSResourceTypeAsString(AMSResourceType res)
{
  if (res == ams::AMS_DEVICE)
    return "device";
  else if (res == ams::AMS_HOST)
    return "host";
  return "unknown-device";
}


SurrogateModel::SurrogateModel(std::string& model_path, bool isDeltaUQ)
    : _model_path(model_path), _is_DeltaUQ(isDeltaUQ)
{

  std::experimental::filesystem::path Path(model_path);
  std::error_code ec;

  if (!std::experimental::filesystem::exists(Path, ec)) {
    FATAL(Surrogate,
          "Path to Surrogate Model (%s) Does not exist",
          model_path.c_str())
  }

  try {
    module = torch::jit::load(model_path);
  } catch (const c10::Error& e) {
    printf("Error opening %s\n", model_path.c_str());
  }
  std::tie(model_device, torch_device) = getModelResourceType();
  std::tie(model_dtype, torch_dtype) = getModelDataType();
  DBG(SurrogateModel,
      "Loaded model with type %s on device %s",
      getAMSDTypeAsString(model_dtype).c_str(),
      getAMSResourceTypeAsString(model_device).c_str());
}

std::tuple<AMSResourceType, torch::DeviceType> SurrogateModel::
    getModelResourceType()
{
  // Iterate through the parameters to determine the device
  for (const auto& parameter : module.parameters()) {
    // Return the device of the first parameter found
    switch (parameter.device().type()) {
      case c10::DeviceType::CUDA:
      case c10::DeviceType::HIP:
        return std::make_tuple(AMS_DEVICE, parameter.device().type());
      case c10::DeviceType::CPU:
        return std::make_tuple(AMS_HOST, parameter.device().type());
      default:
        continue;
    }
  }

  // If no parameters are found, check the buffers
  for (const auto& buffer : module.buffers()) {
    switch (buffer.device().type()) {
      case c10::DeviceType::CUDA:
      case c10::DeviceType::HIP:
        return std::make_tuple(AMS_DEVICE, buffer.device().type());
      case c10::DeviceType::CPU:
        return std::make_tuple(AMS_HOST, buffer.device().type());
      default:
        continue;
    }
  }

  // If no parameters or buffers are found, default to unknown
  FATAL(Surrogate,
        "Cannot determine device type of model %s",
        _model_path.c_str());
  return std::make_tuple(AMS_UNKNOWN,
                         c10::DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES);
}

std::tuple<AMSDType, torch::Dtype> SurrogateModel::getModelDataType()
{
  AMSDType dParamType = AMSDType::AMS_DOUBLE;
  torch::Dtype torchType = at::kDouble;
  for (const auto& parameter : module.parameters()) {
    // Return the device of the first parameter found
    if (parameter.dtype() == at::kFloat) {
      dParamType = AMS_SINGLE;
      torchType = at::kFloat;
    } else if (parameter.dtype() == at::kDouble) {
      dParamType = AMS_DOUBLE;
      torchType = at::kDouble;
    } else {
      throw std::runtime_error(std::string("Invalid datatype ") +
                               std::string(parameter.dtype().name()));
    }
  }

  // Verify
  for (const auto& parameter : module.parameters()) {
    if (parameter.dtype() != torchType)
      throw std::runtime_error("Provided model has mixed data types");
  }

  AMSDType dBufferType = dParamType;
  for (const auto& buffer : module.buffers()) {
    // Return the device of the first parameter found
    if (buffer.dtype() == at::kFloat) {
      dBufferType = AMS_SINGLE;
      torchType = at::kFloat;
    } else if (buffer.dtype() == at::kDouble) {
      dBufferType = AMS_DOUBLE;
      torchType = at::kDouble;
    } else {
      throw std::runtime_error(std::string("Invalid datatype ") +
                               std::string(buffer.dtype().name()));
    }
  }
  // Verify
  for (const auto& buffer : module.buffers()) {
    if (buffer.dtype() != torchType)
      throw std::runtime_error("Provided model has mixed data types");
  }

  if (dParamType != dBufferType)
    throw std::runtime_error(
        "Provided model has mixed data types between parameters and buffers");

  DBG(Surrogate,
      "Detected model data type %s %s",
      getDTypeAsString(torchType).c_str(),
      getAMSDTypeAsString(dParamType).c_str());
  return std::make_tuple(dParamType, torchType);
}


std::tuple<torch::Tensor, torch::Tensor> SurrogateModel::_computeDetlaUQ(
    c10::IValue& deltaUQTuple,
    AMSUQPolicy policy,
    float threshold)
{
  at::Tensor output_mean_tensor = deltaUQTuple.toTuple()
                                      ->elements()[0]
                                      .toTensor()
                                      .set_requires_grad(false)
                                      .detach();
  at::Tensor output_stdev_tensor = deltaUQTuple.toTuple()
                                       ->elements()[1]
                                       .toTensor()
                                       .set_requires_grad(false)
                                       .detach();
  auto outer_dim = output_stdev_tensor.sizes().size() - 1;
  if (policy != AMSUQPolicy::AMS_DELTAUQ_MAX &&
      policy != AMSUQPolicy::AMS_DELTAUQ_MEAN)
    throw std::runtime_error("Invalid DELTA_UQ policy");

  if (policy == AMSUQPolicy::AMS_DELTAUQ_MEAN) {
    auto mean = output_stdev_tensor.mean(outer_dim);
    auto predicate = mean < threshold;
    return std::make_tuple(std::move(output_mean_tensor), std::move(predicate));
  } else if (policy == AMSUQPolicy::AMS_DELTAUQ_MAX) {
    auto tmp = output_stdev_tensor.max(outer_dim);
    torch::Tensor max = std::get<0>(tmp);
    auto predicate = max < threshold;
    return std::make_tuple(std::move(output_mean_tensor), std::move(predicate));
  }
  throw std::runtime_error("Invalid DELTA_UQ policy");
}


std::tuple<torch::Tensor, torch::Tensor> SurrogateModel::_evaluate(
    torch::Tensor& inputs,
    AMSUQPolicy policy,
    float threshold)
{
  if (inputs.dtype() != torch_dtype) {
    throw std::runtime_error(
        "Received inputs of wrong dType. Model is expecting " +
        getDTypeAsString(torch::typeMetaToScalarType(inputs.dtype())) +
        " and model is " + getDTypeAsString(torch_dtype));
  }
  c10::InferenceMode guard(true);
  auto out = module.forward({inputs});
  if (_is_DeltaUQ) {
    return _computeDetlaUQ(out, policy, threshold);
  }

  at::Tensor output_tensor = out.toTensor().set_requires_grad(false).detach();
  // Randomly select indices to set to True
  torch::Tensor predicate =
      torch::zeros({output_tensor.sizes()[0], 1}, torch::kBool);
  auto indices = torch::randperm(output_tensor.sizes()[0])
                     .slice(0, 0, threshold * output_tensor.sizes()[0]);

  // Set selected indices to True
  predicate.index_put_({indices, 0}, true);
  return std::make_tuple(std::move(output_tensor), std::move(predicate));
}


std::tuple<torch::Tensor, torch::Tensor> SurrogateModel::evaluate(
    ams::MutableArrayRef<at::Tensor> Inputs,
    AMSUQPolicy policy,
    float threshold)
{
  if (Inputs.size() == 0) {
    throw std::invalid_argument(
        "Input Vector should always contain at least one tensor");
  }

  torch::DeviceType InputDevice = Inputs[0].device().type();
  torch::Dtype InputDType = torch::typeMetaToScalarType(Inputs[0].dtype());
  auto CAxis = Inputs[0].sizes().size() - 1;

  // Verify input/device matching
  for (auto& In : Inputs) {
    if (InputDevice != In.device().type()) {
      throw std::invalid_argument(
          "Unsupported feature, application domain tensors are on different "
          "devices\n");
    }
    if (InputDType != torch::typeMetaToScalarType(In.dtype())) {
      throw std::invalid_argument(
          "Unsupported feature, application domain tensors have different data "
          "types\n");
    }
  }
  c10::SmallVector<torch::Tensor> ConvertedInputs(Inputs.begin(), Inputs.end());
  // If either the model's execution device or the data type differ
  // in respect to the inputs we need to handle this separately.
  if (InputDevice != torch_device || InputDType != torch_dtype) {
    for (int i = 0; i < ConvertedInputs.size(); i++) {
      ConvertedInputs[i] = ConvertedInputs[i].to(torch_device, torch_dtype);
    }
  }

  auto ITensor = torch::cat(ConvertedInputs, CAxis);
  DBG(Surrogate,
      "Input concatenated tensor is %s",
      shapeToString(ITensor).c_str());

  auto [OTensor, Predicate] = _evaluate(ITensor, policy, threshold);
  if (InputDevice != torch_device) {
    OTensor = OTensor.to(InputDevice);
    Predicate = Predicate.to(InputDevice);
  }
  return std::make_tuple(std::move(OTensor), std::move(Predicate));
}


std::unordered_map<std::string, std::shared_ptr<SurrogateModel>>
    SurrogateModel::instances;

#if 0
//#include <wf/resource_manager.hpp>

//#include <c10/core/Allocator.h>
//#include <c10/core/CPUAllocator.h>
//#include <c10/core/DeviceType.h>
//#include <c10/core/ScalarType.h>
//#include <torch/script.h>  // One-stop header.
// #include <wf/debug.h>


//struct C10_API AMSCPUAllocator final : at::Allocator {
//  AMSCPUAllocator() = default;
//  at::DataPtr allocate(size_t nbytes) const override
//  {
//    auto& rm = ams::ResourceManager::getInstance();
//    uint8_t* data = rm.allocate<uint8_t>(nbytes, AMSResourceType::AMS_HOST);
//
//    return {(void*)data,
//            (void*)data,
//            &ReportAndDelete,
//            at::Device(at::DeviceType::CPU)};
//  }
//
//  static void ReportAndDelete(void* ptr)
//  {
//    if (!ptr) {
//      return;
//    }
//    auto& rm = ams::ResourceManager::getInstance();
//    rm.deallocate(ptr, AMSResourceType::AMS_HOST);
//  }
//
//  at::DeleterFnPtr raw_deleter() const override { return &ReportAndDelete; }
//};
//
//
//AMSCPUAllocator ams_torch;
//
//
//void set_cpu_torch_allocator()
//{
//  SetAllocator(c10::DeviceType::CPU, &ams_torch, (uint8_t)(2 ^ 8 - 1));
//  SetCPUAllocator(&ams_torch, (uint8_t)(2 ^ 8 - 1));
//}
//
//
#endif
