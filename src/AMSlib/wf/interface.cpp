#include <ATen/ops/from_blob.h>
#include <c10/core/DeviceType.h>
#include <c10/util/SmallVector.h>
#include <torch/torch.h>

#include <stdexcept>

#include "AMS.h"
#include "AMSTensor.hpp"
#include "wf/workflow.hpp"

using namespace ams;

static AMSResourceType torchDeviceToAMSDevice(c10::DeviceType dType)
{
  switch (dType) {
    case c10::DeviceType::CUDA:
      return AMSResourceType::AMS_DEVICE;
    case c10::DeviceType::HIP:
      return AMSResourceType::AMS_DEVICE;
    case c10::DeviceType::CPU:
      return AMSResourceType::AMS_HOST;
    default:
      return AMSResourceType::AMS_UNKNOWN;
  }
  return AMSResourceType::AMS_UNKNOWN;
}

static AMSDType torchDTypeToAMSType(torch::Dtype dtype)
{
  static const std::unordered_map<torch::Dtype, AMSDType> dtypeMap = {
      {torch::kFloat32, AMSDType::AMS_SINGLE},
      {torch::kFloat, AMSDType::AMS_SINGLE},  // Alias for float32
      {torch::kFloat64, AMSDType::AMS_DOUBLE},
      {torch::kDouble, AMSDType::AMS_DOUBLE},  // Alias for float64
      {torch::kInt32, AMSDType::AMS_UNKNOWN_TYPE},
      {torch::kInt64, AMSDType::AMS_UNKNOWN_TYPE},
      {torch::kBool, AMSDType::AMS_UNKNOWN_TYPE},
      {torch::kUInt8, AMSDType::AMS_UNKNOWN_TYPE},
      {torch::kInt8, AMSDType::AMS_UNKNOWN_TYPE},
      {torch::kHalf, AMSDType::AMS_UNKNOWN_TYPE},
      {torch::kBFloat16, AMSDType::AMS_UNKNOWN_TYPE}};

  return dtypeMap.count(dtype) ? dtypeMap.at(dtype)
                               : AMSDType::AMS_UNKNOWN_TYPE;
}

static c10::DeviceType amsToTorchDevice(const ams::AMSResourceType resource)
{
  if (resource == ams::AMSResourceType::AMS_HOST)
    return c10::DeviceType::CPU;
  else if (resource == ams::AMSResourceType::AMS_DEVICE)
#if defined(__AMS_ENABLE_CUDA__)
    return c10::DeviceType::CUDA;
#elif defined(__AMS_ENABLE_HIP__)
    return c10::DeviceType::CUDA;
#endif

  throw std::runtime_error("Unknown ams resource type");
  return c10::DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES;
}

static c10::ScalarType amsToTorchDType(const ams::AMSDType dType)
{
  if (dType == ams::AMSDType::AMS_SINGLE)
    return torch::kFloat32;
  else if (dType == ams::AMSDType::AMS_DOUBLE)
    return torch::kFloat64;

  throw std::runtime_error("Unknown ams data type");
  return torch::kHalf;
}


static ams::SmallVector<ams::AMSTensor> torchToAMSTensors(
    ams::MutableArrayRef<torch::Tensor> tensorVector)
{
  ams::SmallVector<ams::AMSTensor> ams_tensors;
  for (auto tensor : tensorVector) {
    // We should be able to completely remove these conversion by using some template "magic."
    // I will leave these for later though
    auto dType = torchDTypeToAMSType(tensor.scalar_type());
    auto rType = torchDeviceToAMSDevice(tensor.device().type());
    // In both cases, I am effectively only forwarding the pointer of begin/end to ams.
    // this is a cheap operating. It should boil down to: shapes.start = tensor.sizes.start, shapes.end = tensor.sizes.end;
    auto shapes = ArrayRef(tensor.sizes().begin(), tensor.strides().size());
    auto strides = ArrayRef(tensor.strides().begin(), tensor.strides().size());
    if (dType == AMSDType::AMS_SINGLE) {
      ams_tensors.push_back(
          AMSTensor::view(tensor.data_ptr<float>(), shapes, strides, rType));
    } else if (dType == AMSDType::AMS_DOUBLE) {
      ams_tensors.push_back(
          AMSTensor::view(tensor.data_ptr<double>(), shapes, strides, rType));
    }
  }
  return ams_tensors;
}

static ams::SmallVector<torch::Tensor> amsToTorchTensors(
    const ams::SmallVector<ams::AMSTensor>& amsTensorVector)
{
  ams::SmallVector<torch::Tensor> ams_tensors;
  for (auto& tensor : amsTensorVector) {
    // We should be able to completely remove these conversion by using some template "magic."
    // I will leave these for later though
    auto dType = amsToTorchDType(tensor.dType());
    auto deviceType = amsToTorchDevice(tensor.location());
    // In both cases, I am effectively only forwarding the pointer of begin/end to ams.
    // this is a cheap operating. It should boil down to: shapes.start = tensor.sizes.start, shapes.end = tensor.sizes.end;
    c10::SmallVector<long> shapes(tensor.shape().begin(), tensor.shape().end());
    c10::SmallVector<long> strides(tensor.strides().begin(),
                                   tensor.strides().end());
    ams_tensors.push_back(torch::from_blob(
        tensor.raw_data(),
        shapes,
        strides,
        torch::TensorOptions().dtype(dType).device(deviceType)));
  }
  return std::move(ams_tensors);
}

void callApplication(ams::DomainLambda CallBack,
                     ams::MutableArrayRef<torch::Tensor> Ins,
                     ams::MutableArrayRef<torch::Tensor> InOuts,
                     ams::MutableArrayRef<torch::Tensor> Outs)
{
  auto AMSIns = torchToAMSTensors(Ins);
  auto AMSInOuts = torchToAMSTensors(InOuts);
  auto AMSOuts = torchToAMSTensors(Outs);
  CallBack(AMSIns, AMSInOuts, AMSOuts);
  return;
}

void callAMS(ams::AMSWorkflow* executor,
             DomainLambda Physics,
             const ams::SmallVector<ams::AMSTensor>& ins,
             ams::SmallVector<ams::AMSTensor>& inouts,
             ams::SmallVector<ams::AMSTensor>& outs)
{
  ams::SmallVector<torch::Tensor> tins = amsToTorchTensors(ins);
  ams::SmallVector<torch::Tensor> tinouts = amsToTorchTensors(inouts);
  ams::SmallVector<torch::Tensor> touts = amsToTorchTensors(outs);

  executor->evaluate(Physics, tins, tinouts, touts);
}
