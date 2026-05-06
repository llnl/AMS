#include <ATen/ops/from_blob.h>
#include <c10/core/DeviceType.h>
#include <c10/util/SmallVector.h>
#include <torch/torch.h>

#include <stdexcept>

#include "AMS.h"
#include "AMSTensor.hpp"
#include "wf/workflow.hpp"

using namespace ams;

// dtype/device helpers
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
      {torch::kInt32, AMSDType::AMS_INT32},
      {torch::kInt64, AMSDType::AMS_INT64},
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
  else if (dType == ams::AMSDType::AMS_INT32)
    return torch::kInt32;
  else if (dType == ams::AMSDType::AMS_INT64)
    return torch::kInt64;

  throw std::runtime_error("Unknown ams data type");
  return torch::kHalf;
}

// single tensor
static ams::AMSTensor torchToAMSTensorView(torch::Tensor& tensor)
{
  auto dType = torchDTypeToAMSType(tensor.scalar_type());
  auto rType = torchDeviceToAMSDevice(tensor.device().type());

  auto shapes = ams::ArrayRef(tensor.sizes().begin(), tensor.sizes().size());
  auto strides =
      ams::ArrayRef(tensor.strides().begin(), tensor.strides().size());

  switch (dType) {
    case AMSDType::AMS_SINGLE:
      return AMSTensor::view(tensor.data_ptr<float>(), shapes, strides, rType);

    case AMSDType::AMS_DOUBLE:
      return AMSTensor::view(tensor.data_ptr<double>(), shapes, strides, rType);

    case AMSDType::AMS_INT32:
      return AMSTensor::view(tensor.data_ptr<int32_t>(),
                             shapes,
                             strides,
                             rType);

    case AMSDType::AMS_INT64:
      return AMSTensor::view(tensor.data_ptr<int64_t>(),
                             shapes,
                             strides,
                             rType);

    default:
      throw std::runtime_error("torchToAMSTensorView: unsupported Torch dtype");
  }
}

static torch::Tensor amsToTorchTensorView(const ams::AMSTensor& tensor)
{
  auto dType = amsToTorchDType(tensor.dType());
  auto deviceType = amsToTorchDevice(tensor.location());

  c10::SmallVector<long> shapes(tensor.shape().begin(), tensor.shape().end());
  c10::SmallVector<long> strides(tensor.strides().begin(),
                                 tensor.strides().end());

  return torch::from_blob(tensor.raw_data(),
                          shapes,
                          strides,
                          torch::TensorOptions().dtype(dType).device(
                              deviceType));
}

// flat containers
static ams::SmallVector<ams::AMSTensor> torchToAMSTensors(
    ams::MutableArrayRef<torch::Tensor> tensorVector)
{
  ams::SmallVector<ams::AMSTensor> ams_tensors;
  for (auto& tensor : tensorVector) {
    ams_tensors.push_back(torchToAMSTensorView(tensor));
  }
  return ams_tensors;
}

static ams::SmallVector<torch::Tensor> amsToTorchTensors(
    const ams::SmallVector<ams::AMSTensor>& amsTensorVector)
{
  ams::SmallVector<torch::Tensor> torch_tensors;
  for (const auto& tensor : amsTensorVector) {
    torch_tensors.push_back(amsToTorchTensorView(tensor));
  }
  return torch_tensors;
}

static ams::AMSTensorMap torchDictToAMSTensorMap(
    const c10::Dict<std::string, torch::Tensor>& dict)
{
  ams::AMSTensorMap out;

  for (const auto& item : dict) {
    const std::string name = item.key();
    torch::Tensor tensor = item.value();

    out.emplace(name, torchToAMSTensorView(tensor));
  }

  return out;
}

static c10::Dict<std::string, torch::Tensor> amsTensorMapToTorchDict(
    const ams::AMSTensorMap& store)
{
  c10::Dict<std::string, torch::Tensor> out;

  for (const auto& [name, tensor] : store) {
    out.insert(name, amsToTorchTensorView(tensor));
  }

  return out;
}

// key helpers
static c10::Dict<std::string, torch::Tensor> toStringTensorDict(
    const c10::IValue& value)
{
  c10::Dict<std::string, torch::Tensor> out;

  auto generic = value.toGenericDict();
  for (const auto& kv : generic) {
    out.insert(kv.key().toStringRef(), kv.value().toTensor());
  }

  return out;
}

static c10::impl::GenericDict toStringIValueDict(const c10::IValue& value)
{
  c10::impl::GenericDict out(c10::StringType::get(), c10::AnyType::get());

  auto generic = value.toGenericDict();
  for (const auto& kv : generic) {
    out.insert(kv.key().toStringRef(), kv.value());
  }

  return out;
}

// homogeneous graphs
static ams::AMSHomogeneousGraph torchToAMSHomogeneousGraph(
    const c10::Dict<std::string, torch::Tensor>& g)
{
  return torchDictToAMSTensorMap(g);
}

static c10::Dict<std::string, torch::Tensor> amsToTorchHomogeneousGraph(
    const ams::AMSHomogeneousGraph& g)
{
  return amsTensorMapToTorchDict(g);
}

// heterogeneous graphs
static std::unordered_map<std::string, ams::AMSTensorMap>
torchDictToAMSNodeStores(const c10::impl::GenericDict& dict)
{
  std::unordered_map<std::string, ams::AMSTensorMap> out;

  for (const auto& item : dict) {
    out.emplace(std::string(item.key().toStringRef()),
                torchDictToAMSTensorMap(toStringTensorDict(item.value())));
  }

  return out;
}

static std::unordered_map<ams::EdgeType, ams::AMSTensorMap, ams::EdgeTypeHash>
torchDictToAMSEdgeStores(const c10::impl::GenericDict& dict)
{
  std::unordered_map<ams::EdgeType, ams::AMSTensorMap, ams::EdgeTypeHash> out;

  for (const auto& item : dict) {
    ams::EdgeType edge_type =
        edgeTypeFromString(std::string(item.key().toStringRef()));

    out.emplace(std::move(edge_type),
                torchDictToAMSTensorMap(toStringTensorDict(item.value())));
  }

  return out;
}

static ams::AMSHeterogeneousGraph torchToAMSHeterogeneousGraph(
    const c10::IValue& value)
{
  auto g = value.toGenericDict();

  ams::AMSHeterogeneousGraph out;

  c10::IValue nodes_ivalue;
  c10::IValue edges_ivalue;
  c10::IValue global_ivalue;

  bool has_nodes = false;
  bool has_edges = false;
  bool has_global = false;

  for (const auto& kv : g) {
    const auto key = kv.key().toStringRef();
    if (key == "node_stores") {
      nodes_ivalue = kv.value();
      has_nodes = true;
    } else if (key == "edge_stores") {
      edges_ivalue = kv.value();
      has_edges = true;
    } else if (key == "global_store") {
      global_ivalue = kv.value();
      has_global = true;
    }
  }

  if (!has_nodes) {
    throw std::runtime_error(
        "torchToAMSHeterogeneousGraph: missing node_stores");
  }
  if (!has_edges) {
    throw std::runtime_error(
        "torchToAMSHeterogeneousGraph: missing edge_stores");
  }
  if (!has_global) {
    throw std::runtime_error(
        "torchToAMSHeterogeneousGraph: missing global_store");
  }

  out.node_stores = torchDictToAMSNodeStores(toStringIValueDict(nodes_ivalue));
  out.edge_stores = torchDictToAMSEdgeStores(toStringIValueDict(edges_ivalue));
  out.global_store = torchDictToAMSTensorMap(toStringTensorDict(global_ivalue));

  return out;
}

static c10::impl::GenericDict amsNodeStoresToTorchDict(
    const std::unordered_map<std::string, ams::AMSTensorMap>& node_stores)
{
  c10::impl::GenericDict out(c10::StringType::get(), c10::AnyType::get());

  for (const auto& [store_name, store] : node_stores) {
    out.insert(store_name, c10::IValue(amsTensorMapToTorchDict(store)));
  }

  return out;
}

static c10::impl::GenericDict amsEdgeStoresToTorchDict(
    const std::unordered_map<ams::EdgeType,
                             ams::AMSTensorMap,
                             ams::EdgeTypeHash>& edge_stores)
{
  c10::impl::GenericDict out(c10::StringType::get(), c10::AnyType::get());

  for (const auto& [edge_type, store] : edge_stores) {
    out.insert(ams::edgeTypeToString(edge_type),
               c10::IValue(amsTensorMapToTorchDict(store)));
  }

  return out;
}

static c10::impl::GenericDict amsToTorchHeterogeneousGraph(
    const ams::AMSHeterogeneousGraph& g)
{
  c10::impl::GenericDict out(c10::StringType::get(), c10::AnyType::get());

  out.insert("node_stores",
             c10::IValue(amsNodeStoresToTorchDict(g.node_stores)));
  out.insert("edge_stores",
             c10::IValue(amsEdgeStoresToTorchDict(g.edge_stores)));
  out.insert("global_store",
             c10::IValue(amsTensorMapToTorchDict(g.global_store)));

  return out;
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
