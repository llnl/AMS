#include "ml/Model.hpp"

#include <fmt/format.h>

#include <filesystem>

#include "AbstractModel.hpp"
#include "wf/fmt_helpers.hpp"

using namespace ams;
using namespace ams::ml;
namespace fs = std::filesystem;


BaseModel::BaseModel(const AbstractModel& AModel)
    : JITModel{torch::jit::load(AModel.getPath().string())},
      ModelDevice(torch::kCPU),
      ModelDType(c10::ScalarType::Undefined)
{
}

AMSExpected<std::unique_ptr<BaseModel>> BaseModel::load(
    const AbstractModel& Descriptor)
{
  std::error_code EC;
  if (!fs::exists(Descriptor.getPath(), EC))
    return AMS_MAKE_ERROR(AMSErrorType::FileDoesNotExist, EC.message());

  AMSExpected<std::unique_ptr<BaseModel>> BModelOrErr =
      [&]() -> AMSExpected<std::unique_ptr<BaseModel>> {
    try {
      return AMSExpected<std::unique_ptr<BaseModel>>{
          std::unique_ptr<BaseModel>(new BaseModel(Descriptor))};
    } catch (const c10::Error& EC) {
      // Here I am using a more verbose error message (that will inlclude stack frames and line info of the internal torch library).
      // These tend to be useful to debug.
      return AMS_MAKE_ERROR(AMSErrorType::TorchInternal, EC.what());
    }
  }();

  if (!BModelOrErr) return std::move(BModelOrErr);

  auto BModel = std::move(*BModelOrErr);

  auto& Module = BModel->getJITModel();

  try {
    auto AMSModelDTypeMethod = Module.find_method("get_ams_dtype");
    if (!AMSModelDTypeMethod) {
      return AMS_MAKE_ERROR(AMSErrorType::InvalidModel,
                            fmt::format("Model store under: {} is not JIT-ed "
                                        "and "
                                        "stored through AMS infrastructure "
                                        "unable to access 'get_ams_dtype' "
                                        "method",
                                        Descriptor.getPath()));
    }

    auto& DTypeIMethod = AMSModelDTypeMethod.value();
    const auto& DTypeIValue = DTypeIMethod({});
    if (!DTypeIValue.isScalar()) {
      return AMS_MAKE_ERROR(AMSErrorType::InvalidModel,
                            fmt::format("Model stored under: {} does not "
                                        "provide "
                                        "the proper data-type type",
                                        Descriptor.getPath()));
    }
    BModel->setDType(DTypeIValue.toScalarType());
  } catch (const c10::Error& EC) {
    return AMS_MAKE_ERROR(AMSErrorType::TorchInternal,
                          fmt::format("Error when getting the model data "
                                      "type\n{}",
                                      EC.what()));
  }


  try {
    const auto& AMSModelDeviceTypeMethod = Module.find_method("get_ams_device");
    if (!AMSModelDeviceTypeMethod) {
      return AMS_MAKE_ERROR(AMSErrorType::InvalidModel,
                            fmt::format("Model store under: {} is not JIT-ed "
                                        "and "
                                        "stored through AMS infrastructure "
                                        "unable to access 'get_ams_device' "
                                        "method",
                                        Descriptor.getPath()));
    }

    auto& DeviceIVMethod = *AMSModelDeviceTypeMethod;
    const auto& DeviceIVal = DeviceIVMethod({});

    if (!DeviceIVal.isDevice())
      return AMS_MAKE_ERROR(AMSErrorType::InvalidModel,
                            fmt::format("Model stored under: {} does not "
                                        "return "
                                        "a proper "
                                        "torch::device",
                                        Descriptor.getPath()));
    BModel->setDevice(DeviceIVal.toDevice());
  } catch (const c10::Error& EC) {
    return AMS_MAKE_ERROR(AMSErrorType::TorchInternal,
                          fmt::format("Error when setting the model device\n",
                                      EC.what()));
  }

  return AMSExpected<std::unique_ptr<BaseModel>>{std::move(BModel)};
}

InferenceModel::InferenceModel(BaseModel&& BM) : BaseModel(std::move(BM))
{
  getJITModel().eval();
}

AMSExpected<std::unique_ptr<InferenceModel>> InferenceModel::load(
    const AbstractModel& Descriptor)
{
  auto BMExp = BaseModel::load(Descriptor);
  if (!BMExp) return tl::unexpected(BMExp.error());

  return AMSExpected<std::unique_ptr<InferenceModel>>(
      new InferenceModel(std::move(*BMExp.value())));
}

AMSExpected<torch::jit::IValue> InferenceModel::operator()(
    std::vector<torch::jit::IValue> Inputs)
{
  auto& JModel = getJITModel();
  try {
    return AMSExpected<torch::jit::IValue>{JModel.forward(std::move(Inputs))};
  } catch (const c10::Error& EC) {
    return AMS_MAKE_ERROR(AMSErrorType::TorchInternal, EC.what());
  } catch (const torch::jit::ErrorReport& e) {
    return AMS_MAKE_ERROR(AMSErrorType::TorchInternal, e.what());
  } catch (const std::runtime_error& e) {
    return AMS_MAKE_ERROR(AMSErrorType::TorchInternal, e.what());
  } catch (...) {
    return AMS_MAKE_ERROR(AMSErrorType::TorchInternal,
                          "Unknown TorchScript exception");
  }
}
