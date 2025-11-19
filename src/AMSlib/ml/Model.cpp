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

AMSExpected<BaseModel> BaseModel::load(const AbstractModel& Descriptor)
{
  std::error_code EC;
  if (!fs::exists(Descriptor.getPath(), EC))
    return AMS_MAKE_ERROR(AMSErrorType::FileDoesNotExist, EC.message());

  AMSExpected<BaseModel> BModelOrErr = [&]() -> AMSExpected<BaseModel> {
    try {
      return BaseModel(Descriptor);
    } catch (const c10::Error& EC) {
      // Here I am using a more verbose error message (that will inlclude stack frames and line info of the internal torch library).
      // These tend to be useful to debug.
      return AMS_MAKE_ERROR(AMSErrorType::TorchInternal, EC.what());
    }
  }();

  if (!BModelOrErr) return BModelOrErr;

  auto BModel = std::move(*BModelOrErr);

  auto& Module = BModel.getJITModel();
  auto AMSModelInfoMethod = Module.find_method("get_ams_info");
  if (!AMSModelInfoMethod) {
    return AMS_MAKE_ERROR(AMSErrorType::InvalidModel,
                          fmt::format("Model store under: {} is not JIT-ed and "
                                      "stored through AMS infrastructure",
                                      Descriptor.getPath()));
  }

  auto ams_info = Module.run_method("get_ams_info");
  auto AMSModelDict = ams_info.toGenericDict();
  for (const auto& Item : AMSModelDict) {
    const auto& Key = Item.key().toStringRef();
    if (Key == "ams_type") {
      BModel.setDType(Item.value().toScalarType());
    } else if (Key == "ams_device") {
      if (Item.value().isDevice()) {
        BModel.setDevice(Item.value().toDevice());
      } else {
        return AMS_MAKE_ERROR(AMSErrorType::InvalidModel,
                              fmt::format("Cannot infer device type of Model "
                                          "stored under: {}"
                                          "the model is likely nnot store "
                                          "using AMS infrastructure",
                                          Descriptor.getPath()));
      }
    } else {
      return AMS_MAKE_ERROR(AMSErrorType::InvalidModel,
                            fmt::format("Unrecognized key entry '{}' under "
                                        "'get_ams_info' in model : {}",
                                        Key,
                                        Descriptor.getPath()));
    }
  }
  return BModel;
}
