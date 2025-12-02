#pragma once

#include <fmt/format.h>
#include <torch/script.h>  // torch::jit::Module, torch::Device, torch::Dtype
#include <torch/torch.h>

#include <optional>
#include <type_traits>

#include "AMSError.hpp"

namespace ams
{
namespace ml
{

class AbstractModel;  // Forward declaration, descriptor-only.


/// \brief Owning handle around a TorchScript module with move-only semantics.
///
/// BaseModel owns a `torch::jit::Module` together with its device and dtype.
/// The underlying Torch module has reference-like semantics: copying a
/// `torch::jit::Module` does not clone the model, but creates another handle
/// to the same internal object.
///
/// Because BaseModel provides mutating operations (for example convertTo()),
/// allowing it to be copyable would introduce surprising aliasing:
///
/// \code
/// BaseModel A = ...;
/// BaseModel B = A;          // B aliases the same JIT module as A
/// B.convertTo<float>();     // mutates both A and B
/// \endcode
///
/// To avoid this class of bugs, BaseModel is intentionally non-copyable and
/// move-only. Ownership can be transferred (returned from factories, stored
/// in containers), but accidental copies are disallowed, similar to
/// std::unique_ptr.
class BaseModel
{
public:
  /// \brief Scalar type used by the model weights.
  using DType = torch::ScalarType;

  /// \brief Device type on which the model resides.
  using DeviceType = torch::Device;

  /// \brief An integer identifier for the model
  using HashT = uint64_t;

  /// \brief Load a model from an AbstractModel descriptor using the
  ///        on-disk dtype and device encoded in the model file.
  ///
  /// On success, returns a fully constructed BaseModel instance.
  /// On error, returns an AMSError describing the failure.
  ///
  static AMSExpected<std::unique_ptr<BaseModel>> load(
      const AbstractModel& Descriptor);


  /// \brief Return true if the model is resident on a device (e.g. GPU).
  ///
  /// This is typically equivalent to checking whether the underlying
  /// device is a CUDA or HIP device, as opposed to a CPU device.
  bool isDevice() const { return !ModelDevice.is_cpu(); }

  /// \brief Return the Torch device on which the model currently lives.
  torch::Device getDevice() const { return ModelDevice; }


  /// \brief Return the scalar dtype of the model weights.
  DType getDType() const { return ModelDType; }


  /// \brief Return true if the model dtype matches the given scalar type.
  ///
  /// Example:
  /// \code
  /// if (Model.isType<double>()) { ... }
  /// \endcode
  template <typename ScalarT>
  bool isType() const
  {
    if constexpr (std::is_floating_point_v<ScalarT>) {
      if constexpr (std::is_same_v<ScalarT, float>)
        return ModelDType == c10::ScalarType::Float;
      else if constexpr (std::is_same_v<ScalarT, double>)
        return ModelDType == c10::ScalarType::Double;
      else
        return false;
    } else if constexpr (std::is_integral_v<ScalarT>) {
      constexpr bool is_signed = std::is_signed_v<ScalarT>;
      constexpr size_t bits = sizeof(ScalarT) * 8;

      if constexpr (is_signed) {
        switch (bits) {
          case 8:
            return ModelDType == c10::ScalarType::Char;
          case 16:
            return ModelDType == c10::ScalarType::Short;
          case 32:
            return ModelDType == c10::ScalarType::Int;
          case 64:
            return ModelDType == c10::ScalarType::Long;
        }
      } else {  // unsigned
        switch (bits) {
          case 8:
            return ModelDType == c10::ScalarType::Byte;
        }
      }
      return false;
    } else {
      static_assert(!sizeof(ScalarT), "Unsupported type in isType()");
    }
    return false;
  }


  /// \brief convert the model to the requested device and dtype.
  ///
  /// This is a combined convenience operation that migrates the model
  /// to a target device and scalar type in one step. The function modifies the current object
  template <typename ScalarT>
  AMSStatus convertTo(std::optional<torch::Device> TargetDevice = std::nullopt)
  {
    constexpr auto ReqModelType = toTorchDType<ScalarT>();

    auto ReqDevice = TargetDevice.value_or(ModelDevice);
    // Fast path: nothing to do
    if (ReqModelType == ModelDType && ReqDevice == ModelDevice) return {};

    try {
      JITModel.to(ReqDevice, ReqModelType, /*non_blocking=*/false);
    } catch (const c10::Error& EC) {
      return AMS_MAKE_ERROR(AMSErrorType::TorchInternal,
                            fmt::format("BaseModel::convertTo failed: {}",
                                        EC.what()));
    }

    ModelDevice = ReqDevice;
    ModelDType = ReqModelType;
    return {};
  }

  BaseModel(const BaseModel&) = delete;
  BaseModel& operator=(const BaseModel&) = delete;

  BaseModel(BaseModel&&) = default;
  BaseModel& operator=(BaseModel&&) = default;

protected:
  ///  \brief Construct a BaseModel from an existing Torch module, device, and dtype.
  ///
  ///  This is primarily intended for use by factory functions (e.g. load())
  ///  and subclasses such as InferenceModel.
  BaseModel(const AbstractModel& AModel);

  ///  \brief Mutable access to the underlying Torch module.
  ///
  ///  Intended for subclasses that need to configure the module (e.g. set
  ///  eval mode, attach buffers, etc.).
  ///
  inline torch::jit::Module& getJITModel() { return JITModel; }


  /// \brief Const access to the underlying Torch module.
  inline const torch::jit::Module& getJITModel() const { return JITModel; }

  /// \brief Set the device of the model (does not move/copy it to that device)
  inline void setDevice(torch::Device Device) { this->ModelDevice = Device; }

  /// \brief Set the device of the model (does not move/copy it to that device)
  inline void setDType(DType ModelDType) { this->ModelDType = ModelDType; }


private:
  template <class T>
  struct dependent_false : std::false_type {
  };

  template <typename ScalarT>
  static constexpr DType toTorchDType()
  {
    // Floats
    if constexpr (std::is_floating_point_v<ScalarT>) {
      if constexpr (std::is_same_v<ScalarT, float>)
        return c10::ScalarType::Float;
      else if constexpr (std::is_same_v<ScalarT, double>)
        return c10::ScalarType::Double;
      else
        static_assert(dependent_false<ScalarT>::value,
                      "Unsupported floating-point type in toTorchDType");
    }
    // Integers
    else if constexpr (std::is_integral_v<ScalarT>) {
      constexpr bool is_signed = std::is_signed_v<ScalarT>;
      constexpr size_t bits = sizeof(ScalarT) * 8;

      if constexpr (is_signed) {
        switch (bits) {
          case 8:
            return c10::ScalarType::Char;
          case 16:
            return c10::ScalarType::Short;
          case 32:
            return c10::ScalarType::Int;
          case 64:
            return c10::ScalarType::Long;
          default:
            static_assert(dependent_false<ScalarT>::value,
                          "Unsupported signed integer width in toTorchDType");
        }
      } else {  // unsigned
        switch (bits) {
          case 8:
            return c10::ScalarType::Byte;
          default:
            static_assert(dependent_false<ScalarT>::value,
                          "Unsupported unsigned integer width in toTorchDType");
        }
      }
    } else {
      static_assert(dependent_false<ScalarT>::value,
                    "Unsupported type in toTorchDType");
    }

    // Unreachable, but keeps some compilers happy
    return c10::ScalarType::Undefined;
  }

  /// \brief Underlying Torch module representing the loaded model.
  torch::jit::script::Module JITModel;

  /// \brief Device on which the model currently resides.
  torch::Device ModelDevice;

  /// \brief Scalar dtype of the model weights.
  DType ModelDType;

  /// \brief unique id/hash of the model. Currently unused
  HashT UUID;
};


/// \brief Callable model interface on top of BaseModel.
///
/// InferenceModel does NOT add state; it adds a clean forward() API
/// via operator(), returning an IValue exactly like PyTorch.
///
/// It inherits move semantics from BaseModel and intentionally remains
/// non-copyable.
class InferenceModel : public BaseModel
{
public:
  using BaseModel::BaseModel;

  /// \brief Load model from descriptor into an InferenceModel.
  ///
  /// This wraps BaseModel::load() and then transfers ownership.
  static AMSExpected<std::unique_ptr<InferenceModel>> load(
      const AbstractModel& Descriptor);

  InferenceModel(const InferenceModel&) = delete;
  InferenceModel& operator=(const InferenceModel&) = delete;

  InferenceModel(InferenceModel&&) = default;
  InferenceModel& operator=(InferenceModel&&) = default;

  /// \brief Generic call: accepts a vector<IValue>.
  ///
  /// This is the most TorchScript-native interface.
  AMSExpected<torch::jit::IValue> operator()(
      std::vector<torch::jit::IValue> Inputs);

  /// \brief Convenience: single-tensor forward call.
  AMSExpected<torch::jit::IValue> operator()(const torch::Tensor& X)
  {
    return (*this)(std::vector<torch::jit::IValue>{X});
  }

  /// \brief Variadic convenience interface.
  ///
  /// Allows calls like:
  ///     model(t1, t2, t3)
  ///
  /// Each argument must be convertible to an IValue.
  template <typename... Ts>
  AMSExpected<torch::jit::IValue> operator()(Ts&&... Args)
  {
    std::vector<torch::jit::IValue> V;
    V.reserve(sizeof...(Args));
    (V.emplace_back(std::forward<Ts>(Args)), ...);
    return (*this)(std::move(V));
  }

  /// \brief Returns whether the model is in training mode (should always be false for InferenceModel).
  bool isTraining() const { return getJITModel().is_training(); }


private:
  /// \brief Construct from an existing BaseModel via move.
  ///
  /// Private because construction should go through load().
  explicit InferenceModel(BaseModel&& BM);
};

}  // namespace ml
}  // namespace ams
