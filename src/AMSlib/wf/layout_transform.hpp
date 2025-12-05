#pragma once

#include <ATen/ATen.h>
#include <torch/script.h>  // for torch::jit::IValue

#include <optional>

#include "wf/tensor_bundle.hpp"

namespace ams
{

/// Abstract base class describing how AMS transforms application-level data
/// (Inputs, Inouts, Outputs) into contiguous model inputs and vice versa.
///
/// - pack() produces the tensor that is fed into the surrogate model.
/// - unpack() receives the model output (an IValue that may contain multiple
///   tensors) and maps it back into Outputs, Inouts, and optionally Uncertainties.
///
/// The AMS pipeline never assumes any particular layout; all shape and packing
/// logic lives in concrete LayoutTransform implementations.
class LayoutTransform
{
public:
  virtual ~LayoutTransform() = default;

  /// Pack the application-level Inputs and Inouts into a single tensor suitable
  /// for feeding into the ML model.
  virtual at::Tensor pack(const TensorBundle& Inputs,
                          const TensorBundle& Inouts) = 0;

  /// Unpack the model's output (an IValue that may be a tensor or a tuple of
  /// tensors) into:
  ///   - Outputs
  ///   - Inouts
  ///   - Uncertainties (optional)
  ///
  /// Concrete layouts determine how the returned IValue maps back to domain
  /// tensors. Only LayoutTransform knows the correct indexing and shapes.
  virtual void unpack(const torch::jit::IValue& ModelOutput,
                      TensorBundle& Outputs,
                      TensorBundle& Inouts,
                      std::optional<at::Tensor>& Uncertainties) = 0;

  /// Optional descriptive name used for debugging, logging, introspection.
  virtual const char* name() const noexcept = 0;
};

}  // namespace ams
