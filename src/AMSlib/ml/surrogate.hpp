/*
 * Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
 * AMSLib Project Developers
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef __AMS_SURROGATE_HPP__
#define __AMS_SURROGATE_HPP__

#include <ATen/core/interned_strings.h>
#include <ATen/core/ivalue.h>
#include <torch/cuda.h>
#include <torch/script.h>  // One-stop header.

#include <experimental/filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>

#include "AMS.h"
#include "ArrayRef.hpp"
#include "wf/debug.h"


namespace UQ
{
static inline bool isDeltaUQ(ams::AMSUQPolicy policy)
{
  if (policy >= ams::AMSUQPolicy::AMS_DELTAUQ_MEAN &&
      policy <= ams::AMSUQPolicy::AMS_DELTAUQ_MAX) {
    return true;
  }
  return false;
}

static inline bool isRandomUQ(ams::AMSUQPolicy policy)
{
  return policy == ams::AMSUQPolicy::AMS_RANDOM;
}


static inline bool isUQPolicy(ams::AMSUQPolicy policy)
{
  if (ams::AMSUQPolicy::AMS_UQ_BEGIN < policy &&
      policy < ams::AMSUQPolicy::AMS_UQ_END)
    return true;
  return false;
}

static std::string UQPolicyToStr(ams::AMSUQPolicy policy)
{
  if (policy == ams::AMSUQPolicy::AMS_RANDOM)
    return "random";
  else if (policy == ams::AMSUQPolicy::AMS_DELTAUQ_MEAN)
    return "deltaUQ (mean)";
  else if (policy == ams::AMSUQPolicy::AMS_DELTAUQ_MAX)
    return "deltaUQ (max)";
  return "Unknown";
}

static ams::AMSUQPolicy UQPolicyFromStr(std::string& policy)
{
  if (policy.compare("random") == 0)
    return ams::AMSUQPolicy::AMS_RANDOM;

  else if (policy.compare("deltaUQ (mean)") == 0)
    return ams::AMSUQPolicy::AMS_DELTAUQ_MEAN;
  else if (policy.compare("deltaUQ (max)") == 0)
    return ams::AMSUQPolicy::AMS_DELTAUQ_MAX;
  return ams::AMSUQPolicy::AMS_UQ_END;
}
};  // namespace UQ

//! ----------------------------------------------------------------------------
//! An implementation for a surrogate model
//! ----------------------------------------------------------------------------
class SurrogateModel
{

private:
  const std::string _model_path;
  ams::AMSResourceType model_device;
  torch::DeviceType torch_device;
  ams::AMSDType model_dtype;
  torch::Dtype torch_dtype;
  const bool _is_DeltaUQ;

  // -------------------------------------------------------------------------
  // variables to store the torch model
  // -------------------------------------------------------------------------
  torch::jit::script::Module module;

protected:
  static std::unordered_map<std::string, std::shared_ptr<SurrogateModel>>
      instances;

  SurrogateModel(std::string& model_path, bool is_DeltaUQ = false);

public:
  // -------------------------------------------------------------------------
  // public interface
  // -------------------------------------------------------------------------

  static std::shared_ptr<SurrogateModel> getInstance(std::string& model_path,
                                                     bool is_DeltaUQ = false)
  {
    auto model = SurrogateModel::instances.find(std::string(model_path));
    if (model != instances.end()) {
      // Model Found
      auto torch_model = model->second;

      DBG(Surrogate,
          "Returning existing model represented under (%s)",
          model_path.empty() ? "" : model_path.c_str());
      return torch_model;
    }

    // Model does not exist. We need to create one
    DBG(Surrogate, "Generating new model under (%s)", model_path.c_str());
    std::shared_ptr<SurrogateModel> torch_model =
        std::shared_ptr<SurrogateModel>(
            new SurrogateModel(model_path, is_DeltaUQ));
    instances.insert(std::make_pair(std::string(model_path), torch_model));
    return torch_model;
  };

  ~SurrogateModel()
  {
    DBG(Surrogate, "Destroying surrogate model at %s", _model_path.c_str());
  }

  std::tuple<torch::Tensor, torch::Tensor> _computeDetlaUQ(
      c10::IValue& deltaUQTuple,
      ams::AMSUQPolicy policy,
      float threshold);

  std::tuple<torch::Tensor, torch::Tensor> _evaluate(torch::Tensor& inputs,
                                                     ams::AMSUQPolicy policy,
                                                     const float threshold);

  std::tuple<torch::Tensor, torch::Tensor> evaluate(
      ams::MutableArrayRef<at::Tensor> Inputs,
      ams::AMSUQPolicy policy,
      const float threshold);


  inline bool is_gpu() const
  {
    return model_device == ams::AMSResourceType::AMS_DEVICE;
  }

  inline bool is_cpu() const
  {
    return model_device == ams::AMSResourceType::AMS_HOST;
  }

  inline bool is_resource(ams::AMSResourceType rType) const
  {
    return model_device == rType;
  }

  inline bool is_float() const { return model_dtype == ams::AMS_SINGLE; }
  inline bool is_double() const { return model_dtype == ams::AMS_DOUBLE; }
  inline bool is_type(ams::AMSDType dType) const
  {
    return model_dtype == dType;
  }

  bool is_DeltaUQ() { return _is_DeltaUQ; }

  std::tuple<ams::AMSResourceType, torch::DeviceType> convertModelResourceType(
      std::string& device);
  std::tuple<ams::AMSDType, torch::Dtype> convertModelDataType(
      std::string& type);

  std::tuple<ams::AMSResourceType, torch::DeviceType> getModelResourceType()
      const;
  std::tuple<ams::AMSDType, torch::Dtype> getModelDataType() const;
};

#endif
