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
// NOTE: AFAIK torch does not provide the respective hip.h, and we should not guard here.
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
  // -------------------------------------------------------------------------
  // variables to store the torch model
  // -------------------------------------------------------------------------
  torch::jit::script::Module module;

protected:
  static std::unordered_map<std::string, std::shared_ptr<SurrogateModel>>
      instances;

  SurrogateModel(std::string& model_path);

public:
  // -------------------------------------------------------------------------
  // public interface
  // -------------------------------------------------------------------------

  static std::shared_ptr<SurrogateModel> getInstance(std::string& model_path)
  {
    auto model = SurrogateModel::instances.find(std::string(model_path));
    if (model != instances.end()) {
      // Model Found
      auto torch_model = model->second;

      AMS_DBG(Surrogate,
              "Returning existing model represented under: '{}'",
              model_path);
      return torch_model;
    }

    // Model does not exist. We need to create one
    AMS_DBG(Surrogate, "Generating new model under '{}'", model_path);
    std::shared_ptr<SurrogateModel> torch_model =
        std::shared_ptr<SurrogateModel>(new SurrogateModel(model_path));
    instances.insert(std::make_pair(std::string(model_path), torch_model));
    return torch_model;
  };

  ~SurrogateModel()
  {
    AMS_DBG(Surrogate, "Destroying surrogate model at '{}'", _model_path);
  }

  std::tuple<torch::Tensor, torch::Tensor> _evaluate(torch::Tensor& inputs,
                                                     const float threshold);

  std::tuple<torch::Tensor, torch::Tensor> evaluate(
      ams::MutableArrayRef<at::Tensor> Inputs,
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

  std::tuple<ams::AMSResourceType, torch::DeviceType> convertModelResourceType(
      std::string& device);
  std::tuple<ams::AMSDType, torch::Dtype> convertModelDataType(
      std::string& type);

  std::tuple<ams::AMSResourceType, torch::DeviceType> getModelResourceType()
      const;
  std::tuple<ams::AMSDType, torch::Dtype> getModelDataType() const;
};

#endif
