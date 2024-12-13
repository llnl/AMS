/*
 * Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
 * AMSLib Project Developers
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef __AMS_SURROGATE_HPP__
#define __AMS_SURROGATE_HPP__

#include <experimental/filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>

#include "AMS.h"
#include "util/ArrayRef.hpp"
//#include "wf/device.hpp"

#include <ATen/core/interned_strings.h>
#include <ATen/core/ivalue.h>
#include <torch/cuda.h>
#include <torch/script.h>  // One-stop header.

#include "wf/data_handler.hpp"
#include "wf/debug.h"

//! ----------------------------------------------------------------------------
//! An implementation for a surrogate model
//! ----------------------------------------------------------------------------
class SurrogateModel
{

private:
  const std::string _model_path;
  AMSResourceType model_device;
  torch::DeviceType torch_device;
  AMSDType model_dtype;
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
      AMSUQPolicy policy,
      float threshold);

  std::tuple<torch::Tensor, torch::Tensor> _evaluate(torch::Tensor& inputs,
                                                     AMSUQPolicy policy,
                                                     const float threshold);

  std::tuple<torch::Tensor, torch::Tensor> evaluate(
      ams::MutableArrayRef<at::Tensor> Inputs,
      AMSUQPolicy policy,
      const float threshold);


  inline bool is_gpu() const
  {
    return model_device == AMSResourceType::AMS_DEVICE;
  }

  inline bool is_cpu() const
  {
    return model_device == AMSResourceType::AMS_HOST;
  }

  inline bool is_resource(AMSResourceType rType) const
  {
    return model_device == rType;
  }

  inline bool is_float() const { return model_dtype == AMS_SINGLE; }
  inline bool is_double() const { return model_dtype == AMS_DOUBLE; }
  inline bool is_type(AMSDType dType) const { return model_dtype == dType; }


  //
  bool is_DeltaUQ() { return _is_DeltaUQ; }
  //
  //  void update(const std::string& new_path)
  //  {
  //    /* This function updates the underlying torch model,
  //     * with a new one pointed at location modelPath. The previous
  //     * one is destructed automatically.
  //     *
  //     * TODO: I decided to not update the model path on the ``instances''
  //     * map. As we currently expect this change will be agnostic to the application
  //     * user. But, in any case we should keep track of which model has been used at which
  //     * invocation. This is currently not done.
  //     */
  //    //if (model_device != AMSResourceType::AMS_DEVICE)
  //    //  _load<TypeInValue>(new_path, "cpu");
  //    //else
  //    //  _load<TypeInValue>(new_path, "cuda");
  //  }

  //  AMSResourceType getModelResource() const { return model_device; }
  std::tuple<AMSResourceType, torch::DeviceType> getModelResourceType();
  std::tuple<AMSDType, torch::Dtype> getModelDataType();
};

#endif
