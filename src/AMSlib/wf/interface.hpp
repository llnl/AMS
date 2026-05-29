#pragma once

#include "AMS.h"
// #include "AMSTensor.hpp"

namespace ams
{
class AMSWorkflow;
}

void callAMS(ams::AMSWorkflow *executor,
             ams::DomainLambda Physics,
             const ams::SmallVector<ams::AMSTensor> &ins,
             ams::SmallVector<ams::AMSTensor> &inouts,
             ams::SmallVector<ams::AMSTensor> &outs);


#if defined(__AMS_ENABLE_TORCH__)

#include <c10/core/DeviceType.h>
#include <torch/torch.h>

void callApplication(ams::DomainLambda CallBack,
                     ams::MutableArrayRef<torch::Tensor> Ins,
                     ams::MutableArrayRef<torch::Tensor> InOuts,
                     ams::MutableArrayRef<torch::Tensor> Outs);

/** @brief Helper to create AMSTensor views from a vector of torch::Tensors.
*  @note The torch::Tensors MUST outlive the returned views.
*/
ams::SmallVector<ams::AMSTensor> torchToAMSTensors(ams::MutableArrayRef<torch::Tensor> tensorVector);
#endif
