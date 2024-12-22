#pragma once
#include <c10/core/DeviceType.h>
#include <torch/torch.h>

#include "AMS.h"

namespace ams
{
class AMSWorkflow;
}

void callApplication(ams::EOSLambda CallBack,
                     ams::MutableArrayRef<torch::Tensor> Ins,
                     ams::MutableArrayRef<torch::Tensor> InOuts,
                     ams::MutableArrayRef<torch::Tensor> Outs);


void callAMS(ams::AMSWorkflow *executor,
             ams::EOSLambda Physics,
             const ams::SmallVector<ams::AMSTensor> &ins,
             ams::SmallVector<ams::AMSTensor> &inouts,
             ams::SmallVector<ams::AMSTensor> &outs);
