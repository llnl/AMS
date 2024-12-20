#include <c10/core/DeviceType.h>
#include <torch/torch.h>

#include "AMS.h"


void callApplication(EOSLambda CallBack,
                     ams::MutableArrayRef<torch::Tensor> Ins,
                     ams::MutableArrayRef<torch::Tensor> InOuts,
                     ams::MutableArrayRef<torch::Tensor> Outs);
