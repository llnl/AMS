#pragma once
#include <c10/core/DeviceType.h>
#include <torch/torch.h>

#include "AMS.h"

namespace ams
{
class AMSWorkflow;
}

void callApplication(ams::DomainLambda CallBack,
                     ams::MutableArrayRef<torch::Tensor> Ins,
                     ams::MutableArrayRef<torch::Tensor> InOuts,
                     ams::MutableArrayRef<torch::Tensor> Outs);

void callApplication(ams::HomogeneousGraphDomainFn CallBack,
                     const ams::AMSHomogeneousGraph& graph,
                     ams::SmallVector<ams::AMSTensor>& outs);

void callApplication(ams::HeterogeneousGraphDomainFn CallBack,
                     const ams::AMSHeterogeneousGraph& graph,
                     ams::SmallVector<ams::AMSTensor>& outs);


void callAMS(ams::AMSWorkflow *executor,
             ams::DomainLambda Physics,
             const ams::SmallVector<ams::AMSTensor> &ins,
             ams::SmallVector<ams::AMSTensor> &inouts,
             ams::SmallVector<ams::AMSTensor> &outs);

void callAMS(ams::AMSWorkflow* executor,
             ams::HomogeneousGraphDomainFn Physics,
             const ams::AMSHomogeneousGraph& graph_input,
             ams::SmallVector<ams::AMSTensor>& outs);

void callAMS(ams::AMSWorkflow* executor,
             ams::HeterogeneousGraphDomainFn Physics,
             const ams::AMSHeterogeneousGraph& graph_input,
             ams::SmallVector<ams::AMSTensor>& outs);
