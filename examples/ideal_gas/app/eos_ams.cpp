/*
 * Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
 * AMSLib Project Developers
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include <SmallVector.hpp>
#include <vector>

#include "eos_ams.hpp"

using namespace ams;

template <typename FPType>
AMSEOS<FPType>::AMSEOS(const AMSDBType db_type,
                       const AMSResourceType resource,
                       const AMSExecPolicy exec_policy,
                       const AMSUQPolicy uq_policy,
                       const int mpi_task,
                       const int mpi_nproc,
                       const double threshold,
                       const char *surrogate_path)
    : res_(resource), IdealGas<FPType>(1.6, 1.4)
{
  AMSCAbstrModel model_descr = ams::AMSRegisterAbstractModel(
      "ideal_gas", uq_policy, threshold, surrogate_path, "ideal_gas");
  wf_ = AMSCreateExecutor(model_descr, mpi_task, mpi_nproc);
}

template <typename FPType>
#ifdef __ENABLE_PERFFLOWASPECT__
__attribute__((annotate("@critical_path(pointcut='around')")))
#endif
void AMSEOS<FPType>::Eval(const int length,
                          const FPType *density,
                          const FPType *energy,
                          FPType *pressure,
                          FPType *soundspeed2,
                          FPType *bulkmod,
                          FPType *temperature) const
{
  ams::SmallVector<ams::AMSTensor> inputs = {
      ams::AMSTensor::view(density, {length, 1}, {1, 1}, res_),
      ams::AMSTensor::view(density, {length, 1}, {1, 1}, res_),
  };

  ams::SmallVector<ams::AMSTensor> inout;
  ams::SmallVector<ams::AMSTensor> outputs = {
      ams::AMSTensor::view(pressure, {length, 1}, {1, 1}, res_),
      ams::AMSTensor::view(soundspeed2, {length, 1}, {1, 1}, res_),
      ams::AMSTensor::view(bulkmod, {length, 1}, {1, 1}, res_),
      ams::AMSTensor::view(temperature, {length, 1}, {1, 1}, res_),
  };

  EOSLambda OrigComputation =
      [&](const ams::SmallVector<ams::AMSTensor> &ams_ins,
          ams::SmallVector<ams::AMSTensor> &ams_inouts,
          ams::SmallVector<ams::AMSTensor> &ams_outs) {
        IdealGas<FPType>::Eval(
            ams_ins[0].shape()[1],
            static_cast<const FPType *>(ams_ins[0].data<FPType>()),
            static_cast<const FPType *>(inputs[1].data<FPType>()),
            static_cast<FPType *>(ams_outs[0].data<FPType>()),
            static_cast<FPType *>(ams_outs[1].data<FPType>()),
            static_cast<FPType *>(ams_outs[2].data<FPType>()),
            static_cast<FPType *>(ams_outs[3].data<FPType>()));
      };


  AMSExecute(wf_, OrigComputation, inputs, inout, outputs);
}

template class AMSEOS<double>;
template class AMSEOS<float>;
