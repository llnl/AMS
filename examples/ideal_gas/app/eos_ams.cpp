/*
 * Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
 * AMSLib Project Developers
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "eos_ams.hpp"

#include <SmallVector.hpp>
#include <vector>

using namespace ams;

template <typename FPType>
AMSEOS<FPType>::AMSEOS(const AMSDBType db_type,
                       const AMSResourceType resource,
                       const AMSExecPolicy exec_policy,
                       const int mpi_task,
                       const int mpi_nproc,
                       const double threshold,
                       const char *surrogate_path)
    : res_(resource), IdealGas<FPType>(1.6, 1.4)
{
  AMSCAbstrModel model_descr = AMSRegisterAbstractModel("ideal_gas",
                                                        threshold,
                                                        surrogate_path,
                                                        "ideal_gas");
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
  SmallVector<AMSTensor> inputs;
  inputs.push_back(
      std::move(AMSTensor::view(density, {length, 1}, {1, 1}, res_)));
  inputs.push_back(
      std::move(AMSTensor::view(density, {length, 1}, {1, 1}, res_)));

  SmallVector<AMSTensor> inout;
  SmallVector<AMSTensor> outputs;
  outputs.push_back(
      std::move(AMSTensor::view(pressure, {length, 1}, {1, 1}, res_)));
  outputs.push_back(
      std::move(AMSTensor::view(soundspeed2, {length, 1}, {1, 1}, res_)));
  outputs.push_back(
      std::move(AMSTensor::view(bulkmod, {length, 1}, {1, 1}, res_)));
  outputs.push_back(
      std::move(AMSTensor::view(temperature, {length, 1}, {1, 1}, res_)));

  DomainLambda OrigComputation = [&,
                                  this](const SmallVector<AMSTensor> &ams_ins,
                                        SmallVector<AMSTensor> &ams_inouts,
                                        SmallVector<AMSTensor> &ams_outs) {
    std::cout << "Shape is " << ams_ins[0].shape()[0] << ", "
              << ams_ins[1].shape()[1] << "\n";
    IdealGas<FPType>::Eval(
        ams_ins[0].shape()[0],
        static_cast<const FPType *>(ams_ins[0].data<FPType>()),
        static_cast<const FPType *>(ams_ins[1].data<FPType>()),
        static_cast<FPType *>(ams_outs[0].data<FPType>()),
        static_cast<FPType *>(ams_outs[1].data<FPType>()),
        static_cast<FPType *>(ams_outs[2].data<FPType>()),
        static_cast<FPType *>(ams_outs[3].data<FPType>()));
  };


  AMSExecute(wf_, OrigComputation, inputs, inout, outputs);
}

template class AMSEOS<double>;
template class AMSEOS<float>;
