/*
 * Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
 * AMSLib Project Developers
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef _AMS_EOS_HPP_
#define _AMS_EOS_HPP_

#include <stdexcept>

#include "AMS.h"
#include "eos_idealgas.hpp"

template <typename FPType>
class AMSEOS : public IdealGas<FPType>
{
  ams::AMSExecutor wf_;
  ams::AMSResourceType res_;

public:
  AMSEOS(const ams::AMSDBType db_type,
         const ams::AMSResourceType resource,
         const ams::AMSExecPolicy exec_policy,
         const int mpi_task,
         const int mpi_nproc,
         const double threshold,
         const char *surrogate_path);

  virtual void Eval(const int length,
                    const FPType *density,
                    const FPType *energy,
                    FPType *pressure,
                    FPType *soundspeed2,
                    FPType *bulkmod,
                    FPType *temperature) const override;
};

#endif  // _AMS_EOS_HPP_
