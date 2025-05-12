/*
 * Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
 * AMSLib Project Developers
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */
#pragma once

#include <cstdint>

#include "AMSTensor.hpp"
#include "AMSTypes.hpp"

namespace ams
{

using EOSLambda =
    std::function<void(const ams::SmallVector<ams::AMSTensor> & /*inputs */,
                       ams::SmallVector<ams::AMSTensor> & /*input - outputs */,
                       ams::SmallVector<ams::AMSTensor> & /* outputs */)>;


using EOSCFn = void (*)(void *,
                        const ams::SmallVector<ams::AMSTensor> &,
                        ams::SmallVector<ams::AMSTensor> &,
                        ams::SmallVector<ams::AMSTensor> &);

using AMSExecutor = int64_t;
using AMSCAbstrModel = int;

void AMSInit();
void AMSFinalize();


AMSExecutor AMSCreateExecutor(AMSCAbstrModel model,
                              int process_id,
                              int world_size);

AMSCAbstrModel AMSRegisterAbstractModel(const char *domain_name,
                                        AMSUQPolicy uq_policy,
                                        double threshold,
                                        const char *surrogate_path,
                                        bool store_data = true);

AMSCAbstrModel AMSQueryModel(const char *domain_model);

void AMSExecute(AMSExecutor executor,
                EOSLambda &OrigComputation,
                const ams::SmallVector<ams::AMSTensor> &ins,
                ams::SmallVector<ams::AMSTensor> &inouts,
                ams::SmallVector<ams::AMSTensor> &outs);

void AMSCExecute(AMSExecutor executor,
                 EOSCFn OrigComputation,
                 void *args,
                 const ams::SmallVector<ams::AMSTensor> &ins,
                 ams::SmallVector<ams::AMSTensor> &inouts,
                 ams::SmallVector<ams::AMSTensor> &outs);

void AMSDestroyExecutor(AMSExecutor executor);

void AMSSetAllocator(ams::AMSResourceType resource, const char *alloc_name);
const char *AMSGetAllocatorName(ams::AMSResourceType device);
void AMSConfigureFSDatabase(AMSDBType db_type, const char *db_path);

};  // namespace ams
