/*
 * Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
 * AMSLib Project Developers
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */
#pragma once

#include <cstdint>
#include <string>

#include "AMSTensor.hpp"
#include "AMSTypes.hpp"

namespace ams
{

using DomainLambda =
    std::function<void(const ams::SmallVector<ams::AMSTensor>& /*inputs */,
                       ams::SmallVector<ams::AMSTensor>& /*input - outputs */,
                       ams::SmallVector<ams::AMSTensor>& /* outputs */)>;


using DomainCFn = void (*)(void*,
                           const ams::SmallVector<ams::AMSTensor>&,
                           ams::SmallVector<ams::AMSTensor>&,
                           ams::SmallVector<ams::AMSTensor>&);

using AMSExecutor = int64_t;
using AMSCAbstrModel = int;

void AMSInit();
void AMSFinalize();


AMSExecutor AMSCreateExecutor(AMSCAbstrModel model,
                              int process_id,
                              int world_size);

/*
In the past AMSlib internally handled the various kinds of UQ (random, duq(max),
duq(mean)). This required our API to require the user to explicitly set the UQ
type of the model. Further on indirect model registration  (through environment
variables pointing to a JSON file) every entry under the models had to define
all of these values. There were the following shortcomings with the previous approach:

1. The user of the model may not be the producer of the model and thus requiring
from the user to know the UQ may be harder.
2. Every new UQ technology needs to be hardcoded in AMSlib. That limits how fast
we can use new ideas regarding UQ.
3. AMSlib required internally to keep some state and implement conditional
nesting depending on the uq type.

AMS expects models to return a Tuple[[Tensor[N, ...], Tensor[N, 1]]. The first index
of the Tuple is the actual model prediction, whereas the second one should be a
tensor of the shape [N, 1]. Lower values on index "k" indicate low uncertainty,
higher values indicate high uncertainty.

Simple implementations of the current duq(mean), duq(max), random can be found on our
model generation files used in testing.
*/

AMSCAbstrModel AMSRegisterAbstractModel(const char* domain_name,
                                        double threshold,
                                        const char* surrogate_path,
                                        bool store_data = true);

AMSCAbstrModel AMSQueryModel(const char* domain_model);

void AMSExecute(AMSExecutor executor,
                DomainLambda& OrigComputation,
                const ams::SmallVector<ams::AMSTensor>& ins,
                ams::SmallVector<ams::AMSTensor>& inouts,
                ams::SmallVector<ams::AMSTensor>& outs);

void AMSCExecute(AMSExecutor executor,
                 DomainCFn OrigComputation,
                 void* args,
                 const ams::SmallVector<ams::AMSTensor>& ins,
                 ams::SmallVector<ams::AMSTensor>& inouts,
                 ams::SmallVector<ams::AMSTensor>& outs);

void AMSDestroyExecutor(AMSExecutor executor);

void AMSSetAllocator(ams::AMSResourceType resource, const char* alloc_name);
const char* AMSGetAllocatorName(ams::AMSResourceType device);
void AMSConfigureFSDatabase(AMSDBType db_type, const char* db_path);
const std::string AMSGetDatabaseName(AMSExecutor wf);

};  // namespace ams
