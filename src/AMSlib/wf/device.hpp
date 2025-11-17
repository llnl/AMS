/*
 * Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
 * AMSLib Project Developers
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef __AMS_DEVICE_HPP__
#define __AMS_DEVICE_HPP__

#include <cstddef>
#include <cstring>
#include <iostream>
#include <stdexcept>

#include "AMS.h"
#include "wf/debug.h"

#define UNDEFINED_FUNC -1

#if defined(__AMS_ENABLE_CUDA__) || defined(__AMS_ENABLE_HIP__)
namespace ams
{
void DtoDMemcpy(void* dest, void* src, size_t nBytes);

void HtoHMemcpy(void* dest, void* src, size_t nBytes);

void HtoDMemcpy(void* dest, void* src, size_t nBytes);

void DtoHMemcpy(void* dest, void* src, size_t nBytes);

void* DeviceAllocate(size_t nBytes);

void DeviceFree(void* ptr);

void* DevicePinnedAlloc(size_t nBytes);

void DeviceFreePinned(void* ptr);

void deviceCheckErrors(const char* file, int line);

void device_random_uq(int seed,
                      bool* uq_flags,
                      int ndata,
                      double acceptable_error);

}  // namespace ams

#endif
#endif
