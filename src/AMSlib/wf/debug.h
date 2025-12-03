/*
 * Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
 * AMSLib Project Developers
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#pragma once

#include <atomic>
#include <mutex>

#include "wf/logger.hpp"

void memUsage(double& vm_usage, double& resident_set);

#define GETNAME2(name) #name
#define GETNAME(name) GETNAME2(name)

#define AMSPRINT(id, condition, lvl, ...)          \
  do {                                             \
    if (condition) {                               \
      auto& logger = ams::Logger::get();           \
      if (logger.isEnabled(lvl)) {                 \
        logger.log(lvl, GETNAME(id), __VA_ARGS__); \
      }                                            \
    }                                              \
  } while (0);

#define AMS_CFATAL(id, condition, ...)                              \
  do {                                                              \
    if (condition) {                                                \
      auto& logger = ams::Logger::get();                            \
      if (logger.isEnabled(ams::LogLevel::Error)) {                 \
        logger.log(ams::LogLevel::Error, GETNAME(id), __VA_ARGS__); \
      }                                                             \
      abort();                                                      \
    }                                                               \
  } while (0);

#define AMS_FATAL(id, ...) AMS_CFATAL(id, true, __VA_ARGS__)

#define THROW(exception, msg) \
  AMS_FATAL(Throw, "{} {} {}", __FILE__, std::to_string(__LINE__).c_str(), msg)

#ifdef LIBAMS_VERBOSE

#define AMS_CWARNING(id, condition, ...) \
  AMSPRINT(id, condition, ams::LogLevel::Warning, __VA_ARGS__)

#define AMS_WARNING(id, ...) AMS_CWARNING(id, true, __VA_ARGS__)

#define AMS_CINFO(id, condition, ...) \
  AMSPRINT(id, condition, ams::LogLevel::Info, __VA_ARGS__)

#define AMS_INFO(id, ...) AMS_CINFO(id, true, __VA_ARGS__)

#define AMS_CDEBUG(id, condition, ...) \
  AMSPRINT(id, condition, ams::LogLevel::Debug, __VA_ARGS__)

#define AMS_DBG(id, ...) AMS_CDEBUG(id, true, __VA_ARGS__)

// clang-format off
#define REPORT_MEM_USAGE(id, phase)                                    \
  do {                                                                 \
    double vm, rs;                                                     \
    size_t watermark, current_size, actual_size;                       \
    auto& rm = ams::ResourceManager::getInstance();                    \
    memUsage(vm, rs);                                                  \
    AMS_DBG(MEM, "Memory usage at {} is VM:{} RS:{}", phase, vm, rs); \
                                                                       \
    for (int i = 0; i < AMSResourceType::AMS_RSEND; i++) {             \
      if (rm.isActive((AMSResourceType)i)) {                           \
        rm.getAllocatorStats((AMSResourceType)i,                       \
                             watermark,                                \
                             current_size,                             \
                             actual_size);                             \
        AMS_DBG(MEM,                                                        \
              "Allocator:{} HWM:{} CS:{} AS:{}) ",                 \
              rm.getAllocatorName((AMSResourceType)i),         \
              watermark,                                               \
              current_size,                                            \
              actual_size);                                            \
      }                                                                \
    }                                                                  \
  } while (0);

// clang-format on

#else  // LIBAMS_VERBOSE is disabled
#define AMS_CWARNING(id, condition, ...)

#define AMS_WARNING(id, ...)

#define AMS_CINFO(id, condition, ...)

#define AMS_INFO(id, ...)

#define AMS_CDEBUG(id, condition, ...)

#define AMS_DBG(id, ...)


#endif  // LIBAMS_VERBOSE
//

#if defined(__AMS_ENABLE_HIP__)

#include <hip/hip_runtime.h>
#define hipErrCheck(CALL)                 \
  {                                       \
    hipError_t err = CALL;                \
    if (err != hipSuccess) {              \
      AMS_FATAL("ERROR @ {}:{} ->  {}\n", \
                __FILE__,                 \
                __LINE__,                 \
                hipGetErrorString(err));  \
      abort();                            \
    }                                     \
  }

#elif defined(__AMS_ENABLE_CUDA__)
#include <cuda.h>
#include <cuda_runtime.h>

#define cudaErrCheck(CALL)                   \
  {                                          \
    cudaError_t err = CALL;                  \
    if (err != cudaSuccess) {                \
      AMS_FATAL("ERROR @ {}:{} ->  {}:{}\n", \
                __FILE__,                    \
                __LINE__,                    \
                cudaGetErrorName(err),       \
                cudaGetErrorString(err));    \
    }                                        \
  }
#endif

#ifdef __AMS_DEBUG__
constexpr bool amsDebug() { return true; }
#else
constexpr bool amsDebug() { return false; }
#endif
