/*
 * Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
 * AMSLib Project Developers
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include <cstdlib>
#include <cstring>

#ifdef __AMS_ENABLE_CUDA__
#include <cuda_runtime.h>
#elif defined(__AMS_ENABLE_HIP__)
#include <hip/hip_runtime.h>
#endif

#include "debug.h"
#include "resource_manager.hpp"

namespace ams
{

template <typename T>
static T roundUp(T num_to_round, int multiple)
{
  return ((num_to_round + multiple - 1) / multiple) * multiple;
}

const std::string AMSAllocator::getName() const { return name; }


struct AMSDefaultDeviceAllocator final : AMSAllocator {
  AMSDefaultDeviceAllocator(std::string name) : AMSAllocator(name) {};
  ~AMSDefaultDeviceAllocator()
  {
    AMS_DBG(AMSDefaultDeviceAllocator, "Destroying default device allocator");
  };

  void* allocate(size_t num_bytes, size_t alignment)
  {
#if defined(__AMS_ENABLE_CUDA__)
    void* devPtr;
    cudaErrCheck(cudaMalloc(&devPtr, num_bytes));
    return devPtr;
#elif defined(__AMS_ENABLE_HIP__)
    void* devPtr;
    hipErrCheck(hipMalloc(&devPtr, num_bytes));
    return devPtr;
#else
    return nullptr;
#endif
  }

  void deallocate(void* ptr)
  {
#if defined(__AMS_ENABLE_CUDA__)
    cudaErrCheck(cudaFree(ptr));
#elif defined(__AMS_ENABLE_HIP__)
    hipErrCheck(hipFree(ptr));
#endif
  }
};

struct AMSDefaultHostAllocator final : AMSAllocator {
  AMSDefaultHostAllocator(std::string name) : AMSAllocator(name) {}
  ~AMSDefaultHostAllocator()
  {
    AMS_DBG(AMSDefaultDeviceAllocator, "Destroying default host allocator");
  }

  void* allocate(size_t num_bytes, size_t alignment)
  {
    return aligned_alloc(alignment, roundUp(num_bytes, alignment));
  }

  void deallocate(void* ptr) { free(ptr); }
};

struct AMSDefaultPinnedAllocator final : AMSAllocator {
  AMSDefaultPinnedAllocator(std::string name) : AMSAllocator(name) {}
  ~AMSDefaultPinnedAllocator() = default;

  void* allocate(size_t num_bytes, size_t alignment)
  {
#if defined(__AMS_ENABLE_CUDA__)
    void* ptr;
    cudaErrCheck(cudaHostAlloc(&ptr, num_bytes, cudaHostAllocPortable));
    return ptr;
#elif defined(__AMS_ENABLE_HIP__)
    void* ptr;
    hipErrCheck(hipHostAlloc(&ptr, num_bytes, hipHostAllocPortable));
    return ptr;
#else
    return nullptr;
#endif
  }

  void deallocate(void* ptr)
  {
#if defined(__AMS_ENABLE_CUDA__)
    cudaErrCheck(cudaFreeHost(ptr));
#elif defined(__AMS_ENABLE_HIP__)
    hipErrCheck(hipFreeHost(ptr));
#endif
  }
};


namespace internal
{
void _raw_copy(void* src,
               AMSResourceType src_dev,
               void* dest,
               AMSResourceType dest_dev,
               size_t num_bytes)
{
  switch (src_dev) {
    case AMSResourceType::AMS_HOST:
    case AMSResourceType::AMS_PINNED:
      switch (dest_dev) {
        case AMSResourceType::AMS_HOST:
        case AMSResourceType::AMS_PINNED:
          std::memcpy(dest, src, num_bytes);
          break;
        case AMSResourceType::AMS_DEVICE:
#if defined(__AMS_ENABLE_CUDA__)
          cudaErrCheck(
              cudaMemcpy(dest, src, num_bytes, cudaMemcpyHostToDevice));
#elif defined(__AMS_ENABLE_HIP__)
          hipErrCheck(hipMemcpy(dest, src, num_bytes, hipMemcpyHostToDevice));
#endif
          break;
        default:
          AMS_FATAL(ResourceManager,
                    "Unknown device type to copy to from HOST");
          break;
      }
      break;
#if defined(__AMS_ENABLE_CUDA__)
    case AMSResourceType::AMS_DEVICE:
      switch (dest_dev) {
        case AMSResourceType::AMS_DEVICE:
          cudaErrCheck(
              cudaMemcpy(dest, src, num_bytes, cudaMemcpyDeviceToDevice));
          break;
        case AMSResourceType::AMS_HOST:
        case AMSResourceType::AMS_PINNED:
          cudaErrCheck(
              cudaMemcpy(dest, src, num_bytes, cudaMemcpyDeviceToHost));
          break;
        default:
          AMS_FATAL(ResourceManager,
                    "Unknown device type to copy to from DEVICE");
          break;
      }
#elif defined(__AMS_ENABLE_HIP__)
    case AMSResourceType::AMS_DEVICE:
      switch (dest_dev) {
        case AMSResourceType::AMS_DEVICE:
          hipErrCheck(hipMemcpy(dest, src, num_bytes, hipMemcpyDeviceToDevice));
          break;
        case AMSResourceType::AMS_HOST:
        case AMSResourceType::AMS_PINNED:
          hipErrCheck(hipMemcpy(dest, src, num_bytes, hipMemcpyDeviceToHost));
          break;
        default:
          AMS_FATAL(ResourceManager,
                    "Unknown device type to copy to from DEVICE");
          break;
      }

#endif
      break;
    default:
      AMS_FATAL(ResourceManager, "Unknown device type to copy from");
  }
}

AMSAllocator* _get_allocator(std::string& alloc_name, AMSResourceType resource)
{
  switch (resource) {
    case AMSResourceType::AMS_DEVICE:
      return new AMSDefaultDeviceAllocator(alloc_name);
      break;
    case AMSResourceType::AMS_HOST:
      return new AMSDefaultHostAllocator(alloc_name);
      break;
    case AMSResourceType::AMS_PINNED:
      return new AMSDefaultPinnedAllocator(alloc_name);
      break;
    default:
      AMS_FATAL(ResourceManager,
                "Unknown resource type to create an allocator for");
  }
}

void _release_allocator(AMSAllocator* allocator) { delete allocator; }

}  // namespace internal
}  // namespace ams
