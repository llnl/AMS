/*
 * Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
 * AMSLib Project Developers
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef __AMS_UTILS_HPP__
#define __AMS_UTILS_HPP__

#include <ATen/core/TensorBody.h>

#include <algorithm>
#include <array>
#include <iostream>
#include <random>
#include <vector>

#include "AMS.h"
#include "SmallVector.hpp"

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

#if __cplusplus < 201402L
template <bool B, typename T = void>
using enable_if_t = typename std::enable_if<B, T>::type;
#else
#endif

template <typename T>
class isDouble
{
public:
  static constexpr bool default_value() { return false; }
};

template <>
class isDouble<double>
{
public:
  static constexpr bool default_value() { return true; }
};

template <>
class isDouble<float>
{
public:
  static constexpr bool default_value() { return false; }
};

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

template <typename T>
inline bool is_real_equal(T l, T r)
{
  return r == std::nextafter(l, r);
}


static inline size_t dtype_to_size(ams::AMSDType dType)
{
  switch (dType) {
    case ams::AMSDType::AMS_DOUBLE:
      return sizeof(double);
    case ams::AMSDType::AMS_SINGLE:
      return sizeof(float);
    default:
      throw std::runtime_error("Requesting the size of unknown object");
  }
}

static inline std::string shapeToString(const at::Tensor& tensor)
{
  std::ostringstream oss;
  oss << tensor.sizes();
  return oss.str();
}

namespace ams
{
namespace tensor
{
SmallVector<at::Tensor> maskTensor(at::Tensor& Src, at::Tensor& Mask);
}  // namespace tensor
}  // namespace ams

#endif
