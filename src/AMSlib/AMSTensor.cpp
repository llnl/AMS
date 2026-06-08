#include "AMSTensor.hpp"

#include <stdexcept>

#include "AMS.h"
#include "ArrayRef.hpp"
#include "SmallVector.hpp"
#include "include/AMSTensor.hpp"
#include "wf/resource_manager.hpp"
#include "wf/utils.hpp"

using namespace ams;

/**
   * @brief Computes the number of elements in the tensor given its shape.
   * @param[in] shapes The shape of the tensor as an array reference.
   * @return The total number of elements in the tensor.
   */
template <typename T>
static inline AMSTensor::IntDimType computeNumElements(ams::ArrayRef<T> shapes)
{
  return std::accumulate(shapes.begin(),
                         shapes.end(),
                         1,
                         std::multiplies<AMSTensor::IntDimType>());
}
// Helper function to check if the tensor is contiguous in memory
bool AMSTensor::isContiguous(AMSTensor::IntDimType expected_stride) const
{
  for (int i = _shape.size() - 1; i >= 0; --i) {
    if (_strides[i] != expected_stride) return false;
    expected_stride *= _shape[i];
  }
  return true;
}

namespace
{
template <typename T>
constexpr AMSDType scalar_to_ams_dtype()
{
  using U = std::remove_cv_t<T>;
  if constexpr (std::is_same_v<U, float>) {
    return AMS_SINGLE;
  } else if constexpr (std::is_same_v<U, double>) {
    return AMS_DOUBLE;
  } else if constexpr (std::is_same_v<U, int32_t>) {
    return AMS_INT32;
  } else if constexpr (std::is_same_v<U, int64_t>) {
    return AMS_INT64;
  } else {
    static_assert(!sizeof(T), "Unsupported AMS scalar type");
  }
}
}  // namespace

AMSTensor::AMSTensor(uint8_t* data,
                     ams::ArrayRef<AMSTensor::IntDimType> shapes,
                     ams::ArrayRef<AMSTensor::IntDimType> strides,
                     AMSDType dType,
                     AMSResourceType location,
                     bool view)
    : _data(data),
      _element_size(dtype_to_size(dType)),
      _shape(shapes),
      _strides(strides),
      _dType(dType),
      _location(location),
      _owned(!view)
{
  _elements = computeNumElements(shapes);
  _bytes = _elements * _element_size;
  _contiguous = isContiguous(1);
  if (!_data) {
    throw std::runtime_error("Generating tensor with Null Pointer AMSTensor.");
  }
}

template <typename ScalarType>
AMSTensor AMSTensor::create(ams::ArrayRef<AMSTensor::IntDimType> shapes,
                            ams::ArrayRef<AMSTensor::IntDimType> strides,
                            AMSResourceType location)
{
  auto numElements = computeNumElements(shapes);
  auto& rm = ams::ResourceManager::getInstance();
  using U = std::remove_cv_t<ScalarType>;
  U* data = rm.allocate<U>(numElements, location, sizeof(U));
  return AMSTensor(reinterpret_cast<uint8_t*>(data),
                   shapes,
                   strides,
                   scalar_to_ams_dtype<U>(),
                   location);
}


template <typename ScalarType>
AMSTensor AMSTensor::view(ScalarType* data,
                          ams::ArrayRef<AMSTensor::IntDimType> shapes,
                          ams::ArrayRef<AMSTensor::IntDimType> strides,
                          AMSResourceType location)
{
  using U = std::remove_cv_t<ScalarType>;
  return AMSTensor(reinterpret_cast<uint8_t*>(const_cast<U*>(data)),
                   shapes,
                   strides,
                   scalar_to_ams_dtype<U>(),
                   location,
                   true);
}

AMSTensor AMSTensor::view(AMSTensor& tensor)
{
  if (tensor._dType == AMS_DOUBLE)
    return AMSTensor::view((double*)tensor._data,
                           tensor._shape,
                           tensor._strides,
                           tensor._location);
  else if (tensor._dType == AMS_SINGLE)
    return AMSTensor::view((float*)tensor._data,
                           tensor._shape,
                           tensor._strides,
                           tensor._location);
  else if (tensor._dType == AMS_INT32)
    return AMSTensor::view((int32_t*)tensor._data,
                           tensor._shape,
                           tensor._strides,
                           tensor._location);
  else if (tensor._dType == AMS_INT64)
    return AMSTensor::view((int64_t*)tensor._data,
                           tensor._shape,
                           tensor._strides,
                           tensor._location);
  throw std::runtime_error(
      "Creating view through copying constructor has incorrect dtype");
}

AMSTensor::~AMSTensor()
{
  // Only release whenwe own the pointer
  if (_owned && _data) {
    auto& rm = ams::ResourceManager::getInstance();
    rm.deallocate(_data, _location);
    _data = nullptr;
    _owned = false;
  }
}

AMSTensor::AMSTensor(AMSTensor&& other) noexcept
    : _data(other._data),
      _elements(other._elements),
      _element_size(other._element_size),
      _shape(std::move(other._shape)),
      _strides(std::move(other._strides)),
      _dType(other._dType),
      _location(other._location),
      _owned(other._owned),
      _contiguous(other._contiguous),
      _bytes(other._bytes)
{
  other._data = nullptr;
  other._owned = false;
}

AMSTensor& AMSTensor::operator=(AMSTensor&& other) noexcept
{
  if (this != &other) {
    if (_owned && _data) {
      auto& rm = ams::ResourceManager::getInstance();
      rm.deallocate(_data, _location);
    }

    // Steal resources from `other`
    _data = other._data;
    _elements = other._elements;
    _element_size = other._element_size;
    _shape = std::move(other._shape);
    _strides = std::move(other._strides);
    _dType = other._dType;
    _location = other._location;
    _owned = other._owned;
    _contiguous = other._contiguous;
    _bytes = other._bytes;

    other._data = nullptr;
    other._owned = false;
  }
  return *this;
}

AMSTensor AMSTensor::transpose(AMSTensor::IntDimType axis1,
                               AMSTensor::IntDimType axis2) const
{
  // Ensure the axes are within bounds
  if (axis1 >= _shape.size() || axis2 >= _shape.size()) {
    throw std::out_of_range("Transpose axes are out of bounds");
  }

  // Create new shape and strides for the transposed tensor
  auto newShape = _shape;
  auto newStrides = _strides;

  // Swap the specified axes in both shape and strides
  std::swap(newShape[axis1], newShape[axis2]);
  std::swap(newStrides[axis1], newStrides[axis2]);

  // Create a new tensor with the same data, new shape, and strides
  if (dType() == AMSDType::AMS_DOUBLE)
    return view((double*)_data, newShape, newStrides, _location);
  else if (dType() == AMSDType::AMS_SINGLE)
    return view((float*)_data, newShape, newStrides, _location);
  else if (dType() == AMSDType::AMS_INT32)
    return view((int32_t*)_data, newShape, newStrides, _location);
  else if (dType() == AMSDType::AMS_INT64)
    return view((int64_t*)_data, newShape, newStrides, _location);
  // NOTE: Use defensive programming here and just crash. We can fix a better interface later
  // for error handling.
  throw std::runtime_error("Unknow data type in transpose\n");
}

template AMSTensor AMSTensor::create<float>(ams::ArrayRef<IntDimType>,
                                            ams::ArrayRef<IntDimType>,
                                            AMSResourceType);
template AMSTensor AMSTensor::create<double>(ams::ArrayRef<IntDimType>,
                                             ams::ArrayRef<IntDimType>,
                                             AMSResourceType);
template AMSTensor AMSTensor::create<int32_t>(ams::ArrayRef<IntDimType>,
                                              ams::ArrayRef<IntDimType>,
                                              AMSResourceType);
template AMSTensor AMSTensor::create<int64_t>(ams::ArrayRef<IntDimType>,
                                              ams::ArrayRef<IntDimType>,
                                              AMSResourceType);

template AMSTensor AMSTensor::view<float>(float*,
                                          ams::ArrayRef<IntDimType>,
                                          ams::ArrayRef<IntDimType>,
                                          AMSResourceType);
template AMSTensor AMSTensor::view<double>(double*,
                                           ams::ArrayRef<IntDimType>,
                                           ams::ArrayRef<IntDimType>,
                                           AMSResourceType);
template AMSTensor AMSTensor::view<int32_t>(int32_t*,
                                            ams::ArrayRef<IntDimType>,
                                            ams::ArrayRef<IntDimType>,
                                            AMSResourceType);
template AMSTensor AMSTensor::view<int64_t>(int64_t*,
                                            ams::ArrayRef<IntDimType>,
                                            ams::ArrayRef<IntDimType>,
                                            AMSResourceType);

template AMSTensor AMSTensor::view<const float>(const float*,
                                                ams::ArrayRef<IntDimType>,
                                                ams::ArrayRef<IntDimType>,
                                                AMSResourceType);
template AMSTensor AMSTensor::view<const double>(const double*,
                                                 ams::ArrayRef<IntDimType>,
                                                 ams::ArrayRef<IntDimType>,
                                                 AMSResourceType);
template AMSTensor AMSTensor::view<const int32_t>(const int32_t*,
                                                  ams::ArrayRef<IntDimType>,
                                                  ams::ArrayRef<IntDimType>,
                                                  AMSResourceType);
template AMSTensor AMSTensor::view<const int64_t>(const int64_t*,
                                                  ams::ArrayRef<IntDimType>,
                                                  ams::ArrayRef<IntDimType>,
                                                  AMSResourceType);
