#include <stdexcept>

#include "AMS.h"
#include "AMSTensor.hpp"
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
  _bytes = _elements * _element_size;
  _elements = computeNumElements(shapes);
  if (!_data) {
    throw std::runtime_error("Generating tensor with Null Pointer AMSTensor.");
  }
}

template <typename FPType, typename>
AMSTensor AMSTensor::create(ams::ArrayRef<AMSTensor::IntDimType> shapes,
                            ams::ArrayRef<AMSTensor::IntDimType> strides,
                            AMSResourceType location)
{
  auto numElements = computeNumElements(shapes);
  auto& rm = ams::ResourceManager::getInstance();
  if constexpr ((std::is_same_v<FPType, float>) ||
                (std::is_same_v<FPType, const float>)) {
    float* _data = rm.allocate<float>(numElements, location, sizeof(float));
    return AMSTensor(reinterpret_cast<uint8_t*>(_data),
                     shapes,
                     strides,
                     AMS_SINGLE,
                     location);
  } else if constexpr ((std::is_same_v<FPType, double>) ||
                       (std::is_same_v<FPType, const double>)) {
    double* _data = rm.allocate<double>(numElements, location, sizeof(double));
    return AMSTensor(reinterpret_cast<uint8_t*>(_data),
                     shapes,
                     strides,
                     AMS_DOUBLE,
                     location);
  } else {
    // This should never happen due to the type restriction
    static_assert(std::is_same_v<FPType, float> ||
                      std::is_same_v<FPType, double>,
                  "AMSTensor only supports float or double tensor creation");
  }
}


template <typename FPType, typename>
AMSTensor AMSTensor::view(FPType* data,
                          ams::ArrayRef<AMSTensor::IntDimType> shapes,
                          ams::ArrayRef<AMSTensor::IntDimType> strides,
                          AMSResourceType location)
{
  if constexpr ((std::is_same_v<FPType, float>) ||
                (std::is_same_v<FPType, const float>)) {
    return AMSTensor(
        (uint8_t*)data, shapes, strides, AMS_SINGLE, location, true);
  } else if constexpr ((std::is_same_v<FPType, double>) ||
                       (std::is_same_v<FPType, const double>)) {
    return AMSTensor(
        (uint8_t*)data, shapes, strides, AMS_DOUBLE, location, true);
  } else {
    static_assert(std::is_same_v<FPType, float> ||
                      std::is_same_v<FPType, const float> ||
                      std::is_same_v<FPType, const double> ||
                      std::is_same_v<FPType, double>,
                  "AMSTensor only supports float or double tensor view");
  }
  throw std::runtime_error("Should never get here\n");
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
      _owned(other._owned)
{
  other._data = nullptr;
  other._owned = false;
}

AMSTensor& AMSTensor::operator=(AMSTensor&& other) noexcept
{
  if (this != &other) {
    // Free existing resources

    // Steal resources from `other`
    _data = other._data;
    _elements = other._elements;
    _element_size = other._element_size;
    _shape = std::move(other._shape);
    _strides = std::move(other._strides);
    _dType = other._dType;
    _location = other._location;
    _owned = other._owned;

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

template AMSTensor AMSTensor::view<float>(float*,
                                          ams::ArrayRef<IntDimType>,
                                          ams::ArrayRef<IntDimType>,
                                          AMSResourceType);
template AMSTensor AMSTensor::view<double>(double*,
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
