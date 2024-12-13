#include "AMS.h"
#include "ArrayRef.hpp"
#include "SmallVector.hpp"
#include "ams_tensor.hpp"
#include "wf/data_handler.hpp"
#include "wf/resource_manager.hpp"
#include "wf/utils.hpp"


/**
   * @brief Computes the number of elements in the tensor given its shape.
   * @param[in] shapes The shape of the tensor as an array reference.
   * @return The total number of elements in the tensor.
   */
template <typename T>
static inline size_t computeNumElements(ams::ArrayRef<T> shapes)
{
  return std::accumulate(shapes.begin(),
                         shapes.end(),
                         1,
                         std::multiplies<std::size_t>());
}
// Helper function to check if the tensor is contiguous in memory
bool AMSTensor::isContiguous(size_t expected_stride) const
{
  for (int i = _shape.size() - 1; i >= 0; --i) {
    if (_strides[i] != expected_stride) return false;
    expected_stride *= _shape[i];
  }
  return true;
}


AMSTensor::AMSTensor(ams::ArrayRef<size_t> shapes,
                     ams::ArrayRef<size_t> strides,
                     AMSDType dType,
                     AMSResourceType location,
                     bool view)
    : _elements(computeNumElements(shapes)),
      _element_size(dtype_to_size(dType)),
      _shape(shapes),
      _strides(strides),
      _dType(dType),
      _location(location),
      _owned(!view)
{
  _contiguous = isContiguous(_element_size);
  auto& rm = ams::ResourceManager::getInstance();
  if (!view) {
    _data = rm.allocate<uint8_t>(_elements * _element_size,
                                 _location,
                                 _element_size);
    _bytes = _elements * _element_size;
    if (!_data) {
      throw std::runtime_error("Failed to allocate memory for AMSTensor.");
    }
  }
}


AMSTensor AMSTensor::create(ams::ArrayRef<size_t> shapes,
                            ams::ArrayRef<size_t> strides,
                            AMSDType dType,
                            AMSResourceType location)
{
  return AMSTensor(shapes, strides, dType, location);
}

AMSTensor AMSTensor::view(uint8_t* data,
                          ams::ArrayRef<size_t> shapes,
                          ams::ArrayRef<size_t> strides,
                          AMSDType dType,
                          AMSResourceType location)
{
  auto tensor = AMSTensor(shapes, strides, dType, location, true);
  tensor._data = data;
  return tensor;
}

AMSTensor AMSTensor::view(AMSTensor& tensor)
{
  return AMSTensor::view(tensor._data,
                         tensor._shape,
                         tensor._strides,
                         tensor._dType,
                         tensor._location);
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

AMSTensor AMSTensor::transpose(size_t axis1, size_t axis2) const
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
  AMSTensor transposedTensor =
      view(_data, newShape, newStrides, _dType, _location);

  return transposedTensor;
}
