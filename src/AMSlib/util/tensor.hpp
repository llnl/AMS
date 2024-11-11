#include <cstdint>
#include <numeric>
#include <stdexcept>

#include "AMS.h"
#include "ArrayRef.hpp"
#include "SmallVector.h"
#include "wf/data_handler.hpp"
#include "wf/resource_manager.hpp"
#include "wf/utils.hpp"

class AMSTensor
{
  uint8_t* _data;
  size_t _elements;
  size_t _element_size;
  ams::SmallVector<size_t, 4> _shape;
  ams::SmallVector<size_t, 4> _strides;
  AMSDType _dType;            // AMS_SINGLE/AMS_DOUBLE
  AMSResourceType _location;  // CPU/GPU/Pinned
  bool _owned;


private:
  /**
 * @brief Creates a new tensor by casting all elements to the specified target type.
 * @tparam TargetType The data type to cast each element to.
 * @param[in] targetType The target AMSDType representing the target data type (e.g., AMS_SINGLE or AMS_DOUBLE).
 * @return A new AMSTensor with the elements cast to the specified type.
 */
  template <typename TargetType>
  AMSTensor castTensor(AMSDType targetType) const
  {
    // Create new tensor with same shape and strides, but different data type
    AMSTensor newTensor(_shape, _strides, targetType, _location);
    auto& rm = ams::ResourceManager::getInstance();

    if (_dType == targetType) {
      rm.copy<uint8_t>(
          _data, _location, newTensor.data<uint8_t>(), _location, _elements);
      return newTensor;
    }

    // Perform element-wise conversion
    assert(_location == AMS_HOST && "Unsupported GPU implementation");
    if (_dType == AMS_SINGLE) {
      ams::DataHandler<float>::cast_from_typevalue(_elements,
                                                   newTensor.data<TargetType>(),
                                                   data<float>());
    } else if (_dType == AMS_DOUBLE) {
      ams::DataHandler<double>::cast_from_typevalue(
          _elements, newTensor.data<TargetType>(), data<double>());
    } else {
      throw std::runtime_error("Unsupported source data type in castTensor.");
    }

    return newTensor;
  }

  /**
 * @brief Constructs a new AMSTensor with the specified shape, strides, data type, and location.
 *        This constructor is private and intended for internal use, such as creating views.
 * @param[in] shapes The shape of the tensor.
 * @param[in] strides The strides of the tensor.
 * @param[in] dType The data type of the tensor elements.
 * @param[in] location The memory location (e.g., CPU, GPU).
 * @param[in] view Set to true if this tensor is a view of another tensor (non-owning).
 */
  explicit AMSTensor(ams::ArrayRef<size_t> shapes,
                     ams::ArrayRef<size_t> strides,
                     AMSDType dType,
                     AMSResourceType location,
                     bool view = false)
      : _elements(computeNumElements(shapes)),
        _element_size(dtype_to_size(dType)),
        _shape(shapes),
        _strides(strides),
        _dType(dType),
        _location(location),
        _owned(view)
  {
    auto& rm = ams::ResourceManager::getInstance();
    if (!view) {
      _data = rm.allocate<uint8_t>(_elements * _element_size,
                                   _location,
                                   _element_size);
      if (!_data) {
        throw std::runtime_error("Failed to allocate memory for AMSTensor.");
      }
    }
  }


public:
  /**
 * @brief Creates a new AMSTensor and allocates the tensor memory.
 * @param[in] shapes The shape of the tensor.
 * @param[in] strides The strides of the tensor.
 * @param[in] dType The data type of the tensor elements.
 * @param[in] location The memory location (e.g., CPU, GPU).
 * @return A new AMSTensor with allocated memory.
 */
  static AMSTensor create(ams::ArrayRef<size_t> shapes,
                          ams::ArrayRef<size_t> strides,
                          AMSDType dType,
                          AMSResourceType location)
  {
    return AMSTensor(shapes, strides, dType, location);
  }

  /**
 * @brief Creates a view on an existing memory buffer.
 * @param[in] data Pointer to the existing data to be viewed.
 * @param[in] shapes The shape of the view tensor.
 * @param[in] strides The strides of the view tensor.
 * @param[in] dType The data type of the tensor elements.
 * @param[in] location The memory location (e.g., CPU, GPU).
 * @return A new AMSTensor that acts as a view of the existing data.
 */
  static AMSTensor view(uint8_t* data,
                        ams::ArrayRef<size_t> shapes,
                        ams::ArrayRef<size_t> strides,
                        AMSDType dType,
                        AMSResourceType location)
  {
    auto tensor = AMSTensor(shapes, strides, dType, location, true);
    tensor._data = data;
    return tensor;
  }

  /**
 * @brief Destructor for AMSTensor, deallocates memory if this tensor owns it.
 */
  ~AMSTensor()
  {
    // Only release whenwe own the pointer
    if (_owned && _data) {
      auto& rm = ams::ResourceManager::getInstance();
      rm.deallocate(_data, _location);
    }
  }

  /**
 * @brief Deleted copy assignment operator to prevent copying of tensors.
 */
  AMSTensor(const AMSTensor&) = delete;

  /**
 * @brief Move constructor for AMSTensor, transfers ownership of data.
 * @param[in,out] other The tensor to move from. It will be left in a valid but unspecified state.
 */
  AMSTensor& operator=(const AMSTensor&) = delete;

  /**
 * @brief Move assignment operator for AMSTensor, transfers ownership of data.
 * @param[in,out] other The tensor to move from. It will be left in a valid but unspecified state.
 * @return A reference to the updated tensor after move assignment.
 */
  AMSTensor(AMSTensor&& other) noexcept
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

  // Define move assignment operator
  AMSTensor& operator=(AMSTensor&& other) noexcept
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


  /**
 * @brief Retrieves a typed pointer to the underlying data.
 * @tparam T The data type to retrieve.
 * @return A typed pointer to the tensor's data.
 */
  template <typename T>
  T* data() const
  {
    return reinterpret_cast<T*>(_data);
  }


  /**
 * @brief Retrieves a specific element by index with bounds checking.
 * @tparam T The data type of the element to retrieve.
 * @param[in] index The index of the element to access.
 * @return A reference to the specified element.
 * @throw std::out_of_range if index is out of bounds.
 */
  template <typename T>
  T& at(std::size_t index) const
  {
    if (index >= _elements) {
      throw std::out_of_range("Index out of bounds in AMSTensor.");
    }
    return data<T>()[index];
  }

  /**
 * @brief Creates a copy of the tensor with elements converted to float (32-bit) representation.
 * @return A new AMSTensor with float (32-bit) elements.
 */
  AMSTensor fp32() const { return castTensor<float>(AMS_SINGLE); }

  /**
 * @brief Creates a copy of the tensor with elements converted to double (64-bit) representation.
 * @return A new AMSTensor with double (64-bit) elements.
 */
  AMSTensor fp64() const { return castTensor<double>(AMS_DOUBLE); }


  /**
 * @brief Creates a transposed view of the tensor by swapping two specified axes.
 * @param[in] axis1 The first axis to swap in the transposition.
 * @param[in] axis2 The second axis to swap in the transposition.
 * @return A new AMSTensor that is a transposed view of the original tensor.
 * @throw std::out_of_range if any axis is out of bounds.
 */
  AMSTensor transpose(size_t axis1 = 0, size_t axis2 = 1) const
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


  /**
 * @brief Concatenates multiple tensors along a specified axis, with broadcasting for singleton dimensions.
 * @param[in] tensors Array of tensors to concatenate.
 * @param[in] axis The axis along which to concatenate.
 * @return A new AMSTensor representing the concatenated result.
 * @throw std::invalid_argument if tensor shapes are incompatible or axis is out of bounds.
 */
  static AMSTensor concatenate(const ams::ArrayRef<AMSTensor> tensors,
                               size_t axis)
  {
    if (tensors.empty()) {
      throw std::invalid_argument("No tensors provided for concatenation.");
    }

    // Check compatibility of shapes, data types, and locations
    const AMSDType dType = tensors[0]._dType;
    const AMSResourceType location = tensors[0]._location;
    ams::SmallVector<size_t, 4> newShape = tensors[0]._shape;

    for (const auto& tensor : tensors) {
      if (tensor._dType != dType || tensor._location != location) {
        throw std::invalid_argument(
            "All tensors must have the same data type and location.");
      }
      if (tensor._shape.size() != newShape.size()) {
        throw std::invalid_argument(
            "All tensors must have the same number of dimensions.");
      }

      // Ensure dimensions match, except along the concatenation axis, allowing for broadcasting
      for (size_t i = 0; i < tensor._shape.size(); ++i) {
        if (i == axis) {
          newShape[axis] +=
              tensor._shape
                  [axis];  // Accumulate size along the concatenation axis
        } else if (tensor._shape[i] != newShape[i] && tensor._shape[i] != 1 &&
                   newShape[i] != 1) {
          throw std::invalid_argument(
              "All tensors must have compatible shapes for concatenation.");
        } else {
          // If one of the dimensions is 1, set newShape[i] to the max of both dimensions (broadcasting)
          newShape[i] = std::max(newShape[i], tensor._shape[i]);
        }
      }
    }

    // Calculate new strides for a contiguous layout in the new tensor
    ams::SmallVector<size_t, 4> newStrides(newShape.size());
    size_t elementSize = dtype_to_size(dType);
    newStrides.back() = elementSize;
    for (int i = newStrides.size() - 2; i >= 0; --i) {
      newStrides[i] = newShape[i + 1] * newStrides[i + 1];
    }

    // Create the new tensor with the calculated shape and strides
    AMSTensor result = AMSTensor::create(newShape, newStrides, dType, location);

    // Copy data from each tensor to the correct position in `result`
    size_t offset = 0;
    for (const auto& tensor : tensors) {
      size_t copySize = tensor._elements * elementSize;

      // Calculate starting offset for this tensor in the concatenated result
      uint8_t* destPtr = result._data + offset;

      // Handle broadcasting along singleton dimensions
      if (tensor._shape[axis] == 1) {
        // Broadcast the data along the concatenation axis
        for (size_t i = 0; i < newShape[axis] / tensor._shape[axis]; ++i) {
          std::memcpy(destPtr + i * newStrides[axis], tensor._data, copySize);
        }
      } else {
        // Regular copy without broadcasting
        std::memcpy(destPtr, tensor._data, copySize);
      }

      // Advance the offset along the concatenation axis
      offset += tensor._shape[axis] * newStrides[axis];
    }

    return result;
  }

  /**
 * @brief Accesses an element of the tensor at the specified multi-dimensional index.
 *
 * This function uses the tensor's strides and shape to calculate the correct
 * memory offset based on the provided indices, allowing access to an element
 * regardless of the tensor's layout or any transpositions.
 *
 * @tparam T The data type of the element to retrieve.
 * @param[in] indices A vector of indices specifying the location of the element
 *                    in each dimension. The number of indices must match the 
 *                    tensor's number of dimensions.
 * @return A reference to the element at the specified location.
 * @throw std::out_of_range if the number of indices does not match the tensor's 
 *                          dimensions or if any index is out of bounds.
 */
  template <typename T>
  T& elementAt(const ams::ArrayRef<size_t> indices) const
  {
    if (indices.size() != _shape.size()) {
      throw std::out_of_range(
          "Number of indices does not match tensor dimensions.");
    }

    size_t offset = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
      if (indices[i] >= _shape[i]) {
        throw std::out_of_range("Index out of bounds in elementAt.");
      }
      offset += indices[i] * _strides[i];
    }

    return *reinterpret_cast<T*>(_data + offset);
  }

  /**
 * @brief Computes the number of elements in the tensor given its shape.
 * @param[in] shapes The shape of the tensor as an array reference.
 * @return The total number of elements in the tensor.
 */
  static size_t computeNumElements(ams::ArrayRef<size_t> shapes)
  {
    return std::accumulate(shapes.begin(),
                           shapes.end(),
                           1,
                           std::multiplies<std::size_t>());
  }


  size_t elements() const { return _elements; }
  size_t element_size() const { return _element_size; }
  AMSDType dtype() const { return _dType; }
  AMSResourceType location() const { return _location; }
  ams::ArrayRef<size_t> strides() const { return _strides; }
  ams::ArrayRef<size_t> shape() const { return _shape; }
};
