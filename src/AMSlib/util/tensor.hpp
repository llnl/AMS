#include <sys/types.h>

#include <cstdint>
#include <numeric>
#include <stdexcept>

#include "AMS.h"
#include "ArrayRef.hpp"
#include "SmallVector.hpp"
#include "wf/data_handler.hpp"
#include "wf/resource_manager.hpp"
#include "wf/utils.hpp"

class AMSTensor
{
  uint8_t* _data;
  size_t _elements;
  size_t _element_size;
  size_t _batch_axis;
  ams::SmallVector<size_t> _shape;
  ams::SmallVector<size_t> _strides;
  AMSDType _dType;            // AMS_SINGLE/AMS_DOUBLE
  AMSResourceType _location;  // CPU/GPU/Pinned
  bool _owned;
  bool _contiguous;


private:
  // Helper function to check broadcasting compatibility for two dimensions
  static inline bool isBroadcastCompatible(size_t dim1, size_t dim2)
  {
    return (dim1 == dim2) || (dim1 == 1) || (dim2 == 1);
  }

  // Helper function to determine the broadcasted dimension size
  static inline size_t getBroadcastSize(size_t dim1, size_t dim2)
  {
    return std::max(dim1, dim2);
  }


  // Helper function to check if the tensor is contiguous in memory
  bool isContiguous(size_t expected_stride) const
  {
    for (int i = _shape.size() - 1; i >= 0; --i) {
      if (_strides[i] != expected_stride) return false;
      expected_stride *= _shape[i];
    }
    return true;
  }

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
   * @param[in] _batch_dim The axis holding the samples.
   * @param[in] view Set to true if this tensor is a view of another tensor (non-owning).
   */
  explicit AMSTensor(ams::ArrayRef<size_t> shapes,
                     ams::ArrayRef<size_t> strides,
                     AMSDType dType,
                     AMSResourceType location,
                     size_t batch_axis = 0,
                     bool view = false)
      : _elements(computeNumElements(shapes)),
        _element_size(dtype_to_size(dType)),
        _batch_axis(batch_axis),
        _shape(shapes),
        _strides(strides),
        _dType(dType),
        _location(location),
        _owned(!view)
  {
    _contiguous = isContiguous(_element_size);
    auto& rm = ams::ResourceManager::getInstance();
    if (_batch_axis >= _shape.size())
      throw std::invalid_argument(
          "Batch axis is larger than the shape of the tensor");
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
   * @param[in] batch_axis The axis that contains the batch data.
   * @return A new AMSTensor with allocated memory.
   */
  static AMSTensor create(ams::ArrayRef<size_t> shapes,
                          ams::ArrayRef<size_t> strides,
                          AMSDType dType,
                          AMSResourceType location,
                          size_t batch_axis = 0)
  {
    return AMSTensor(shapes, strides, dType, location, batch_axis);
  }

  /**
   * @brief Creates a view on an existing memory buffer.
   * @param[in] data Pointer to the existing data to be viewed.
   * @param[in] shapes The shape of the view tensor.
   * @param[in] strides The strides of the view tensor.
   * @param[in] dType The data type of the tensor elements.
   * @param[in] location The memory location (e.g., CPU, GPU).
   * @param[in] batch_axis The axis represeing the samples.
   * @return A new AMSTensor that acts as a view of the existing data.
   */
  static AMSTensor view(uint8_t* data,
                        ams::ArrayRef<size_t> shapes,
                        ams::ArrayRef<size_t> strides,
                        AMSDType dType,
                        AMSResourceType location,
                        size_t batch_axis = 0)
  {
    auto tensor = AMSTensor(shapes, strides, dType, location, batch_axis, true);
    tensor._data = data;
    return tensor;
  }

  // Helper function to compute if a reshape is feasible without copying
  bool canReshapeWithStrides(const ams::ArrayRef<size_t> new_shape)
  {
    // Check if total number of elements is the same
    if (computeNumElements(new_shape) != computeNumElements(_shape)) {
      return false;
    }

    // Check if the original data layout is contiguous
    return isContiguous(_element_size);

    // If we reach here, data is contiguous and can be reshaped to any compatible shape
    return true;
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
      _data = nullptr;
      _owned = false;
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
        _batch_axis(other._batch_axis),
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
      _batch_axis = other._batch_axis;
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
    auto _new_batch_axis = _batch_axis;
    if (_new_batch_axis == axis1)
      _new_batch_axis = axis2;
    else if (_new_batch_axis == axis2)
      _new_batch_axis = axis1;

    // Swap the specified axes in both shape and strides
    std::swap(newShape[axis1], newShape[axis2]);
    std::swap(newStrides[axis1], newStrides[axis2]);

    // Create a new tensor with the same data, new shape, and strides
    AMSTensor transposedTensor =
        view(_data, newShape, newStrides, _dType, _location, _new_batch_axis);

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
    ams::SmallVector<size_t> newShape = tensors[0]._shape;
    newShape[axis] = 0;

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
    std::cout << "Strides are : ";
    for (int i = 0; i < newStrides.size(); i++) {
      std::cout << newStrides[i] << " ";
    }
    std::cout << "\n";
    std::cout << "Shapes are : ";
    for (int i = 0; i < newShape.size(); i++) {
      std::cout << newShape[i] << " ";
    }
    std::cout << "\n";


    // Create the new tensor with the calculated shape and strides
    AMSTensor result = AMSTensor::create(newShape, newStrides, dType, location);

    // Copy data from each tensor to the correct position in `result`
    size_t offset = 0;
    // compute num elements left from concatenation-axis
    size_t num_elements_left = std::accumulate(result._shape.begin(),
                                               &result._shape[axis],
                                               1,
                                               std::multiplies<size_t>());
    int tid = 0;
    int index = 0;
    for (auto& tensor : tensors) {
      std::cout << "Tid is " << tid << "\n";
      int src_index = 0;
      // Determine the copy size for a single slice along the concatenation axis
      size_t copy_size_per_slice = tensor._strides[axis];
      // tensors differ only on the axis dimension
      size_t num_elements_right = std::accumulate(&tensor._shape[axis],
                                                  tensor._shape.end(),
                                                  1,
                                                  std::multiplies<size_t>());
      std::cout << "Tensor Shape " << tensor.shape()[0] << ","
                << tensor.shape()[1] << "\n";
      std::cout << "num_elements_left " << num_elements_left << "\n";
      std::cout << "num_elements_right " << num_elements_right << "\n";
      for (size_t left = 0; left < num_elements_left; left++) {
        uint8_t* dst_start_ptr =
            &result._data[left * result._strides[axis - 1]] +
            tid * result._strides[axis];
        uint8_t* src_start_ptr =
            &tensor._data[left * tensor._strides[axis - 1]];
        std::cout << "Next left\n";
        std::cout << "result._strides " << result._strides[axis - 1] << "\n";
        std::cout << "tensor._strides " << tensor._strides[axis - 1] << "\n";
        for (int right = 0; right < num_elements_right; right++) {
          index = ((uintptr_t)dst_start_ptr - (uintptr_t)result._data) /
                  result._element_size;
          src_index = ((uintptr_t)src_start_ptr - (uintptr_t)tensor._data) /
                      tensor._element_size;
          std::cout << "Dest index: " << index << "\n";
          std::cout << "Src index: " << src_index << "\n";
          if (tensor._dType == AMS_SINGLE) {
            float tmp = *reinterpret_cast<float*>(src_start_ptr);
            std::cout << "Assigning float " << tmp << "\n";
            for (int i = 0; i < sizeof(float); i++) {
              dst_start_ptr[i] = src_start_ptr[i];
            }
            tmp = *reinterpret_cast<float*>(dst_start_ptr);
            std::cout << "Dest value " << tmp << "\n";

            dst_start_ptr += sizeof(float);
            src_start_ptr += sizeof(float);
          } else if (tensor._dType == AMS_DOUBLE) {
            for (int i = 0; i < sizeof(double); i++) {
              dst_start_ptr[i] = src_start_ptr[i];
            }
            dst_start_ptr += sizeof(double);
            src_start_ptr += sizeof(double);
          }
          index++;
          src_index++;
        }
      }
      // Here we move the pointer of the 'tid' index.
      tid += tensor._shape[axis];
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

    if (sizeof(T) != _element_size)
      throw std::invalid_argument(
          "Accessing element at tensor-position with incompatible "
          "data-type-size");

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

  // Helper function to compute contiguous strides for a given shape
  static ams::SmallVector<size_t> computeStrides(
      const ams::ArrayRef<size_t>& shape,
      const size_t element_size)
  {
    ams::SmallVector<size_t> strides(shape.size());
    size_t stride = element_size;
    for (int i = shape.size() - 1; i >= 0; --i) {
      strides[i] = stride;
      stride *= shape[i];
    }
    return strides;
  }

  // Reshape function with attempt to adjust strides
  AMSTensor reshape(const ams::ArrayRef<size_t>& new_shape)
  {
    size_t new_elements = computeNumElements(new_shape);
    if (new_elements != _elements) {
      throw std::invalid_argument(
          "New shape must have the same total number of elements as the "
          "original tensor.");
    }

    if (canReshapeWithStrides(new_shape)) {
      // Data is contiguous and can be reshaped without copying
      auto new_strides = computeStrides(new_shape, _element_size);
      return AMSTensor::view(_data, new_shape, new_strides, _dType, _location);
    } else {
      // Data is non-contiguous or not compatible with new strides, create a contiguous copy
      AMSTensor contiguous_tensor =
          AMSTensor::create(new_shape,
                            computeStrides(new_shape, _element_size),
                            _dType,
                            _location);

      // Copy data from the original tensor to the contiguous tensor
      auto& rm = ams::ResourceManager::getInstance();
      rm.copy(_data,
              _location,
              contiguous_tensor.data<uint8_t>(),
              _location,
              _elements * _element_size);

      return contiguous_tensor;
    }
  }

  // Function to align and expand a tensor for batch alignment
  // Helper function to align and expand a tensor's shape for batch alignment
  static AMSTensor alignAndExpand(AMSTensor& tensor,
                                  int target_batch_size,
                                  int max_rank)
  {
    auto shape = tensor.shape();
    auto batch_axis = tensor.batch_axis();

    ams::SmallVector<size_t, 4> new_shape(max_rank, 1);
    new_shape[batch_axis] = target_batch_size;  // Set the target batch size
    std::cout << "Batch Axis is " << batch_axis << "\n";
    std::cout << "New shape size is " << new_shape.size() << "\n";

    // Move tensor's dimensions to the appropriate location in `new_shape`
    for (size_t i = 0, j = 0; i < shape.size(); ++i, ++j) {
      if (i == batch_axis) {
        new_shape[batch_axis] = shape[batch_axis];
      } else {
        new_shape[j] = shape[i];
      }
    }
    std::cout << "Before reshape \n";
    tensor.dump();
    return tensor.reshape(new_shape);  // Reshape without copying if possible
  }

  void dump_vector(const std::string name, ams::ArrayRef<size_t> Vec)
  {
    std::cout << name << " [";
    for (auto I : Vec) {
      std::cout << I << ", ";
    }
    std::cout << "]\n";
  }

  void dump()
  {
    dump_vector("Stride", _strides);
    dump_vector("Shape", _shape);
  }

  // Main concatenation function with broadcasting checks
  static AMSTensor concatenateTensors(ams::MutableArrayRef<AMSTensor> tensors)
  {
    if (tensors.empty()) {
      throw std::invalid_argument("No tensors provided for concatenation.");
    }

    // Determine the batch axis and target batch size based on the first tensor

    int batch_axis = tensors[0].batch_axis();
    int target_batch_size = tensors[0].shape()[batch_axis];

    // Find the maximum rank among all tensors to align shapes
    int max_rank = 0;
    for (const auto& tensor : tensors) {
      max_rank = std::max(max_rank, static_cast<int>(tensor.shape().size()));
    }

    // Prepare final shape based on broadcasting rules and target batch size
    ams::SmallVector<size_t, 4> final_shape(max_rank, 1);
    final_shape[batch_axis] =
        target_batch_size;  // Set batch size in final shape

    std::vector<AMSTensor> aligned_tensors;
    std::cout << "Max Rank is " << max_rank << "\n";

    // Align all tensors to the same rank and check for broadcasting compatibility
    for (AMSTensor& tensor : tensors) {
      // Align and expand each tensor to have the same rank as `max_rank`
      std::cout << "Original Tensor is \n";
      tensor.dump();
      AMSTensor aligned_tensor =
          alignAndExpand(tensor, target_batch_size, max_rank);
      std::cout << "Aligned Tensor is \n";
      aligned_tensor.dump();

      // Update final shape based on broadcasting compatibility
      const auto& aligned_shape = aligned_tensor.shape();
      for (size_t i = 0; i < max_rank; ++i) {
        std::cout << "final_shape rank: " << final_shape.size()
                  << ", aligned_shape rank: " << aligned_shape.size()
                  << std::endl;
        assert(final_shape.size() == aligned_shape.size() &&
               "Shapes must have the same rank for broadcasting");
        if (!isBroadcastCompatible(final_shape[i], aligned_shape[i])) {
          throw std::invalid_argument(
              "Tensors have incompatible shapes for concatenation.");
        }
        final_shape[i] = getBroadcastSize(final_shape[i], aligned_shape[i]);
      }
      aligned_tensors.push_back(std::move(aligned_tensor));
    }

    // Concatenate tensors along the last dimension or specified axis
    for (auto& V : aligned_tensors) {
      std::cout << "Shape ";
      for (auto s : V.shape()) {
        std::cout << " " << s << " ";
      }
      std::cout << "\n";
    }

    std::cout << "Max Ranking is " << max_rank << "\n";


    return AMSTensor::concatenate(aligned_tensors, max_rank - 1);
  }

  size_t elements() const { return _elements; }
  size_t element_size() const { return _element_size; }
  size_t batch_axis() const { return _batch_axis; }
  AMSDType dType() const { return _dType; }
  AMSResourceType location() const { return _location; }
  ams::ArrayRef<size_t> strides() const { return _strides; }
  ams::ArrayRef<size_t> shape() const { return _shape; }
  bool contiguous() const { return _contiguous; }
};
