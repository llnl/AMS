#pragma once
#include <sys/types.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "AMSTypes.hpp"
#include "ArrayRef.hpp"
#include "SmallVector.hpp"

namespace ams
{

class AMSTensor
{
public:
  using IntDimType = long int;
  IntDimType elements() const { return _elements; }
  IntDimType element_size() const { return _element_size; }
  AMSDType dType() const { return _dType; }
  AMSResourceType location() const { return _location; }
  ams::ArrayRef<IntDimType> strides() const { return _strides; }
  ams::ArrayRef<IntDimType> shape() const { return _shape; }
  bool contiguous() const { return _contiguous; }


private:
  uint8_t* _data;
  IntDimType _elements;
  IntDimType _element_size;
  ams::SmallVector<IntDimType> _shape;
  ams::SmallVector<IntDimType> _strides;
  AMSDType _dType;            // AMS_SINGLE/AMS_DOUBLE
  AMSResourceType _location;  // CPU/GPU/Pinned
  bool _owned;
  bool _contiguous;
  bool _bytes;

  // Helper function to check if the tensor is contiguous in memory
  bool isContiguous(IntDimType expected_stride) const;

  /**
   * @brief Constructs a new AMSTensor with the specified shape, strides, data type, and location.
   *        This constructor is private and intended for internal use, such as creating views.
   * @param[in] shapes The shape of the tensor.
   * @param[in] strides The strides of the tensor.
   * @param[in] dType The data type of the tensor elements.
   * @param[in] location The memory location (e.g., CPU, GPU).
   * @param[in] view Set to true if this tensor is a view of another tensor (non-owning).
   */
  explicit AMSTensor(uint8_t* data,
                     ams::ArrayRef<IntDimType> shapes,
                     ams::ArrayRef<IntDimType> strides,
                     AMSDType dType,
                     AMSResourceType location,
                     bool view = false);


public:
  /**
   * @brief Creates a new AMSTensor and allocates the tensor memory.
   * @param[in] shapes The shape of the tensor.
   * @param[in] strides The strides of the tensor.
   * @param[in] dType The data type of the tensor elements.
   * @param[in] location The memory location (e.g., CPU, GPU).
   * @return A new AMSTensor with allocated memory.
   */
  template <typename FPType,
            typename = std::enable_if_t<std::is_floating_point<FPType>::value>>
  static AMSTensor create(ams::ArrayRef<IntDimType> shapes,
                          ams::ArrayRef<IntDimType> strides,
                          AMSResourceType location);

  /**
   * @brief Creates a view on an existing memory buffer.
   * @param[in] data Pointer to the existing data to be viewed.
   * @param[in] shapes The shape of the view tensor.
   * @param[in] strides The strides of the view tensor.
   * @param[in] dType The data type of the tensor elements.
   * @param[in] location The memory location (e.g., CPU, GPU).
   * @return A new AMSTensor that acts as a view of the existing data.
   */
  template <typename FPType,
            typename = std::enable_if_t<std::is_floating_point<FPType>::value>>
  static AMSTensor view(FPType* data,
                        ams::ArrayRef<IntDimType> shapes,
                        ams::ArrayRef<IntDimType> strides,
                        AMSResourceType location);


  static AMSTensor view(AMSTensor& tensor);

  /**
   * @brief Destructor for AMSTensor, deallocates memory if this tensor owns it.
   */
  ~AMSTensor();


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
  AMSTensor(AMSTensor&& other) noexcept;

  // Define move assignment operator
  AMSTensor& operator=(AMSTensor&& other) noexcept;

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

  void* raw_data() const { return reinterpret_cast<void*>(_data); }

  /**
   * @brief Creates a transposed view of the tensor by swapping two specified axes.
   * @param[in] axis1 The first axis to swap in the transposition.
   * @param[in] axis2 The second axis to swap in the transposition.
   * @return A new AMSTensor that is a transposed view of the original tensor.
   * @throw std::out_of_range if any axis is out of bounds.
   */
  AMSTensor transpose(IntDimType axis1 = 0, IntDimType axis2 = 1) const;
};

// Explicit instantiation declarations
extern template AMSTensor AMSTensor::create<float>(
    ams::ArrayRef<AMSTensor::IntDimType> shapes,
    ams::ArrayRef<AMSTensor::IntDimType> strides,
    AMSResourceType location);
extern template AMSTensor AMSTensor::create<double>(
    ams::ArrayRef<AMSTensor::IntDimType> shapes,
    ams::ArrayRef<AMSTensor::IntDimType> strides,
    AMSResourceType location);
}  // namespace ams
