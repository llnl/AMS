#include <AMS.h>
#include <ATen/Context.h>

#include <cstdint>
#include <numeric>

#include "resource_manager.hpp"
#include "utils.hpp"

// Utility function to calculate element size
static size_t elementSize(const AMSDType type)
{
  switch (type) {
    case AMS_SINGLE:
      return sizeof(float);
    case AMS_DOUBLE:
      return sizeof(double);
    default:
      throw std::invalid_argument("Unsupported data type");
  }
}

struct HostMemoryDeleter {
  void operator()(void* ptr) const
  {
    auto& rm = ams::ResourceManager::getInstance();
    rm.deallocate(ptr, AMSResourceType::AMS_HOST);
  }
};


struct DeviceMemoryDeleter {
  void operator()(void* ptr) const
  {
    auto& rm = ams::ResourceManager::getInstance();
    rm.deallocate(ptr, AMSResourceType::AMS_DEVICE);
  }
};


// AMSTensor class definition
class AMSTensor
{

private:
  ams::ResourceManager& rm;
  std::vector<size_t> shape_;
  std::vector<size_t> strides_;
  AMSDType dtype_;
  AMSResourceType location_;
  size_t num_elements_;
  std::shared_ptr<uint8_t> data_;  // Pointer to tensor data
  size_t bytes_;

  size_t calculateNumElements(const std::vector<size_t>& shape) const
  {
    return std::accumulate(shape.begin(),
                           shape.end(),
                           1,
                           std::multiplies<size_t>());
  }

  std::vector<size_t> calculateStrides(const std::vector<size_t>& shape) const
  {
    std::vector<size_t> strides(shape.size());
    size_t stride = 1;
    for (int i = shape.size() - 1; i >= 0; --i) {
      strides[i] = stride;
      stride *= shape[i];
    }
    return strides;
  }

public:
  // Constructor for AMSTensor
  AMSTensor(const std::vector<size_t>& shape,
            AMSDType dtype,
            AMSResourceType location = AMSResourceType::AMS_HOST)
      : rm(ams::ResourceManager::getInstance()),
        shape_(shape),
        dtype_(dtype),
        location_(location)
  {

    // Calculate the total number of elements
    num_elements_ = 1;
    for (auto dim : shape) {
      num_elements_ *= dim;
    }

    // Allocate memory for the tensor based on datatype and shape
    bytes_ = num_elements_ * elementSize(dtype);
    if (location == AMSResourceType::AMS_HOST)
      data_ = std::shared_ptr<uint8_t>(rm.allocate<uint8_t>(bytes_, location),
                                       HostMemoryDeleter());
    else if (location == AMSResourceType::AMS_DEVICE)
      data_ = std::shared_ptr<uint8_t>(rm.allocate<uint8_t>(bytes_, location),
                                       DeviceMemoryDeleter());
  }

  // Destructor to free memory
  ~AMSTensor() {}

  // Accessors
  const std::vector<size_t>& shape() const { return shape_; }
  AMSDType dtype() const { return dtype_; }
  AMSResourceType location() const { return location_; }
  size_t numElements() const { return num_elements_; }

  // Data access methods
  template <typename T>
  std::shared_ptr<uint8_t> data() const
  {
    return (data_);
  }

  // Reshape the tensor (in-place)
  void reshape(const std::vector<size_t>& new_shape)
  {
    size_t new_num_elements = calculateNumElements(new_shape);
    if (new_num_elements != num_elements_) {
      throw std::invalid_argument(
          "Total number of elements must remain constant when reshaping");
    }
    shape_ = new_shape;
    strides_ = calculateStrides(new_shape);
  }

  // Index calculation
  size_t flattenIndex(const std::vector<size_t>& indices) const
  {
    if (indices.size() != shape_.size()) {
      throw std::invalid_argument("Incorrect number of indices");
    }

    size_t flat_index = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
      flat_index += indices[i] * strides_[i];
    }
    return flat_index;
  }


  template <typename TypeValue>
  static AMSTensor from(std::vector<TypeValue*> data,
                        size_t totalElements,
                        AMSDType dType,
                        AMSResourceType location)
  {
    // This allocates the memory we need and the shape we need.
    AMSTensor ATensor({totalElements, data.size()}, dType, location);
    if (getDType<float>() == dType) {
    }
  }
};
