#include <H5Tpublic.h>
#include <hdf5.h>
#include <torch/torch.h>

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "wf/basedb.hpp"

template <typename T>
int testReadHDF5Dataset(const std::string& filePath,
                        const std::string& datasetName,
                        hid_t DataType,
                        std::vector<T> correct_contents)
{
  // Open the HDF5 file
  hid_t file_id = H5Fopen(filePath.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file_id < 0) {
    std::cerr << "Failed to open HDF5 file: " << filePath << std::endl;
    return -1;
  }

  // Open the dataset
  hid_t dataset_id = H5Dopen(file_id, datasetName.c_str(), H5P_DEFAULT);
  if (dataset_id < 0) {
    std::cerr << "Failed to open dataset: " << datasetName << std::endl;
    H5Fclose(file_id);
    return -1;
  }

  // Get the dataspace of the dataset
  hid_t dataspace_id = H5Dget_space(dataset_id);
  if (dataspace_id < 0) {
    std::cerr << "Failed to get dataspace for dataset: " << datasetName
              << std::endl;
    H5Dclose(dataset_id);
    H5Fclose(file_id);
    return -1;
  }

  // Get the number of dimensions and size of each dimension
  int ndims = H5Sget_simple_extent_ndims(dataspace_id);
  std::vector<hsize_t> dims(ndims);
  H5Sget_simple_extent_dims(dataspace_id, dims.data(), nullptr);

  // Print dimensions
  for (size_t i = 0; i < dims.size(); ++i) {
    std::cout << dims[i] << (i < dims.size() - 1 ? " x " : "\n");
  }

  // Determine the datatype of the dataset
  hid_t datatype_id = H5Dget_type(dataset_id);
  // Get the size of the datatype
  size_t datatype_size = H5Tget_size(datatype_id);


  // Only handle floating-point data for this example
  std::vector<T> data;
  if (datatype_size == sizeof(T)) {
    // Allocate memory for the dataset
    size_t total_elements = 1;
    for (auto dim : dims) {
      total_elements *= dim;
    }
    data.resize(total_elements);
    // Read the dataset
    if (H5Dread(
            dataset_id, DataType, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data()) <
        0)
      std::cerr << "Failed to read dataset: " << datasetName << std::endl;
  } else {
    std::cerr << "Unsupported data type for dataset: " << datasetName
              << std::endl;
    return -1;
  }

  // Close HDF5 objects
  H5Tclose(datatype_id);
  H5Sclose(dataspace_id);
  H5Dclose(dataset_id);
  H5Fclose(file_id);
  std::cout << "Read: " << std::string(data.begin(), data.end());
  return (data == correct_contents) ? 0 : -1;
}

// Function to read a dataset and compare it with the expected tensor
bool verifyDatasetContents(const std::string& fileName,
                           const std::string& datasetName,
                           const std::vector<torch::Tensor>& expectedTensors)
{
  // Open the HDF5 file
  hid_t file_id = H5Fopen(fileName.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file_id < 0) {
    throw std::runtime_error("Failed to open HDF5 file.");
  }

  // Open the dataset
  hid_t dset_id = H5Dopen2(file_id, datasetName.c_str(), H5P_DEFAULT);
  if (dset_id < 0) {
    H5Fclose(file_id);
    throw std::runtime_error("Failed to open dataset.");
  }

  // Get the dataspace
  hid_t space_id = H5Dget_space(dset_id);
  if (space_id < 0) {
    H5Dclose(dset_id);
    H5Fclose(file_id);
    throw std::runtime_error("Failed to get dataspace.");
  }

  // Get the dataset dimensions
  int ndims = H5Sget_simple_extent_ndims(space_id);
  if (ndims < 0) {
    H5Sclose(space_id);
    H5Dclose(dset_id);
    H5Fclose(file_id);
    throw std::runtime_error("Failed to get number of dimensions.");
  }

  std::vector<hsize_t> dims(ndims);
  if (H5Sget_simple_extent_dims(space_id, dims.data(), NULL) < 0) {
    H5Sclose(space_id);
    H5Dclose(dset_id);
    H5Fclose(file_id);
    throw std::runtime_error("Failed to get dataset dimensions.");
  }

  // Close dataspace
  H5Sclose(space_id);

  // Flatten the dataset dimensions into a total size
  size_t totalSize = 1;
  for (const auto& dim : dims) {
    totalSize *= dim;
  }

  // Allocate a tensor to read the dataset
  auto readTensor =
      torch::empty({static_cast<int64_t>(totalSize)}, torch::kFloat);

  // Read the dataset into the tensor
  herr_t status = H5Dread(dset_id,
                          H5T_NATIVE_FLOAT,
                          H5S_ALL,
                          H5S_ALL,
                          H5P_DEFAULT,
                          readTensor.data_ptr());
  if (status < 0) {
    H5Dclose(dset_id);
    H5Fclose(file_id);
    throw std::runtime_error("Failed to read dataset.");
  }

  // Close dataset and file
  H5Dclose(dset_id);
  H5Fclose(file_id);

  // Concatenate all expected tensors into one
  auto expectedTensor = torch::cat(expectedTensors).flatten();

  // Compare the tensors
  if (!torch::allclose(readTensor, expectedTensor)) {
    throw std::runtime_error(
        "Dataset contents do not match the expected tensors.");
  }

  std::cout << "Dataset contents match the expected tensors!" << std::endl;
  return true;
}


int main(int argc, char* argv[])
{
  if (argc != 3) {
    std::cerr << "Wrong command line, correct one should be:\n";
    std::cerr << argv[0] << " <path-to-directory> <domain-name> <rid>";
  }
  std::string directory(argv[1]);
  std::string domain_name(argv[2]);
  std::string filename;

  std::vector<torch::Tensor> inputTensors, outputTensors;
  for (int i = 0; i < 2; i++) {
    {
      // Scope it to automatically close C++ deconstructor and close file
      auto db = ams::db::hdf5DB(directory, domain_name, 0);
      torch::Tensor IData =
          torch::rand({21, 4}, torch::TensorOptions().dtype(torch::kFloat));
      torch::Tensor OData =
          torch::rand({21, 4}, torch::TensorOptions().dtype(torch::kFloat));

      // Test 1. Open file and write data to it.
      filename = db.getFilename();
      db.store(IData, OData);
      inputTensors.emplace_back(std::move(IData));
      outputTensors.emplace_back(std::move(OData));
    }
    if (!verifyDatasetContents(filename, "input_data", inputTensors) ||
        !verifyDatasetContents(filename, "output_data", outputTensors))
      return -1;
    std::cout << ((i == 0) ? "Creating empty file and checking contents is "
                             "correct\n"
                           : "Opening existing file and checking contents is "
                             "correct\n");
  }

  std::string dn("domain_name");
  std::vector<char> _dn(domain_name.begin(), domain_name.end());
  return testReadHDF5Dataset(filename, dn, H5T_NATIVE_CHAR, _dn);
}
