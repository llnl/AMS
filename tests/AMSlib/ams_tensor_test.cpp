#include <cassert>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <stdexcept>

#include "util/tensor.hpp"


void testCastTensor()
{
  ams::SmallVector<size_t, 4> shape = {2, 3};
  ams::SmallVector<size_t, 4> strides = {3 * sizeof(float), sizeof(float)};
  AMSTensor tensor = AMSTensor::create(shape, strides, AMS_SINGLE, AMS_HOST);

  for (size_t i = 0; i < tensor.elements(); ++i) {
    tensor.at<float>(i) = static_cast<float>(i + 1);
  }

  // Cast tensor to double
  AMSTensor doubleTensor = tensor.fp64();

  // Check that data was cast correctly
  for (size_t i = 0; i < doubleTensor.elements(); ++i) {
    assert(doubleTensor.at<double>(i) == static_cast<double>(i + 1) &&
           "castTensor failed.");
  }
}

void testComputeNumElements()
{
  ams::SmallVector<size_t> shape = {4, 5, 6};
  size_t expected_elements = 4 * 5 * 6;
  size_t elements = AMSTensor::computeNumElements(shape);
  assert(elements == expected_elements && "computeNumElements failed.");
}

void testCreate()
{
  ams::SmallVector<size_t> shape = {4, 4};
  ams::SmallVector<size_t> strides = {4 * sizeof(float), sizeof(float)};
  AMSTensor tensor = AMSTensor::create(shape, strides, AMS_SINGLE, AMS_HOST);

  assert(shape == tensor.shape() && "Shape mismatch in create.");
  assert(strides == tensor.strides() && "Stride mismatch in create.");
  assert(tensor.dType() == AMS_SINGLE && "Data type mismatch in create.");
}


void testView()
{
  ams::SmallVector<size_t> shape = {2, 2};
  ams::SmallVector<size_t> strides = {2 * sizeof(float), sizeof(float)};
  AMSTensor tensor = AMSTensor::create(shape, strides, AMS_SINGLE, AMS_HOST);

  for (size_t i = 0; i < tensor.elements(); ++i) {
    tensor.at<float>(i) = static_cast<float>(i);
  }

  // Create a view of the original tensor
  AMSTensor viewTensor = AMSTensor::view(
      tensor.data<uint8_t>(), shape, strides, AMS_SINGLE, AMS_HOST);
  assert(shape == viewTensor.shape() && "Shape mismatch in view.");
  assert(strides == viewTensor.strides() && "Stride mismatch in view.");

  for (size_t i = 0; i < viewTensor.elements(); ++i) {
    assert(viewTensor.at<float>(i) == tensor.at<float>(i) &&
           "View data mismatch.");
  }
}


void testMove()
{
  ams::SmallVector<size_t> shape = {3, 3};
  ams::SmallVector<size_t> strides = {3 * sizeof(float), sizeof(float)};
  AMSTensor tensor = AMSTensor::create(shape, strides, AMS_SINGLE, AMS_HOST);
  tensor.at<float>(0) = 1.0f;

  // Move constructor
  AMSTensor movedTensor(std::move(tensor));
  assert(shape == movedTensor.shape() && "Shape mismatch in move constructor.");
  assert(movedTensor.at<float>(0) == 1.0f &&
         "Data mismatch in move constructor.");

  // Move assignment
  AMSTensor anotherTensor = std::move(movedTensor);
  assert(anotherTensor.at<float>(0) == 1.0f &&
         "Data mismatch in move assignment.");
}

void testDataAccess()
{
  ams::SmallVector<size_t> shape = {3, 3};
  ams::SmallVector<size_t> strides = {3 * sizeof(float), sizeof(float)};
  AMSTensor tensor = AMSTensor::create(shape, strides, AMS_SINGLE, AMS_HOST);

  tensor.at<float>(0) = 10.0f;
  assert(tensor.data<float>()[0] == 10.0f && "Data access mismatch.");
}


void testConversion()
{
  ams::SmallVector<size_t, 2> shape = {3, 3};
  ams::SmallVector<size_t, 2> strides = {3 * sizeof(double), sizeof(double)};
  AMSTensor tensor = AMSTensor::create(shape, strides, AMS_DOUBLE, AMS_HOST);
  assert(tensor.element_size() == sizeof(double));

  tensor.at<double>(0) = 5.0;
  AMSTensor floatTensor = tensor.fp32();
  assert(floatTensor.element_size() == sizeof(float));
  assert(floatTensor.at<float>(0) == 5.0f && "fp32 conversion failed.");
}


void testTranspose()
{
  // Define shape and strides for a 2x3 tensor
  ams::SmallVector<size_t> shape = {2, 3};
  ams::SmallVector<size_t> strides = {3 * sizeof(float), sizeof(float)};
  AMSTensor tensor = AMSTensor::create(shape, strides, AMS_SINGLE, AMS_HOST);

  // Populate tensor with values for easy checking after transpose
  tensor.at<float>(0) = 1.0f;  // (0,0)
  tensor.at<float>(1) = 2.0f;  // (0,1)
  tensor.at<float>(2) = 3.0f;  // (0,2)
  tensor.at<float>(3) = 4.0f;  // (1,0)
  tensor.at<float>(4) = 5.0f;  // (1,1)
  tensor.at<float>(5) = 6.0f;  // (1,2)

  // Transpose the tensor (swap axes 0 and 1)
  AMSTensor transposed = tensor.transpose(0, 1);

  for (int i = 0; i < tensor.elements(); i++) {
    std::cout << "Value at " << i << " is : " << tensor.at<float>(i)
              << " Transposed is " << transposed.at<float>(i) << "\n";
  }

  // Check that the transposed tensor has the correct shape
  assert(transposed.shape()[0] == 3 && transposed.shape()[1] == 2 &&
         "Shape mismatch in transpose.");

  // Verify that the transposed data matches expected positions
  assert(transposed.at<float>(0) == 1.0f &&
         "mismatch at continuous view at "
         "(0,0)");
  assert(transposed.at<float>(1) == 2.0f &&
         "mismatch at continuous view at "
         "(0,1)");
  assert(transposed.at<float>(2) == 3.0f &&
         "mismatch at continuous view at "
         "(1,0)");
  assert(transposed.at<float>(3) == 4.0f &&
         "mismatch at continuous view at "
         "(1,1)");
  assert(transposed.at<float>(4) == 5.0f &&
         "mismatch at continuous view at "
         "(2,0)");
  assert(transposed.at<float>(5) == 6.0f &&
         "mismatch at continuous view at "
         "(2,1)");

  assert(transposed.elementAt<float>({0, 0}) == 1.0f &&
         "Transpose value mismatch at (0,0)");
  assert(transposed.elementAt<float>({0, 1}) == 4.0f &&
         "Transpose value mismatch at (0,1)");
  assert(transposed.elementAt<float>({1, 0}) == 2.0f &&
         "Transpose value mismatch at (1,0)");
  assert(transposed.elementAt<float>({1, 1}) == 5.0f &&
         "Transpose value mismatch at (1,1)");
  assert(transposed.elementAt<float>({2, 0}) == 3.0f &&
         "Transpose value mismatch at (2,0)");
  assert(transposed.elementAt<float>({2, 1}) == 6.0f &&
         "Transpose value mismatch at (2,1)");
}


void test_canReshapeWithStrides()
{
  ams::SmallVector<size_t> shape = {2, 3};
  ams::SmallVector<size_t> strides = {3 * sizeof(float), sizeof(float)};
  AMSTensor tensor = AMSTensor::create(shape, strides, AMS_SINGLE, AMS_HOST);

  assert(tensor.canReshapeWithStrides({6}) &&
         "This should be a valid reshape {6} -> {2,3}");
  assert(tensor.canReshapeWithStrides({2, 3}) &&
         "This should be a valid reshape {2,3} -> {6}");
  assert((tensor.canReshapeWithStrides({3, 2})) &&
         "This should be an ivalid reshape (discontinuous shapes)");
  assert((!tensor.canReshapeWithStrides({8})) &&
         "This should be an ivalid reshape (differing number of elements)");
}

void test_reshape()
{
  {
    // Test for reshape case with contiguous new shape
    ams::SmallVector<size_t> shape = {2, 3};
    ams::SmallVector<size_t> strides = {3 * sizeof(float), sizeof(float)};
    AMSTensor tensor = AMSTensor::create(shape, strides, AMS_SINGLE, AMS_HOST);

    tensor.at<float>(0) = 1.0f;  // (0,0)
    tensor.at<float>(1) = 2.0f;  // (0,1)
    tensor.at<float>(2) = 3.0f;  // (0,2)
    tensor.at<float>(3) = 4.0f;  // (1,0)
    tensor.at<float>(4) = 5.0f;  // (1,1)
    tensor.at<float>(5) = 6.0f;  // (1,2)
    assert(tensor.canReshapeWithStrides({3, 2}) &&
           "This should be a valid reshape {2,3} -> {3,2}");
    auto reshaped = tensor.reshape({3, 2});

    assert(reshaped.elements() == tensor.elements() &&
           "Both tensors hsould have the same number of elements");
    assert(reshaped.data<uint8_t>() == tensor.data<uint8_t>() &&
           "Both tensors are equivalent, and copy should not be there");
    for (int i = 0; i < reshaped.elements(); i++) {
      assert(tensor.at<float>(i) == reshaped.at<float>(i) &&
             "Data should be identical");
    }
  }
  {
    // Test for reshape case with contiguous new shape
    ams::SmallVector<size_t> shape = {2, 3};
    ams::SmallVector<size_t> strides = {4 * sizeof(float), sizeof(float)};

    AMSTensor tensor = AMSTensor::create(shape, strides, AMS_SINGLE, AMS_HOST);
    tensor.at<float>(0) = 1.0f;  // (0,0)
    tensor.at<float>(1) = 2.0f;  // (0,1)
    tensor.at<float>(2) = 3.0f;  // (0,2)
    tensor.at<float>(3) = 4.0f;  // (1,0)
    tensor.at<float>(4) = 5.0f;  // (1,1)
    tensor.at<float>(5) = 6.0f;  // (1,2)

    auto reshaped = tensor.reshape({3, 2});

    assert(reshaped.elements() == tensor.elements() &&
           "Both tensors hsould have the same number of elements");
    assert(reshaped.data<uint8_t>() != tensor.data<uint8_t>() &&
           "Tensor should not be equivalent, and copy should not be there");
    for (int i = 0; i < reshaped.elements(); i++) {
      assert(tensor.at<float>(i) == reshaped.at<float>(i) &&
             "Data should be identical");
    }
  }
}

void testConcatenateTensors()
{
  // Test Case 1: Basic Concatenation with Compatible Shapes
  {
    AMSTensor tensor1 =
        AMSTensor::create({2, 3},
                          {3 * sizeof(float), 1 * sizeof(float)},
                          AMS_SINGLE,
                          AMS_HOST);  // Shape: [2, 3]
    AMSTensor tensor2 =
        AMSTensor::create({2, 3},
                          {3 * sizeof(float), 1 * sizeof(float)},
                          AMS_SINGLE,
                          AMS_HOST);  // Shape: [2, 3]

    tensor1.at<float>(0) = 1.0f;
    tensor1.at<float>(1) = 2.0f;
    tensor1.at<float>(2) = 3.0f;
    tensor1.at<float>(3) = 4.0f;
    tensor1.at<float>(4) = 5.0f;
    tensor1.at<float>(5) = 6.0f;

    tensor2.at<float>(0) = 7.0f;
    tensor2.at<float>(1) = 8.0f;
    tensor2.at<float>(2) = 9.0f;
    tensor2.at<float>(3) = 10.0f;
    tensor2.at<float>(4) = 11.0f;
    tensor2.at<float>(5) = 12.0f;

    ams::SmallVector<AMSTensor> iTensors;
    float* ptr1 = tensor1.data<float>();
    float* ptr2 = tensor2.data<float>();
    iTensors.push_back(std::move(tensor1));
    iTensors.push_back(std::move(tensor2));
    AMSTensor result = AMSTensor::concatenateTensors(iTensors);

    ams::SmallVector<size_t> expected_shape({2, 6});

    assert(expected_shape == result.shape() && "Expecting shape to be {2,6}");
    int elements = std::accumulate(result.shape().begin(),
                                   result.shape().end(),
                                   1,
                                   std::multiplies<size_t>());

    for (size_t i = 0; i < 2; i++) {
      for (size_t j = 0; j < 6; j++) {
        std::cout << result.elementAt<float>({i, j}) << " "
                  << iTensors[j / 3].elementAt<float>({i, j % 3}) << "\n";
        assert(result.elementAt<float>({i, j}) ==
                   iTensors[j / 3].elementAt<float>({i, j % 3}) &&
               "Values do not match");
      }
    }

    std::cout << "Test Case 1 Passed!" << std::endl;
  }

  // Test Case 2: Concatenation with Broadcasting
  {
    AMSTensor tensor1 = AMSTensor::create({2, 1},
                                          {1, 1},
                                          AMS_SINGLE,
                                          AMS_HOST);  // Shape: [2, 1]
    AMSTensor tensor2 = AMSTensor::create({2, 3},
                                          {3, 1},
                                          AMS_SINGLE,
                                          AMS_HOST);  // Shape: [2, 3]

    tensor1.at<float>(0) = 1.0f;
    tensor1.at<float>(1) = 2.0f;
    tensor2.at<float>(0) = 3.0f;
    tensor2.at<float>(1) = 4.0f;
    tensor2.at<float>(2) = 5.0f;
    tensor2.at<float>(3) = 6.0f;
    tensor2.at<float>(4) = 7.0f;
    tensor2.at<float>(5) = 8.0f;

    AMSTensor result = concatenateTensors({tensor1, tensor2});

    // Expected Shape: [2, 4] after broadcasting tensor1 to [2, 3]
    assert(result.shape() == ams::SmallVector<size_t, 4>{2, 4});
    assert(result.at<float>(0) == 1.0f && result.at<float>(1) == 1.0f);
    assert(result.at<float>(4) == 3.0f && result.at<float>(7) == 5.0f);

    std::cout << "Test Case 2 Passed!" << std::endl;
  }
  //
  //  // Test Case 3: Concatenating a Single Tensor
  //  {
  //    AMSTensor tensor = AMSTensor::create({3, 2},
  //                                         {2, 1},
  //                                         AMS_SINGLE,
  //                                         AMS_HOST);  // Shape: [3, 2]
  //    tensor.at<float>(0) = 1.0f;
  //    tensor.at<float>(1) = 2.0f;
  //    tensor.at<float>(2) = 3.0f;
  //    tensor.at<float>(3) = 4.0f;
  //    tensor.at<float>(4) = 5.0f;
  //    tensor.at<float>(5) = 6.0f;
  //
  //    AMSTensor result = concatenateTensors({tensor});
  //
  //    // Expected Shape: [3, 2] - should match original tensor
  //    assert(result.shape() == tensor.shape());
  //    assert(result.at<float>(0) == 1.0f && result.at<float>(5) == 6.0f);
  //
  //    std::cout << "Test Case 3 Passed!" << std::endl;
  //  }
  //
  //  // Test Case 4: Incompatible Shapes (Expect Exception)
  //  try {
  //    AMSTensor tensor1 = AMSTensor::create({2, 2},
  //                                          {2, 1},
  //                                          AMS_SINGLE,
  //                                          AMS_HOST);  // Shape: [2, 2]
  //    AMSTensor tensor2 = AMSTensor::create({3, 2},
  //                                          {2, 1},
  //                                          AMS_SINGLE,
  //                                          AMS_HOST);  // Shape: [3, 2]
  //
  //    concatenateTensors({tensor1, tensor2});
  //    std::cerr << "Test Case 4 Failed: Expected exception was not thrown."
  //              << std::endl;
  //  } catch (const std::invalid_argument &e) {
  //    std::cout << "Test Case 4 Passed!" << std::endl;
  //  }
}


int main(int argc, char* argv[])
{
  auto& rm = ams::ResourceManager::getInstance();
  rm.init();
  std::string func = std::string(argv[1]);
  if (func.compare("cast") == 0)
    testCastTensor();
  else if (func.compare("num_elements") == 0)
    testComputeNumElements();
  else if (func.compare("create") == 0)
    testCreate();
  else if (func.compare("view") == 0)
    testView();
  else if (func.compare("move") == 0)
    testMove();
  else if (func.compare("access") == 0)
    testDataAccess();
  else if (func.compare("conversion") == 0)
    testConversion();
  else if (func.compare("transpose") == 0)
    testTranspose();
  else if (func.compare("test-reshape") == 0)
    test_canReshapeWithStrides();
  else if (func.compare("reshape") == 0)
    test_reshape();
  else if (func.compare("concat") == 0) {
    testConcatenateTensors();
  } else {
    throw std::runtime_error("Unknown test option :'" + func + "'");
  }
  return 0;
}
