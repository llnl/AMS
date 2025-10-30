#include <ATen/core/TensorBody.h>
#include <ATen/ops/rand.h>
#include <c10/core/DeviceType.h>
#include <torch/types.h>

#include <algorithm>
#include <iostream>
#include <string>

#include "wf/workflow.hpp"

std::vector<int> generateRandomVector(int target_sum, int size)
{
  if (target_sum < size) {
    throw std::invalid_argument(
        "Target sum must be at least equal to the size of the vector.");
  }

  std::vector<int> result(
      size, 1);  // Start with each element as 1 (minimum positive integer).
  target_sum -=
      size;  // Reduce the remaining sum by the size (since all elements are 1).

  std::random_device rd;
  std::mt19937 gen(0);
  std::uniform_int_distribution<> dis(0, target_sum);

  // Generate random values and distribute the remaining sum
  for (int i = 0; i < target_sum; ++i) {
    int index = dis(gen) % size;  // Pick a random index
    ++result[index];              // Increment the value at the chosen index
  }

  // Shuffle the vector for more randomness
  std::shuffle(result.begin(), result.end(), gen);

  return result;
}


int main(int argc, char* argv[])
{

  if (argc != 4) {
    std::cout << "Wrong command line\n";
    std::cout << argv[0]
              << " <ml-type(float|double)> <ph-type (float|double)> <device>\n";
    return -1;
  }
  torch::Dtype mlDType = torch::kFloat32;
  torch::Dtype phDType = torch::kFloat32;
  torch::DeviceType dev = c10::DeviceType::CPU;

  std::string mlType(argv[1]);
  std::string phType(argv[2]);
  std::string device(argv[3]);


  if (mlType.compare("double") == 0) {
    mlDType = torch::kFloat64;
  }

  if (phType.compare("double") == 0) phDType = torch::kFloat64;

  if (device.compare("cuda") == 0)
    dev = c10::DeviceType::CUDA;
  else if (device.compare("hip") == 0)
    dev = c10::DeviceType::CUDA;

  torch::Tensor Src =
      torch::rand({32, 11}, torch::TensorOptions().dtype(mlDType).device(dev));
  auto shapes = generateRandomVector(11, 8);
  ams::SmallVector<torch::Tensor> Dest;
  auto tmp = torch::arange(0, 32, torch::kInt64) % 2;
  auto Predicate = tmp.to(torch::kBool) != 0;


  for (auto V : shapes) {
    Dest.push_back(
        torch::zeros({32, V},
                     torch::TensorOptions().dtype(phDType).device(dev)));
  }
  ams::SmallVector<torch::Tensor> subset(Dest.begin(), Dest.end() - 1);
  int offset =
      ams::AMSWorkflow::MLDomainToApplication(Src, subset, Predicate, 0);
  ams::AMSWorkflow::MLDomainToApplication(Src,
                                          {Dest[Dest.size() - 1]},
                                          Predicate,
                                          offset);

  auto Input = torch::cat(Dest, Dest[0].sizes().size() - 1);

  auto result = torch::cat(Dest, 1).to(at::TensorOptions().dtype(mlDType));
  auto inverted = ~Predicate;
  Src.index_put_({inverted}, torch::zeros({1, Src.size(1)}, Src.options()));

  bool close = torch::allclose(Src, result, 1e-5, 1e-8);
  if (close) return 0;
  return 1;
}
