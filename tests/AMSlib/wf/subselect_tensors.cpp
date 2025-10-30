
#include <ATen/core/TensorBody.h>
#include <ATen/ops/rand.h>
#include <c10/core/DeviceType.h>
#include <torch/types.h>

#include <algorithm>
#include <iostream>
#include <string>

#include "wf/workflow.hpp"


int main(int argc, char* argv[])
{

  if (argc != 3) {
    std::cout << "Wrong command line\n";
    std::cout << argv[0] << " <type (float|double)> <device>\n";
    return -1;
  }
  torch::Dtype DType = torch::kFloat32;
  torch::DeviceType dev = c10::DeviceType::CPU;

  std::string Type(argv[1]);
  std::string device(argv[2]);


  if (Type.compare("double") == 0) {
    DType = torch::kFloat64;
  }

  if (Type.compare("double") == 0) DType = torch::kFloat64;

  if (device.compare("cuda") == 0)
    dev = c10::DeviceType::CUDA;
  else if (device.compare("hip") == 0)
    dev = c10::DeviceType::CUDA;

  ams::SmallVector<torch::Tensor> vectors;
  for (int i = 0; i < 4; i++)
    vectors.push_back(
        torch::rand({32, 11}, torch::TensorOptions().dtype(DType).device(dev)));

  auto tmp = torch::arange(0, 32, torch::kInt64) % 2;
  auto Predicate = tmp.to(torch::kBool) != 0;

  auto subselectedTensors =
      ams::AMSWorkflow::subSelectTensors(vectors, Predicate);

  for (int i = 0; i < vectors.size(); i++) {
    auto sb = subselectedTensors[i];
    auto orig = vectors[i];
    orig = orig.index({Predicate});
    bool close = torch::allclose(orig, sb, 1e-5, 1e-8);

    if (!close) return 1;
  }

  return 0;
}
