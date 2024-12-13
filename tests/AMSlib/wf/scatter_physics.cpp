#include <ATen/core/TensorBody.h>
#include <ATen/ops/rand.h>
#include <c10/core/DeviceType.h>
#include <torch/types.h>

#include <algorithm>
#include <iostream>
#include <string>

#include "wf/workflow.hpp"


int main(int argc, char *argv[])
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

  if (device.compare("cuda") == 0) dev = c10::DeviceType::CUDA;

  ams::SmallVector<torch::Tensor> entireDomain;
  ams::SmallVector<torch::Tensor> computedDomain;
  for (int i = 0; i < 4; i++) {
    entireDomain.push_back(
        torch::zeros({128, 7},
                     torch::TensorOptions().dtype(DType).device(dev)));
    computedDomain.push_back(
        torch::rand({64, 7}, torch::TensorOptions().dtype(DType).device(dev)));
  }

  auto tmp = torch::arange(0, 128, torch::kInt64) % 2;
  auto Predicate = tmp.to(torch::kBool) != 0;


  ams::AMSWorkflow::ScatterPhysicOutputsToOrigDomain(computedDomain,
                                                     Predicate,
                                                     entireDomain);

  for (int i = 0; i < computedDomain.size(); i++) {
    auto cd = computedDomain[i];
    auto ed = entireDomain[i].index({Predicate});
    bool close = torch::allclose(ed, cd, 1e-5, 1e-8);

    if (!close) return 1;
  }

  return 0;
}
