#include <torch/torch.h>

#include <iostream>
#include <stdexcept>
#include <string>

#include "AMS.h"
#include "ml/surrogate.hpp"

std::vector<std::int64_t> getDims(const std::string input, char delimiter)
{
  std::vector<int64_t> tokens;
  std::stringstream ss(input);
  std::string token;

  while (std::getline(ss, token, delimiter)) {
    tokens.push_back(std::stoi(token));
  }

  return tokens;
}


bool verify(torch::Tensor& input,
            torch::Tensor& output,
            torch::Tensor& predicate,
            float threshold)
{
  if (!torch::equal(input, output))
    throw std::runtime_error("Tensors are not identical");

  torch::Tensor float_tensor = predicate.to(torch::kDouble);
  // Calculate the probability (mean of the tensor)
  double probability = float_tensor.mean().item<double>();
  std::cout << "probability is " << probability << "\n";
  if (probability >= threshold + 0.1 || probability <= threshold - 0.1)
    throw std::runtime_error(
        "Expecing a probability of 0.0 in the case of threshold " +
        std::to_string(threshold) +
        "  but "
        "instead got " +
        std::to_string(probability));
  return true;
}

void test(SurrogateModel& model,
          std::vector<int64_t>& iDims,
          std::vector<int64_t>& oDims)
{
  auto model_type = model.getModelDataType();
  auto model_device = model.getModelResourceType();
  torch::Tensor input = torch::rand(iDims,
                                    torch::TensorOptions()
                                        .dtype(std::get<1>(model_type))
                                        .device(std::get<1>(model_device)));
  {
    std::cout << "Staring Test-1 with threshold of 0.0\n";
    auto [out, predicate] = model._evaluate(input, 0.0);
    verify(input, out, predicate, 0.0);
    std::cout << "SUCCESS\n";
  }
  {
    std::cout << "Staring Test-2 with threshold of 0.5\n";
    auto [out, predicate] = model._evaluate(input, 0.5);
    verify(input, out, predicate, 0.5);
    std::cout << "SUCCESS\n";
  }
  {
    std::cout << "Staring Test-3 with threshold of 1.0\n";
    auto [out, predicate] = model._evaluate(input, 1.0);
    verify(input, out, predicate, 1.0);
    std::cout << "SUCCESS\n";
  }
}

int main(int argc, char* argv[])
{
  if (argc != 5) {
    std::cerr << "Wrong command line, expecting , "
                 "<input-dim-shape (1024,2,4)> <output-dim-shape> (1024, 2, "
                 "6) <model-path> <duq_type (mean|max)>\n";
    return -1;
  }

  std::vector<int64_t> iShape(getDims(argv[1], ','));
  std::vector<int64_t> oShape(getDims(argv[2], ','));
  std::string model_path(argv[3]);
  std::string uq(argv[4]);

  auto model = SurrogateModel::getInstance(model_path);
  test(*model, iShape, oShape);
}
