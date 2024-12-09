#include <iostream>
#include <string>

#include "ml/surrogate.hpp"


int main(int argc, char *argv[])
{
  if (argc != 3) {
    std::cerr << "Wrong command line, expecting <device (cpu|gpu)>  "
                 "<path-to-model>\n";
    return -1;
  }
  std::string device(argv[1]);
  std::string model_path(argv[2]);
  std::cout << "Opening model under " << model_path;
  std::cout << " on device " << device << "\n";

  auto model = SurrogateModel::getInstance(model_path);
  if (device.compare("gpu") == 0 && model->is_gpu()) {
    std::cout << "SUCCESS: " << "Model will execute on device\n";
    return 0;
  }
  if (device.compare("cpu") == 0 && model->is_cpu()) {
    std::cout << "SUCCESS: " << "Model will execute on host\n";
    return 0;
  }

  if (device.compare("cpu") != 0 && device.compare("gpu") != 0)
    std::cout << "AMS Surrogate does not support " << device
              << " as a execution device\n";
  return 1;
}
