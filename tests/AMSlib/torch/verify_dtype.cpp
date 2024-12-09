#include <iostream>
#include <string>

#include "ml/surrogate.hpp"

int main(int argc, char *argv[])
{
  if (argc != 3) {
    std::cerr << "Wrong command line, expecting <precision "
                 "(float|double) <path-to-model>\n";
    return -1;
  }
  std::string precision(argv[1]);
  std::string model_path(argv[2]);
  std::cout << "Opening model under " << model_path;
  std::cout << " with precision " << precision << "\n";
  auto model = SurrogateModel::getInstance(model_path);
  if (precision.compare("single") == 0 && model->is_float()) return 0;
  if (precision.compare("double") == 0 && model->is_double()) return 0;

  if (precision.compare("single") != 0 && precision.compare("double") != 0) {
    std::cout << "AMS Surrogate does not support " << precision
              << " as a dataype\n";
  }
  return 1;
}
