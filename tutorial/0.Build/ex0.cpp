#include <AMS.h>

#include <iostream>

#include "common.hpp"

void InitMemBlob(double* ptr, int size)
{
  for (int i = 0; i < size; i++) {
    ptr[i] = i;
  }
}

void ExampleCompute(double* in, double* out, int size)
{
  for (int i = 0; i < size; i++) {
    out[i] = in[i];
  }
}


double ComputeSum(double* out, int size)
{
  double sum = 0;
  for (int i = 0; i < size; i++) {
    sum += out[i];
  }
  return sum;
}

int main(int argc, char* argv[])
{
  int length;
  ExampleArgs args;
  args.AddOption(&length,
                 "-l",
                 "--length",
                 "The size of the vectors to be initialized");
  args.Parse(argc, argv);
  if (!args.Good()) {
    std::cout << "Wrong command line arguments\n";
    args.PrintOptions();
    return -1;
  }

  double* input = new double[length];
  double* output = new double[length];

  InitMemBlob(input, length);
  ExampleCompute(input, output, length);
  auto sum = ComputeSum(output, length);

  std::cout << "[Example] Expected output is " << (length * (length - 1)) / 2
            << " and computed " << sum << "\n";


  delete[] input;
  delete[] output;


  return 0;
}
