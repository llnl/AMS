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

void ExampleAMSTensorCompute(ams::AMSTensor& in, ams::AMSTensor& out)
{
  ExampleCompute(in.data<double>(), out.data<double>(), in.shape()[0]);
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

  ams::AMSInit();

  double* input = new double[length];
  double* output = new double[length];

  InitMemBlob(input, length);

  /*
   * Create AMS tensors for memory blobs
   */

  // We represet both input/output as blobs of lenth 'samples', each sample as 1 element.
  ams::AMSTensor InT = ams::AMSTensor::view(input,
                                            {length, 1},
                                            {1, 1},
                                            ams::AMSResourceType::AMS_HOST);

  ams::AMSTensor OutT = ams::AMSTensor::view(output,
                                             {length, 1},
                                             {1, 1},
                                             ams::AMSResourceType::AMS_HOST);


  ExampleAMSTensorCompute(InT, OutT);
  auto sum = ComputeSum(output, length);

  std::cout << "[Example] Expected output is " << (length * (length - 1)) / 2
            << " and computed " << sum << "\n";


  delete[] input;
  delete[] output;
  ams::AMSFinalize();


  return 0;
}
