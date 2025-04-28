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
  using namespace ams;
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

  AMSCAbstrModel model_descr = AMSRegisterAbstractModel(
      "compute", ams::AMSUQPolicy::AMS_RANDOM, -1.0, "", "compute");


  /*
   * Create AMS tensors for memory blobs
   */

  // We represet both input/output as blobs of lenth 'samples', each sample as 1 element.
  SmallVector<AMSTensor> input_tensors;
  SmallVector<AMSTensor> inout_tensors;
  SmallVector<AMSTensor> output_tensors;

  input_tensors.push_back(ams::AMSTensor::view(
      input, {length, 1}, {1, 1}, ams::AMSResourceType::AMS_HOST));

  output_tensors.push_back(ams::AMSTensor::view(
      output, {length, 1}, {1, 1}, ams::AMSResourceType::AMS_HOST));

  auto Computation = [&](const ams::SmallVector<ams::AMSTensor>& ams_ins,
                         ams::SmallVector<ams::AMSTensor>& ams_inouts,
                         ams::SmallVector<ams::AMSTensor>& ams_outs) {
    ExampleAMSTensorCompute(ams_ins[0], ams_outs[0]);
  };

  Computation(input_tensors, output_tensors);
  auto sum = ComputeSum(output, length);

  std::cout << "[Example] Expected output is " << (length * (length - 1)) / 2
            << " and computed " << sum << "\n";


  delete[] input;
  delete[] output;
  ams::AMSFinalize();


  return 0;
}
