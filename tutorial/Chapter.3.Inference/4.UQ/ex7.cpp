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

void ExampleAMSTensorCompute(const ams::AMSTensor& in, ams::AMSTensor& out)
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
  std::string db_path;
  std::string fn;
  std::string model_path;
  float threshold;
  ExampleArgs args;
  args.AddOption(&length,
                 "-l",
                 "--length",
                 "The size of the vectors to be initialized");
  args.AddOption(&db_path,
                 "-p",
                 "--db-path",
                 "The path of the file system database to store data to");
  args.AddOption(&fn,
                 "-f",
                 "--filename-prefix",
                 "The path of the file system database to store data to");
  args.AddOption(&model_path,
                 "-m",
                 "--model-path",
                 "The path to a torchscript model");
  args.AddOption(&threshold, "-t", "--threshold", "Model uncertainty");


  args.Parse(argc, argv);
  if (!args.Good()) {
    std::cout << "Wrong command line arguments\n";
    args.PrintOptions();
    return -1;
  }

  ams::AMSInit();

  double* input = new double[length];
  double* output = new double[length];

  AMSConfigureFSDatabase(ams::AMSDBType::AMS_HDF5, db_path.c_str());
  InitMemBlob(input, length);

  AMSCAbstrModel model_descr =
      AMSRegisterAbstractModel("compute",
                               ams::AMSUQPolicy::AMS_DELTAUQ_MEAN,
                               threshold,
                               model_path.c_str(),
                               fn.c_str());


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

  EOSLambda Computation = [&](const ams::SmallVector<ams::AMSTensor>& ams_ins,
                              ams::SmallVector<ams::AMSTensor>& ams_inouts,
                              ams::SmallVector<ams::AMSTensor>& ams_outs) {
    ExampleAMSTensorCompute(ams_ins[0], ams_outs[0]);
  };

  AMSExecutor wf = AMSCreateExecutor(model_descr, 0, 1);
  std::cout << "Calling AMS Execute\n";
  AMSExecute(wf, Computation, input_tensors, inout_tensors, output_tensors);
  std::cout << "Called AMS Execute\n";

  auto sum = ComputeSum(output, length);

  std::cout << "[Example] Expected output is " << (length * (length - 1)) / 2
            << " and computed " << sum << "\n";


  delete[] input;
  delete[] output;
  ams::AMSFinalize();


  return 0;
}
