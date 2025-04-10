#include <stdexcept>
#ifdef __AMS_ENABLE_MPI__
#include <mpi.h>
#endif
#include <unistd.h>

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <wf/basedb.hpp>
#include <wf/resource_manager.hpp>

#include "AMS.h"
#include "ml/surrogate.hpp"

using namespace ams;

AMSDType getDataType(char *d_type)
{
  AMSDType dType = AMSDType::AMS_DOUBLE;
  if (std::strcmp(d_type, "float") == 0) {
    dType = AMSDType::AMS_SINGLE;
  } else if (std::strcmp(d_type, "double") == 0) {
    dType = AMSDType::AMS_DOUBLE;
  } else {
    assert(false && "Unknown data type");
  }
  return dType;
}

template <typename DType>
void OrigComputation(void *cls,
                     const ams::SmallVector<ams::AMSTensor> &ams_ins,
                     ams::SmallVector<ams::AMSTensor> &ams_inouts,
                     ams::SmallVector<ams::AMSTensor> &ams_outs);

template <typename DType>
struct Problem {
  int num_inputs;
  int num_outputs;
  int multiplier;
  int scalar;
  Problem(int ni, int no) : num_inputs(ni), num_outputs(no), multiplier(100) {}

  void run(long num_elements, DType **inputs, DType **outputs, DType scalar)
  {
    std::cout << "In run " << num_inputs << " " << num_outputs << "\n";
    for (int i = 0; i < num_elements; i++) {
      DType sum = 0;
      for (int j = 0; j < num_inputs; j++) {
        sum += inputs[j][i];
      }

      for (int j = 0; j < num_outputs; j++) {
        outputs[j][i] = sum + scalar;
      }
    }
  }


  DType *initialize_inputs(DType *inputs, long length)
  {
    for (int i = 0; i < length; i++) {
      inputs[i] = static_cast<DType>(i);
    }
    return inputs;
  }

  void ams_run(AMSExecutor &wf,
               AMSResourceType resource,
               int iterations,
               int num_elements,
               int scalar)
  {
    this->scalar = scalar;
    for (int i = 0; i < iterations; i++) {
      int elements = num_elements;  // * ((DType)(rand()) / RAND_MAX) + 1;
      SmallVector<AMSTensor> input_tensors;
      SmallVector<AMSTensor> output_tensors;
      std::cout << "Num Inputs are " << num_inputs << " num outputs are "
                << num_outputs << "\n";
      // Allocate Input memory
      for (int j = 0; j < num_inputs; j++) {
        DType *data = new DType[elements];
        DType *ptr = initialize_inputs(data, elements);
        std::cout << "Input_" << j << " is " << std::hex << ptr << std::dec
                  << "\n";
        input_tensors.push_back(AMSTensor::view(
            ptr,
            SmallVector<ams::AMSTensor::IntDimType>({num_elements, 1}),
            SmallVector<ams::AMSTensor::IntDimType>({1, 1}),
            resource));
      }

      // Allocate Output memory
      for (int j = 0; j < num_outputs; j++) {
        auto tmp = new DType[elements];
        std::cout << "output " << j << " is " << std::hex << tmp << std::dec
                  << "\n";
        output_tensors.push_back(AMSTensor::view(
            initialize_inputs(tmp, elements),
            SmallVector<ams::AMSTensor::IntDimType>({num_elements, 1}),
            SmallVector<ams::AMSTensor::IntDimType>({1, 1}),
            resource));
      }

      ams::SmallVector<AMSTensor> inouts;
      AMSCExecute(wf,
                  OrigComputation<DType>,
                  (void *)this,
                  input_tensors,
                  inouts,
                  output_tensors);

      for (int i = 0; i < input_tensors.size(); i++) {
        delete input_tensors[i].data<DType>();
      }


      for (int i = 0; i < output_tensors.size(); i++) {
        delete output_tensors[i].data<DType>();
      }
    }
  }
};

template <typename DType>
void OrigComputation(void *cls,
                     const ams::SmallVector<ams::AMSTensor> &ams_ins,
                     ams::SmallVector<ams::AMSTensor> &ams_inouts,
                     ams::SmallVector<ams::AMSTensor> &ams_outs)
{
  std::cout << "Num Inputs are " << ams_ins.size() + ams_inouts.size() << "\n";
  std::cout << "Num Ouputs are " << ams_outs.size() + ams_inouts.size() << "\n";
  DType *ins[ams_ins.size() + ams_inouts.size()];
  DType *outs[ams_outs.size() + ams_inouts.size()];
  Problem<DType> *Prob = (Problem<DType> *)cls;


  // Here I can use domain knowledge (inouts is empty)
  int num_elements = ams_ins[0].shape()[0];
  for (int i = 0; i < ams_ins.size(); i++) {
    ins[i] = ams_ins[i].data<DType>();
    std::cout << "Input_" << i << " is " << std::hex << ins[i] << std::dec
              << "\n";
    if (ams_ins[i].shape()[0] != num_elements)
      throw std::runtime_error("Expected tensors to have the same shape");
  }
  for (int i = 0; i < ams_outs.size(); i++) {
    outs[i] = ams_outs[i].data<DType>();
    std::cout << "Output_" << i << " is " << std::hex << outs[i] << std::dec
              << "\n";
    if (ams_outs[i].shape()[0] != num_elements)
      throw std::runtime_error("Expected tensors to have the same shape");
  }
  Prob->run(num_elements, ins, outs, Prob->scalar);
};


int main(int argc, char **argv)
{

  if (argc != 9) {
    std::cout << "Wrong cli\n";
    std::cout << argv[0]
              << " use_device(0|1) num_inputs num_outputs "
                 "data_type(float|double)"
                 "num_iterations avg_num_values 'model-name-1' 'model-name-2'";
    return -1;
  }


  int use_device = std::atoi(argv[1]);
  int num_inputs = std::atoi(argv[2]);
  int num_outputs = std::atoi(argv[3]);
  AMSDType data_type = getDataType(argv[4]);
  int num_iterations = std::atoi(argv[5]);
  int avg_elements = std::atoi(argv[6]);
  const char *model1 = argv[7];
  const char *model2 = argv[8];
  AMSResourceType resource = AMSResourceType::AMS_HOST;
  srand(time(NULL));

  AMSCAbstrModel model_descr = AMSQueryModel(model1);
  AMSCAbstrModel model_descr1 = AMSQueryModel(model2);

  std::cout << "Running with " << num_iterations << "\n";
  AMSExecutor wf1 = AMSCreateExecutor(model_descr, 0, 1);
  AMSExecutor wf2 = AMSCreateExecutor(model_descr1, 0, 1);
  for (int i = 0; i < 10; i++) {
    if (data_type == AMSDType::AMS_SINGLE) {
      Problem<float> prob1(num_inputs, num_outputs);
      Problem<float> prob2(num_inputs + 1, num_outputs + 1);


      prob1.ams_run(wf1, resource, num_iterations, avg_elements, 0);
      prob2.ams_run(wf2, resource, num_iterations, avg_elements, 1);
    } else {
      Problem<double> prob1(num_inputs, num_outputs);
      Problem<double> prob2(num_inputs + 1, num_outputs + 1);
      prob2.ams_run(wf2, resource, num_iterations, avg_elements, 1);
      prob1.ams_run(wf1, resource, num_iterations, avg_elements, 0);
    }
  }

  return 0;
}
