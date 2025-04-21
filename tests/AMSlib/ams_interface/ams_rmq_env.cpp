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

#include "../utils.hpp"
#include "AMS.h"
#include "wf/debug.h"

template <typename DType>
struct Problem {
  int num_inputs;
  int num_outputs;
  Problem(int ni, int no) : num_inputs(ni), num_outputs(no) {}

  void run(long num_elements, DType **inputs, DType **outputs)
  {
    for (int i = 0; i < num_elements; i++) {
      DType sum = 0;
      for (int j = 0; j < num_inputs; j++) {
        sum += inputs[j][i];
      }

      for (int j = 0; j < num_outputs; j++) {
        outputs[j][i] = sum;
      }
    }
  }


  const DType *initialize_inputs(DType *inputs, long length)
  {
    for (int i = 0; i < length; i++) {
      inputs[i] = static_cast<DType>(i);
    }
    return inputs;
  }

  void ams_run(AMSExecutor &wf,
               AMSResourceType resource,
               int iterations,
               int num_elements)
  {
    for (int i = 0; i < iterations; i++) {
      int elements = num_elements;  // * ((DType)(rand()) / RAND_MAX) + 1;
      std::vector<const DType *> inputs;
      std::vector<DType *> outputs;

      // Allocate Input memory
      for (int j = 0; j < num_inputs; j++) {
        DType *data = new DType[elements];
        inputs.push_back(initialize_inputs(data, elements));
      }

      // Allocate Output memory
      for (int j = 0; j < num_outputs; j++) {
        outputs.push_back(new DType[elements]);
      }

      AMSExecute(wf,
                 (void *)this,
                 elements,
                 reinterpret_cast<const void **>(inputs.data()),
                 reinterpret_cast<void **>(outputs.data()),
                 inputs.size(),
                 outputs.size());

      for (int i = 0; i < num_outputs; i++) {
        delete[] outputs[i];
        outputs[i] = nullptr;
      }


      for (int i = 0; i < num_inputs; i++) {
        delete[] inputs[i];
        inputs[i] = nullptr;
      }
    }
  }
};

void callBackDouble(void *cls, long elements, void **inputs, void **outputs)
{
  std::cout << "Called the double precision model\n";
  static_cast<Problem<double> *>(cls)->run(elements,
                                           (double **)(inputs),
                                           (double **)(outputs));
}


void callBackSingle(void *cls, long elements, void **inputs, void **outputs)
{
  std::cout << "Called the single precision model\n";
  static_cast<Problem<float> *>(cls)->run(elements,
                                          (float **)(inputs),
                                          (float **)(outputs));
}


int main(int argc, char **argv)
{

  if (argc != 7) {
    std::cout << "Wrong cli\n";
    std::cout << argv[0]
              << " use_device(0|1) num_inputs num_outputs "
                 "data_type(float|double) "
                 "num_iterations num_elements"
              << std::endl;
    return -1;
  }

  installSignals();
  AMSInit();

  int use_device = std::atoi(argv[1]);
  int num_inputs = std::atoi(argv[2]);
  int num_outputs = std::atoi(argv[3]);
  AMSDType data_type = getDataType(argv[4]);
  int num_iterations = std::atoi(argv[5]);
  int num_elements = std::atoi(argv[6]);
  AMSResourceType resource = AMSResourceType::AMS_HOST;
  srand(time(NULL));


  AMSCAbstrModel ams_model = AMSRegisterAbstractModel("rmq_db_no_model",
                                                      AMSUQPolicy::AMS_RANDOM,
                                                      0.5,
                                                      "",
                                                      "",
                                                      "rmq_db_no_model",
                                                      1);

  if (data_type == AMSDType::AMS_SINGLE) {
    Problem<float> prob(num_inputs, num_outputs);
    AMSExecutor wf = AMSCreateExecutor(ams_model,
                                       AMSDType::AMS_SINGLE,
                                       resource,
                                       (AMSPhysicFn)callBackSingle,
                                       0,
                                       1);

    prob.ams_run(wf, resource, num_iterations, num_elements);
  } else {
    Problem<double> prob(num_inputs, num_outputs);
    AMSExecutor wf = AMSCreateExecutor(ams_model,
                                       AMSDType::AMS_DOUBLE,
                                       resource,
                                       (AMSPhysicFn)callBackDouble,
                                       0,
                                       1);
    prob.ams_run(wf, resource, num_iterations, num_elements);
  }

  std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  AMSFinalize();
  return 0;
}
