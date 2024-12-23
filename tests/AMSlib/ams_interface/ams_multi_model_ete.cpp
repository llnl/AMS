#ifdef __AMS_ENABLE_MPI__
#include <mpi.h>
#endif
#include <unistd.h>

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <ml/uq.hpp>
#include <umpire/Umpire.hpp>
#include <umpire/strategy/QuickPool.hpp>
#include <wf/basedb.hpp>
#include <wf/resource_manager.hpp>

#include "AMS.h"
#include "wf/debug.h"

void createUmpirePool(std::string parent_name, std::string pool_name)
{
  auto &rm = umpire::ResourceManager::getInstance();
  auto alloc_resource = rm.makeAllocator<umpire::strategy::QuickPool, true>(
      pool_name, rm.getAllocator(parent_name));
}


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
struct Problem {
  int num_inputs;
  int num_outputs;
  int multiplier;
  Problem(int ni, int no) : num_inputs(ni), num_outputs(no), multiplier(100) {}

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
    auto &rm = umpire::ResourceManager::getInstance();

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

      for (int j = 0; j < num_outputs; j++) {
        delete[] outputs[j];
      }


      for (int j = 0; j < num_inputs; j++) {
        delete[] inputs[j];
      }
    }
  }
};

void callBackDouble(void *cls, long elements, void **inputs, void **outputs)
{
  std::cout << "Called the double model\n";
  static_cast<Problem<double> *>(cls)->run(elements,
                                           (double **)(inputs),
                                           (double **)(outputs));
}


void callBackSingle(void *cls, long elements, void **inputs, void **outputs)
{
  std::cout << "Called the single model\n";
  static_cast<Problem<float> *>(cls)->run(elements,
                                          (float **)(inputs),
                                          (float **)(outputs));
}


int main(int argc, char **argv)
{
  if (argc != 15) {
    std::cout << "Wrong cli\n";
    std::cout << argv[0]
              << " use_device(0|1) num_inputs num_outputs model_path "
                 "data_type(float|double) uq_policy(random|deltaUQ "
                 "(mean)|deltaUQ (max)) threshold(0) "
                 "num_iterations avg_num_values db_type(none|csv|hdf5) "
                 "db_path(path to existing path to store data)";
    return -1;
  }


  int use_device = std::atoi(argv[1]);
  int num_inputs_1 = std::atoi(argv[2]);
  int num_outputs_1 = std::atoi(argv[3]);
  char *model_path_1 = argv[4];
  AMSDType data_type = getDataType(argv[5]);
  std::string uq_name = std::string(argv[6]);
  const AMSUQPolicy uq_policy = BaseUQ::UQPolicyFromStr(uq_name);
  float threshold = std::atof(argv[7]);
  int num_iterations = std::atoi(argv[8]);
  int avg_elements = std::atoi(argv[9]);
  std::string db_type_str = std::string(argv[10]);
  std::string fs_path = std::string(argv[11]);
  int num_inputs_2 = std::atoi(argv[12]);
  int num_outputs_2 = std::atoi(argv[13]);
  char *model_path_2 = argv[14];
  AMSDBType db_type = ams::db::getDBType(db_type_str);
  AMSResourceType resource = AMSResourceType::AMS_HOST;
  srand(time(NULL));

  AMSConfigureFSDatabase(db_type, fs_path.c_str());

  assert((uq_policy == AMSUQPolicy::AMS_DELTAUQ_MAX ||
          uq_policy == AMSUQPolicy::AMS_DELTAUQ_MEAN ||
          uq_policy == AMSUQPolicy::AMS_RANDOM) &&
         "Test only supports duq models");

  createUmpirePool("HOST", "TEST_HOST");
  AMSSetAllocator(AMSResourceType::AMS_HOST, "TEST_HOST");

  AMSCAbstrModel model_descr_1 = AMSRegisterAbstractModel(
      "test_1", uq_policy, threshold, model_path_1, nullptr, "test_1", -1);

  AMSCAbstrModel model_descr_2 = AMSRegisterAbstractModel(
      "test_2", uq_policy, threshold, model_path_2, nullptr, "test_2", -1);


  if (data_type == AMSDType::AMS_SINGLE) {
    Problem<float> prob1(num_inputs_1, num_outputs_1);
    Problem<float> prob2(num_inputs_2, num_outputs_2);

    AMSExecutor wf_1 = AMSCreateExecutor(model_descr_1,
                                         AMSDType::AMS_SINGLE,
                                         resource,
                                         (AMSPhysicFn)callBackSingle,
                                         0,
                                         1);

    AMSExecutor wf_2 = AMSCreateExecutor(model_descr_2,
                                         AMSDType::AMS_SINGLE,
                                         resource,
                                         (AMSPhysicFn)callBackSingle,
                                         0,
                                         1);

    for (int i = 0; i < num_iterations; i++) {
      size_t elems =
          (static_cast<double>(rand()) / RAND_MAX) * 2 * avg_elements -
          avg_elements + avg_elements;
      prob1.ams_run(wf_1, resource, 1, elems);
      size_t elems1 =
          (static_cast<double>(rand()) / RAND_MAX) * 2 * avg_elements -
          avg_elements + avg_elements;
      prob2.ams_run(wf_2, resource, 1, elems1);
    }
  } else {
    Problem<double> prob1(num_inputs_1, num_outputs_1);
    Problem<double> prob2(num_inputs_2, num_outputs_2);

    AMSExecutor wf_1 = AMSCreateExecutor(model_descr_1,
                                         AMSDType::AMS_DOUBLE,
                                         resource,
                                         (AMSPhysicFn)callBackDouble,
                                         0,
                                         1);

    AMSExecutor wf_2 = AMSCreateExecutor(model_descr_2,
                                         AMSDType::AMS_DOUBLE,
                                         resource,
                                         (AMSPhysicFn)callBackDouble,
                                         0,
                                         1);

    for (int i = 0; i < num_iterations; i++) {
      size_t elems = avg_elements;
      //          (static_cast<double>(rand()) / RAND_MAX) * 2 * avg_elements -
      //          avg_elements + avg_elements;
      prob1.ams_run(wf_1, resource, 1, elems);
      size_t elems1 = avg_elements;
      //          (static_cast<double>(rand()) / RAND_MAX) * 2 * avg_elements -
      //          avg_elements + avg_elements;
      prob2.ams_run(wf_2, resource, 1, elems1);
    }
  }

  return 0;
}
