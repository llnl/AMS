#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>

#include "AMS.h"

using real_t = double;
using namespace ams;

void eval(real_t *density,
          real_t *e_mass,
          real_t *qc,
          real_t deltaTime,
          real_t **mat,
          int NumComps,
          int NumZones)
{
  // Density is a 0->vector.
  real_t *Dense = density;
  real_t *eMass = e_mass;
  real_t *QC = qc;

  for (int j = 0; j < NumZones; j++) {
    real_t A = Dense[j];  // Reactant A
    for (int i = 0; i < NumComps; i++) {
      real_t k = mat[j][i];  // Reaction rate constant
      real_t reaction_rate = k * A * deltaTime;
      Dense[j] -= reaction_rate;
      eMass[j] = reaction_rate * k;
      QC[j] += reaction_rate;
    }
  }
}

void eval_ams(AMSExecutor &wf,
              real_t *density,
              real_t *e_mass,
              real_t *qc,
              real_t deltaTime,
              real_t **mat,
              int NumComps,
              int NumZones)
{
  // Density is a 0->vector.
  SmallVector<AMSTensor> input_tensors;
  SmallVector<AMSTensor> inout_tensors;
  SmallVector<AMSTensor> output_tensors;
  // Density is inout.
  inout_tensors.push_back(
      AMSTensor::view(density,
                      SmallVector<ams::AMSTensor::IntDimType>({NumZones, 1}),
                      SmallVector<ams::AMSTensor::IntDimType>({1, 1}),
                      AMSResourceType::AMS_HOST));
  // QC is inout
  inout_tensors.push_back(
      AMSTensor::view(qc,
                      SmallVector<ams::AMSTensor::IntDimType>({NumZones, 1}),
                      SmallVector<ams::AMSTensor::IntDimType>({1, 1}),
                      AMSResourceType::AMS_HOST));

  input_tensors.push_back(AMSTensor::view(
      &mat[0][0],
      SmallVector<ams::AMSTensor::IntDimType>({NumZones, NumComps}),
      SmallVector<ams::AMSTensor::IntDimType>({NumComps, 1}),
      AMSResourceType::AMS_HOST));

  // deltaTime is a scalar input, I BROADCAST it now with 0 strides.
  input_tensors.push_back(
      AMSTensor::view(&deltaTime,
                      SmallVector<ams::AMSTensor::IntDimType>({NumZones, 1}),
                      SmallVector<ams::AMSTensor::IntDimType>({0, 0}),
                      AMSResourceType::AMS_HOST));

  // e_mass is just an output
  output_tensors.push_back(
      AMSTensor::view(e_mass,
                      SmallVector<ams::AMSTensor::IntDimType>({NumZones, 1}),
                      SmallVector<ams::AMSTensor::IntDimType>({1, 1}),
                      AMSResourceType::AMS_HOST));


  EOSLambda OrigComputation =
      [&](const ams::SmallVector<ams::AMSTensor> &ams_ins,
          ams::SmallVector<ams::AMSTensor> &ams_inouts,
          ams::SmallVector<ams::AMSTensor> &ams_outs) {
        int prunedZones = ams_ins[0].shape()[0];
        std::cout << "Pruned are " << prunedZones << "\n";
        real_t *pruned_mat[prunedZones];
        // The 2D data of materials are unnder a c_vector.
        real_t *c_mats = ams_ins[0].data<real_t>();
        // We need this as eval requires a c like 2D vector
        for (int i = 0; i < prunedZones; i++) {
          pruned_mat[i] = &c_mats[i * ams_ins[0].shape()[1]];
        }
        eval(ams_inouts[0].data<real_t>(),// density was the first entry in inout
             ams_outs[0].data<real_t>(),
             ams_inouts[1].data<real_t>(), // qc was the second entry in inout
             *ams_ins[1].data<real_t>(),
             pruned_mat,
             NumComps,
             prunedZones);
      };
  // After I call this, I expect the database to have the following order:
  // input_Data: **input_tensors, **inout_tensors 
  // input_Data: **output_tensors, **inout_tensors 
  // In this example the database will have the following:
  // Input: |Mat_0|Mat_1|dt|density|qc| Output : |e_mass|density|qc|
  AMSExecute(wf, OrigComputation, input_tensors, inout_tensors, output_tensors);
}

void initializeRandom(real_t *data,
                      size_t NumElements,
                      real_t minVal = 0.0,
                      real_t maxVal = 1.0)
{
  std::random_device rd;
  std::mt19937 gen(0);
  std::uniform_real_distribution<real_t> dist(minVal, maxVal);
  for (size_t i = 0; i < NumElements; i++) {
    data[i] = dist(gen);
  }
}


int main(int argc, char *argv[])
{
  int numZones = std::atoi(argv[1]);
  int numComps = std::atoi(argv[2]);
  real_t *actualDensity = new real_t[numZones];
  initializeRandom(actualDensity, numZones);
  real_t *eMass = new real_t[numZones];
  initializeRandom(eMass, numZones);
  real_t *qc = new real_t[numZones];
  initializeRandom(qc, numZones);
  real_t dt = 1.0;
  ams::AMSConfigureFSDatabase(ams::AMSDBType::AMS_HDF5, "./");
  ams::AMSCAbstrModel model_descr = AMSRegisterAbstractModel(
      "test", ams::AMSUQPolicy::AMS_RANDOM, 0.0, nullptr, "test");
  ams::AMSExecutor wf = ams::AMSCreateExecutor(model_descr, 0, 1);

  // Here I am uncertain if materials are NumComps or NumZones.
  // NOTE: Materials may or may not be contineous on the outer dimension.
  // We take a worst case scenario here, in which data are non contineous.
  real_t *materials[numZones];
  real_t *tmpData = new real_t[numZones * numComps];
  for (int i = 0; i < numZones; i++) {
    materials[i] = &tmpData[i * numComps];
    initializeRandom(materials[i], numComps);
  }

#if 0
  // THIS WE DO NOT SUPPORT CAUSE the materials data will be a non contineous vector
  real_t *materials[numZones];
  for (int i = 0; i < numZones; i++) {
    materials[i] = new real_t[numComps];
    initializeRandom(materials[i], numComps);
  }
#endif
  std::cout << std::fixed << std::setprecision(2);

  std::cout << "Before\n";
  for (int i = 0; i < numZones; i++) {
    std::cout << "Dense: " << actualDensity[i] << " eMass:" << eMass[i]
              << " QC:" << qc[i];
    for (int j = 0; j < numComps; j++) {
      std::cout << " Mat_" << j << " " << materials[i][j];
    }
    std::cout << "\n";
  }


  eval_ams(wf, actualDensity, eMass, qc, dt, materials, numComps, numZones);

  std::cout << "After\n";
  for (int i = 0; i < numZones; i++) {
    std::cout << "Dense: " << actualDensity[i] << " eMass:" << eMass[i]
              << " QC:" << qc[i];
    for (int j = 0; j < numComps; j++) {
      std::cout << " Mat_" << j << " " << materials[i][j];
    }
    std::cout << "\n";
  }
}
