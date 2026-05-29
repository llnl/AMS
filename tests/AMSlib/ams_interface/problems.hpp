#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

#include "AMS.h"
using namespace ams;

template <typename DType>
struct Problem2D {
  int num_inputs;
  int num_inouts;
  int num_outputs;
  int multiplier;
  Problem2D(int ni, int nio, int no)
      : num_inputs(ni), num_inouts(nio), num_outputs(no), multiplier(100)
  {
  }

  void run(long num_elements,
           const DType* input1,
           const DType* input2,
           DType* inout,
           DType* out1,
           DType* out2,
           int num_inout)
  {
    for (int i = 0; i < num_elements; i++) {
      DType sum = input1[i] + input2[i];
      for (int j = 0; j < num_inout; j++)
        sum += inout[i * num_inout + j];

      out1[i] = sum;
      out2[i] = sum;
      for (int j = 0; j < num_inout; j++)
        inout[i * num_inout + j] = sum;
    }
  }


  DType* initialize_inputs(DType* inputs, long length)
  {
    for (int i = 0; i < length; i++) {
      inputs[i] = static_cast<DType>(i);
    }
    return inputs;
  }

  DType* initialize_inout(DType* inputs, long length, int elements_per_row)
  {
    for (int i = 0; i < length; i++) {
      for (int j = 0; j < elements_per_row; j++) {
        inputs[i * elements_per_row + j] = static_cast<DType>(i);
      }
    }
    return inputs;
  }


  void ams_run(AMSExecutor& wf,
               AMSResourceType resource,
               int iterations,
               int num_elements)
  {

    for (int i = 0; i < iterations; i++) {
      int elements = num_elements;  // * ((DType)(rand()) / RAND_MAX) + 1;
      SmallVector<AMSTensor> input_tensors;
      SmallVector<AMSTensor> inout_tensors;
      SmallVector<AMSTensor> output_tensors;

      std::vector<std::vector<DType>> input_buffers(num_inputs,
                                                    std::vector<DType>(
                                                        elements));
      for (int j = 0; j < num_inputs; j++) {
        input_tensors.push_back(AMSTensor::view(
            initialize_inputs(input_buffers[j].data(), elements),
            SmallVector<ams::AMSTensor::IntDimType>({num_elements, 1}),
            SmallVector<ams::AMSTensor::IntDimType>({1, 1}),
            resource));
      }

      std::vector<DType> inout_data(elements * num_inouts);
      inout_tensors.push_back(AMSTensor::view(
          initialize_inout(inout_data.data(), elements, num_inouts),
          SmallVector<ams::AMSTensor::IntDimType>({num_elements, num_inouts}),
          SmallVector<ams::AMSTensor::IntDimType>({num_inouts, 1}),
          resource));

      std::vector<std::vector<DType>> output_buffers(num_outputs,
                                                     std::vector<DType>(
                                                         elements));
      for (int j = 0; j < num_outputs; j++) {
        output_tensors.push_back(AMSTensor::view(
            initialize_inputs(output_buffers[j].data(), elements),
            SmallVector<ams::AMSTensor::IntDimType>({num_elements, 1}),
            SmallVector<ams::AMSTensor::IntDimType>({1, 1}),
            resource));
      }

      DomainLambda OrigComputation = [&](const ams::SmallVector<ams::AMSTensor>&
                                             ams_ins,
                                         ams::SmallVector<ams::AMSTensor>&
                                             ams_inouts,
                                         ams::SmallVector<ams::AMSTensor>&
                                             ams_outs) {
        std::vector<const DType*> ins(num_inputs);
        std::vector<DType*> outs(num_outputs);
        DType* inout;

        if (ams_inouts.size() != 1) {
          throw std::runtime_error("Expecting a single inout tensor");
        }

        if (ams_inouts[0].shape()[1] != num_inouts)
          throw std::runtime_error("Inout shape should be 'num_inout'");

        inout = ams_inouts[0].data<DType>();
        int num_elements = ams_inouts[0].shape()[0];
        for (int i = 0; i < num_inputs; i++) {
          ins[i] = ams_ins[i].data<DType>();
          if (ams_ins[i].shape()[0] != num_elements)
            throw std::runtime_error("Expected tensors to have the same shape");
        }
        for (int i = 0; i < num_outputs; i++) {
          outs[i] = ams_outs[i].data<DType>();
          if (ams_outs[i].shape()[0] != num_elements)
            throw std::runtime_error("Expected tensors to have the same shape");
        }
        run(num_elements,
            ins[0],
            ins[1],
            inout,
            outs[0],
            outs[1],
            // I have access to inouts, because we captured everything by reference.
            num_inouts);
      };

      AMSExecute(
          wf, OrigComputation, input_tensors, inout_tensors, output_tensors);
    }
  }
};


template <typename DType>
struct Problem {
  int num_inputs;
  int num_outputs;
  int multiplier;
  Problem(int ni, int no) : num_inputs(ni), num_outputs(no), multiplier(100) {}

  void run(long num_elements, const DType* const* inputs, DType** outputs)
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


  DType* initialize_inputs(DType* inputs, long length)
  {
    for (int i = 0; i < length; i++) {
      inputs[i] = static_cast<DType>(i);
    }
    return inputs;
  }

  void ams_run(AMSExecutor& wf,
               AMSResourceType resource,
               int iterations,
               int num_elements)
  {
    for (int i = 0; i < iterations; i++) {
      int elements = num_elements;  // * ((DType)(rand()) / RAND_MAX) + 1;
      SmallVector<AMSTensor> input_tensors;
      SmallVector<AMSTensor> output_tensors;

      std::vector<std::vector<DType>> input_buffers(num_inputs,
                                                    std::vector<DType>(
                                                        elements));
      for (int j = 0; j < num_inputs; j++) {
        input_tensors.push_back(AMSTensor::view(
            initialize_inputs(input_buffers[j].data(), elements),
            SmallVector<ams::AMSTensor::IntDimType>({num_elements, 1}),
            SmallVector<ams::AMSTensor::IntDimType>({1, 1}),
            resource));
      }

      std::vector<std::vector<DType>> output_buffers(num_outputs,
                                                     std::vector<DType>(
                                                         elements));
      for (int j = 0; j < num_outputs; j++) {
        output_tensors.push_back(AMSTensor::view(
            initialize_inputs(output_buffers[j].data(), elements),
            SmallVector<ams::AMSTensor::IntDimType>({num_elements, 1}),
            SmallVector<ams::AMSTensor::IntDimType>({1, 1}),
            resource));
      }

      DomainLambda OrigComputation =
          [&](const ams::SmallVector<ams::AMSTensor>& ams_ins,
              ams::SmallVector<ams::AMSTensor>& /*ams_inouts*/,
              ams::SmallVector<ams::AMSTensor>& ams_outs) {
            std::vector<const DType*> ins(num_inputs);
            std::vector<DType*> outs(num_outputs);
            if (static_cast<std::size_t>(num_inputs) != ams_ins.size())
              throw std::runtime_error(
                  "Expecting dimensions of inputs to remain the same");
            else if (static_cast<std::size_t>(num_outputs) != ams_outs.size())
              throw std::runtime_error(
                  "Expecting dimensions of outputs to remain the same");

            // Here I can use domain knowledge (inouts is empty)
            int num_elements = ams_ins[0].shape()[0];
            for (int i = 0; i < num_inputs; i++) {
              ins[i] = ams_ins[i].data<DType>();
              if (ams_ins[i].shape()[0] != num_elements)
                throw std::runtime_error(
                    "Expected tensors to have the same shape");
            }
            for (int i = 0; i < num_outputs; i++) {
              outs[i] = ams_outs[i].data<DType>();
              if (ams_outs[i].shape()[0] != num_elements)
                throw std::runtime_error(
                    "Expected tensors to have the same shape");
            }
            run(num_elements, ins.data(), outs.data());
          };

      ams::SmallVector<AMSTensor> inouts;
      AMSExecute(wf, OrigComputation, input_tensors, inouts, output_tensors);
    }
  }
};


template <typename DType>
struct ProblemBroadcast {
  int num_inputs;
  int num_outputs;
  int multiplier;
  ProblemBroadcast(int ni, int no)
      : num_inputs(ni), num_outputs(no), multiplier(100)
  {
  }

  void run(long num_elements,
           const DType* const* inputs,
           DType** outputs,
           DType constant)
  {
    for (int i = 0; i < num_elements; i++) {
      DType sum = constant;
      for (int j = 0; j < num_inputs - 1; j++) {
        sum += inputs[j][i];
      }

      for (int j = 0; j < num_outputs; j++) {
        outputs[j][i] = sum;
      }
    }
  }


  DType* initialize_inputs(DType* inputs, long length)
  {
    for (int i = 0; i < length; i++) {
      inputs[i] = static_cast<DType>(i);
    }
    return inputs;
  }

  void ams_run(AMSExecutor& wf,
               AMSResourceType resource,
               int iterations,
               int num_elements)
  {
    DType value = 1.0;
    for (int i = 0; i < iterations; i++) {
      int elements = num_elements;  // * ((DType)(rand()) / RAND_MAX) + 1;
      SmallVector<AMSTensor> input_tensors;
      SmallVector<AMSTensor> output_tensors;

      std::vector<std::vector<DType>> input_buffers(num_inputs - 1,
                                                    std::vector<DType>(
                                                        elements));
      for (int j = 0; j < num_inputs - 1; j++) {
        input_tensors.push_back(AMSTensor::view(
            initialize_inputs(input_buffers[j].data(), elements),
            SmallVector<ams::AMSTensor::IntDimType>({num_elements, 1}),
            SmallVector<ams::AMSTensor::IntDimType>({1, 1}),
            resource));
      }
      value = num_inputs - 1;
      input_tensors.push_back(AMSTensor::view(
          &value,
          SmallVector<ams::AMSTensor::IntDimType>({num_elements, 1}),
          SmallVector<ams::AMSTensor::IntDimType>({0, 0}),
          resource));


      std::vector<std::vector<DType>> output_buffers(num_outputs,
                                                     std::vector<DType>(
                                                         elements));
      for (int j = 0; j < num_outputs; j++) {
        output_tensors.push_back(AMSTensor::view(
            initialize_inputs(output_buffers[j].data(), elements),
            SmallVector<ams::AMSTensor::IntDimType>({num_elements, 1}),
            SmallVector<ams::AMSTensor::IntDimType>({1, 1}),
            resource));
      }

      DomainLambda OrigComputation =
          [&](const ams::SmallVector<ams::AMSTensor>& ams_ins,
              ams::SmallVector<ams::AMSTensor>& /*ams_inouts*/,
              ams::SmallVector<ams::AMSTensor>& ams_outs) {
            std::vector<const DType*> ins(num_inputs - 1);
            std::vector<DType*> outs(num_outputs);
            if (static_cast<std::size_t>(num_inputs) != ams_ins.size())
              throw std::runtime_error(
                  "Expecting dimensions of inputs to remain the same");
            else if (static_cast<std::size_t>(num_outputs) != ams_outs.size())
              throw std::runtime_error(
                  "Expecting dimensions of outputs to remain the same");

            // Here I can use domain knowledge (inouts is empty)
            int num_elements = ams_ins[0].shape()[0];
            for (int i = 0; i < num_inputs - 1; i++) {
              ins[i] = ams_ins[i].data<DType>();
              if (ams_ins[i].shape()[0] != num_elements)
                throw std::runtime_error(
                    "Expected tensors to have the same shape");
            }
            for (int i = 0; i < num_outputs; i++) {
              outs[i] = ams_outs[i].data<DType>();
              if (ams_outs[i].shape()[0] != num_elements)
                throw std::runtime_error(
                    "Expected tensors to have the same shape");
            }
            run(num_elements,
                ins.data(),
                outs.data(),
                *ams_ins[num_inputs - 1].data<DType>());
          };

      ams::SmallVector<AMSTensor> inouts;
      AMSExecute(wf, OrigComputation, input_tensors, inouts, output_tensors);
    }
  }
};
