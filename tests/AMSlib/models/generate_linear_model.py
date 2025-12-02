import argparse
import math
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)

import numpy as np
import torch
import torch.nn as nn
from ams_model import create_ams_model_old as create_ams_model
from torch import Tensor


# Ugly code that expands the fake_uq to the shape we need as an output
def to_tupple(y: Tensor, fake_uq: Tensor, is_max: bool) -> Tuple[Tensor, Tensor]:
    outer_dim = y.shape[0]
    fake_uq_dim = fake_uq.shape[0]
    tmp = fake_uq.clone().detach()
    additional_dims = torch.div(outer_dim, fake_uq_dim, rounding_mode="floor") + outer_dim % fake_uq_dim
    final_shape = (additional_dims * fake_uq_dim, *fake_uq.shape[1:])
    tmp = tmp.unsqueeze(0)
    my_list = [1] * len(fake_uq.shape)
    new_dims = (additional_dims, *my_list)
    tmp = tmp.repeat(new_dims)
    tmp = tmp.reshape(final_shape)
    std = tmp[: y.shape[0], ...]
    if is_max:
        max_std, _ = std.max(dim=1, keepdim=True)
        return y, max_std

    return y, std.mean(dim=1, keepdim=True)


def random_tuple(y: Tensor) -> Tuple[Tensor, Tensor]:
    return y, torch.rand(y.shape[0], 1)


# An example of a structure of D-UQ model. This is how AMS expects all models. Forward returns 2 Tensors, the prediction and the uncertainty.
class TuppleModel(torch.nn.Module):
    def __init__(self, inputSize, outputSize, fake_uq, is_max):
        super(TuppleModel, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize, False)
        self.fake_uq = torch.nn.Parameter(fake_uq, requires_grad=False)
        self._is_max = is_max
        self.initialize_weights()

    def initialize_weights(self):
        # Check if in_features == out_features for identity initialization
        if self.linear.weight.shape[0] == self.linear.weight.shape[1]:
            nn.init.eye_(self.linear.weight)  # Initialize with identity matrix
        else:
            raise ValueError("Identity initialization requires in_features == out_features")

    def forward(self, x):
        y = self.linear(x)
        return to_tupple(y, self.fake_uq, self._is_max)


class SimpleModel(nn.Module):
    def __init__(self, in_features, out_features):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(in_features, out_features, False)
        self.initialize_weights()

    def initialize_weights(self):
        # Check if in_features == out_features for identity initialization
        if self.fc.weight.shape[0] == self.fc.weight.shape[1]:
            nn.init.eye_(self.fc.weight)  # Initialize with identity matrix
        else:
            raise ValueError("Identity initialization requires in_features == out_features")

    def forward(self, x) -> Tuple[Tensor, Tensor]:
        return random_tuple(self.fc(x))


def generate_header(directory, tests):
    print(f"Writting model header under {directory}")
    with open(f"{directory}/linear_models.hpp", "w") as fd:
        fd.write('#include "./ams_test_linear_models.hpp"\n')
        fd.write("const std::vector<linear_model> simple_models = {\n")
        for t in tests:
            fd.write("{" + f'"{t[0]}", "{t[1]}", "{t[2]}", "{t[3]}", 8, 8' + "},\n")
        fd.write("};\n")


def main(args):
    parser = argparse.ArgumentParser(description="Generate and save a scripted model.")
    parser.add_argument("directory", type=str, help="Directory to save the model.")
    parser.add_argument("inputDim", type=int, help="The dimensions of the input data")
    parser.add_argument("outputDim", type=int, help="the dimensions of the output data")
    args = parser.parse_args()

    # Initialize model
    tests = []
    for precision in ["single", "double"]:
        for a_device in ["cpu", "gpu"]:
            for uq in ["random", "duq_mean", "duq_max"]:
                if a_device == "gpu" and not torch.cuda.is_available():
                    continue

                example_input = None
                if uq == "duq_mean":
                    fake_uq = torch.rand(2, 8)
                    # This sets odd uq to less than 0.5
                    fake_uq[0, ...] *= 0.5
                    # This sets even uq to larger than 0.5
                    fake_uq[1, ...] = 0.5 + 0.5 * (fake_uq[1, ...])
                    model = TuppleModel(8, 8, fake_uq, False)
                    example_input = torch.randn(2, 8)
                elif uq == "duq_max":
                    fake_uq = torch.rand(2, 8)
                    max_val = torch.max(fake_uq, axis=1).values
                    scale = 0.49 / max_val
                    fake_uq *= scale.unsqueeze(0).T
                    fake_uq[1, 2] = 0.51
                    model = TuppleModel(8, 8, fake_uq, True)
                    example_input = torch.randn(2, 8)
                elif uq == "random":
                    model = SimpleModel(8, 8)
                else:
                    sys.exit(-1)
                    print("I am missing valid uq method")

                # Set the precision based on command-line argument
                prec = torch.float32
                if precision == "single":
                    model = model.float()  # Set to single precision (float32)
                    prec = torch.float32
                elif precision == "double":
                    model = model.double()  # Set to double precision (float64)
                    prec = torch.float64

                # Set the device based on command-line argument
                if (a_device == "gpu") and torch.cuda.is_available():
                    device = torch.device("cuda")
                else:
                    device = torch.device("cpu")

                model = model.to(device)
                model.eval()

                x = torch.rand((1, args.inputDim), device=device, dtype=prec)
                y_before_jit = model(x)

                # Generate the file name
                file_name = f"{precision}_{a_device}_{uq}.pt"
                ams_model = create_ams_model(model, device, prec, example_input)
                file_path = f"{args.directory}/linear_{file_name}"
                print(f"Model saved to {file_path}")
                ams_model.save(file_path)
                tests.append((str(Path(file_path).resolve()), precision, a_device, uq))
    generate_header(args.directory, tests)


if __name__ == "__main__":
    main(sys.argv)
