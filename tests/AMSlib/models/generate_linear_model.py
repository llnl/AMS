import argparse
import math
import sys
from typing import Dict, Tuple

import numpy as np
import torch


class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize, bias=True)

    def forward(self, x):
        return self.linear(x), torch.rand(x.shape[0], 1)


class AMSModel(torch.nn.Module):
    ams_info: Dict[str, str]

    def __init__(self, model, meta: Dict[str, str]):
        super(AMSModel, self).__init__()
        self._model = model
        self.ams_info = meta

    @torch.jit.export
    def get_ams_info(self) -> Dict[str, str]:
        return self.ams_info

    def forward(self, x):
        return self._model(x)


def create_ams_model(model, trace_input, device, precision):
    if not isinstance(device, torch.device):
        raise RuntimeError(f"Expected a model to be of type torch.device instead got {type(device)}")

    if not isinstance(precision, torch.dtype):
        raise RuntimeError(f"Expected a model precision of type torch.dtype instead got {type(precision)}")

    ams_device = device.type

    if precision == torch.float32:
        ams_dtype = "float32"
    elif precision == torch.float64:
        ams_dtype = "float64"
    else:
        raise RuntimeError(f"AMS library does not support type of {precision}")

    model = model.to(device, dtype=precision)
    inp = trace_input.to(device, dtype=precision)
    ams_model = AMSModel(model, meta={"ams_type": ams_dtype, "ams_device": ams_device})

    # Trace the model
    scripted_model = torch.jit.trace(ams_model, inp)
    return scripted_model


def main(args):
    parser = argparse.ArgumentParser(description="Generate and save a scripted model.")
    parser.add_argument("precision", choices=["single", "double"], help="Model precision: 'single' or 'double'.")
    parser.add_argument("device", choices=["cpu", "gpu"], help="Device: 'cpu' or 'gpu'.")
    parser.add_argument("directory", type=str, help="Directory to save the model.")
    parser.add_argument(
        "uq", choices=["random", "duq_mean", "duq_max"], help="The UQ Type to use (this is ignored a.t.m)"
    )
    parser.add_argument("inputDim", type=int, help="The dimensions of the input data")
    parser.add_argument("outputDim", type=int, help="the dimensions of the output data")
    args = parser.parse_args()

    model = linearRegression(args.inputDim, args.outputDim)

    # Set the precision based on command-line argument
    prec = torch.float32
    if args.precision == "single":
        model = model.float()  # Set to single precision (float32)
        prec = torch.float32
    elif args.precision == "double":
        model = model.double()  # Set to double precision (float64)
        prec = torch.float64

    # Set the device based on command-line argument
    if args.device == "gpu" and torch.cuda.is_available():
        device = torch.device("cuda")
        model = model.cuda()
    else:
        device = torch.device("cpu")

    model.eval()

    x = torch.rand((1, args.inputDim), device=device, dtype=prec)
    y_before_jit = model(x)

    # Generate the file name
    file_name = f"{args.precision}_{args.device}_{args.uq}.pt"
    ams_model = create_ams_model(model, torch.randn(args.inputDim, dtype=prec), device, prec)
    file_path = f"{args.directory}/linear_{file_name}"
    print(f"Model saved to {file_path}")
    ams_model.save(file_path)


if __name__ == "__main__":
    main(sys.argv)
