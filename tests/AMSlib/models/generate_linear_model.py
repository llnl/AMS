import torch
import sys
import numpy as np
import math
import argparse


class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize, bias=True)

    def forward(self, x):
        y = self.linear(x)
        return y


def main(args):
    parser = argparse.ArgumentParser(description="Generate and save a scripted model.")
    parser.add_argument("precision", choices=["single", "double"], help="Model precision: 'single' or 'double'.")
    parser.add_argument("device", choices=["cpu", "gpu"], help="Device: 'cpu' or 'gpu'.")
    parser.add_argument("directory", type=str, help="Directory to save the model.")
    parser.add_argument("uq", choices=["random", "duq_mean", "duq_max"], help="The UQ Type to use (this is ignored a.t.m)")
    parser.add_argument("inputDim", type=int, help="The dimensions of the input data")
    parser.add_argument("outputDim", type=int, help="the dimensions of the output data")
    args = parser.parse_args()

    model = linearRegression(args.inputDim, args.outputDim)

    # Set the precision based on command-line argument
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


    with torch.jit.optimized_execution(True):
        scripted = torch.jit.script(model)
        file_path = f"{args.directory}/linear_scripted_{file_name}"
        scripted.save(file_path)
        file_path = f"{args.directory}/linear_traced_{file_name}"
        traced = torch.jit.trace(model, (torch.randn(args.inputDim, dtype=prec).to(device),))
        traced.save(file_path)


if __name__ == "__main__":
    main(sys.argv)
