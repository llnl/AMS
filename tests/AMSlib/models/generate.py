import torch
import sys
import torch.nn as nn
import argparse
from torch import Tensor
from typing import Tuple


# Ugly code that expands the fake_uq to the shape we need as an output
def to_tupple(y: Tensor, fake_uq: Tensor) -> Tuple[Tensor, Tensor]:
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
    return y, std


class TuppleModel(torch.nn.Module):
    def __init__(self, inputSize, outputSize, fake_uq):
        super(TuppleModel, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize, False)
        self.fake_uq = torch.nn.Parameter(fake_uq, requires_grad=False)
        self.initialize_weights()

    def initialize_weights(self):
        # Check if in_features == out_features for identity initialization
        if self.linear.weight.shape[0] == self.linear.weight.shape[1]:
            nn.init.eye_(self.linear.weight)  # Initialize with identity matrix
        else:
            raise ValueError("Identity initialization requires in_features == out_features")

    def forward(self, x):
        y = self.linear(x)
        return to_tupple(y, self.fake_uq)


# Define a simple model
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

    def forward(self, x):
        return self.fc(x)


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate and save a scripted model.")
    parser.add_argument("precision", choices=["single", "double"], help="Model precision: 'single' or 'double'.")
    parser.add_argument("device", choices=["cpu", "gpu"], help="Device: 'cpu' or 'gpu'.")
    parser.add_argument("directory", type=str, help="Directory to save the model.")
    parser.add_argument("uq", choices=["random", "duq_mean", "duq_max"], help="The UQ Type to use")
    args = parser.parse_args()

    # Initialize model
    if args.uq == "duq_mean":
        fake_uq = torch.rand(2, 8)
        # This sets odd uq to less than 0.5
        fake_uq[0, ...] *= 0.5
        # This sets even uq to larger than 0.5
        fake_uq[1, ...] = 0.5 + 0.5 * (fake_uq[1, ...])
        model = TuppleModel(8, 8, fake_uq)
    elif args.uq == "duq_max":
        fake_uq = torch.rand(2, 8)
        max_val = torch.max(fake_uq, axis=1).values
        scale = 0.49 / max_val
        fake_uq *= scale.unsqueeze(0).T
        fake_uq[1, 2] = 0.51
        model = TuppleModel(8, 8, fake_uq)
    elif args.uq == "random":
        model = SimpleModel(8, 8)
    else:
        sys.exit(-1)
        print("I am missing valid uq method")

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
    else:
        device = torch.device("cpu")

    # Move model to the appropriate device
    model.to(device)

    # Create example input tensor
    example_input = torch.randn(2, 8, device=device, dtype=prec)

    # Trace the model
    scripted_model = torch.jit.trace(model, example_input)

    # Generate the file name
    file_name = f"{args.precision}_{args.device}_{args.uq}.pt"
    file_path = f"{args.directory}/{file_name}"

    # Save the scripted model
    scripted_model.save(file_path)

    print(f"Model saved to {file_path}")


if __name__ == "__main__":
    main()
    sys.exit(0)
