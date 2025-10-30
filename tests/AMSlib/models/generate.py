import argparse
import sys
from typing import Dict, Tuple

import torch
import torch.nn as nn
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


# An example of the structure of a random model. This is how AMS expects all models. Forward returns 2 Tensors, the prediction and the uncertainty. Uncertainty in random models is defined by a random value generated in the uniform distribution [0, 1).
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


class AMSModel(nn.Module):
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


def create_ams_model(model, device, precision, trace_input=None):
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

    model.eval()
    with torch.jit.optimized_execution(True):
        model = model.to(device, dtype=precision)
        ams_model = AMSModel(model, meta={"ams_type": ams_dtype, "ams_device": ams_device})

        if trace_input is None:
            return torch.jit.script(ams_model)

        inp = trace_input.to(device, dtype=precision)
        return torch.jit.trace(ams_model, inp)


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate and save a scripted model.")
    parser.add_argument("precision", choices=["single", "double"], help="Model precision: 'single' or 'double'.")
    parser.add_argument("device", choices=["cpu", "gpu"], help="Device: 'cpu' or 'gpu'.")
    parser.add_argument("directory", type=str, help="Directory to save the model.")
    parser.add_argument("uq", choices=["random", "duq_mean", "duq_max"], help="The UQ Type to use")
    args = parser.parse_args()

    # Initialize model
    example_input = None
    prec = torch.float32
    if args.uq == "duq_mean":
        fake_uq = torch.rand(2, 8)
        # This sets odd uq to less than 0.5
        fake_uq[0, ...] *= 0.5
        # This sets even uq to larger than 0.5
        fake_uq[1, ...] = 0.5 + 0.5 * (fake_uq[1, ...])
        model = TuppleModel(8, 8, fake_uq, False)
        example_input = torch.randn(2, 8)
    elif args.uq == "duq_max":
        fake_uq = torch.rand(2, 8)
        max_val = torch.max(fake_uq, axis=1).values
        scale = 0.49 / max_val
        fake_uq *= scale.unsqueeze(0).T
        fake_uq[1, 2] = 0.51
        model = TuppleModel(8, 8, fake_uq, True)
        example_input = torch.randn(2, 8)
    elif args.uq == "random":
        model = SimpleModel(8, 8)
    else:
        sys.exit(-1)
        print("I am missing valid uq method")

    # Set the precision based on command-line argument
    if args.precision == "single":
        prec = torch.float32
    elif args.precision == "double":
        prec = torch.float64

    # Set the device based on command-line argument
    if (args.device == "gpu") and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Create example input tensor
    scripted_model = create_ams_model(model, device, prec, example_input)

    # Move model to the appropriate device

    # Generate the file name
    file_name = f"{args.precision}_{args.device}_{args.uq}.pt"
    file_path = f"{args.directory}/{file_name}"

    # Save the scripted model
    scripted_model.save(file_path)

    print(f"Model saved to {file_path}")


if __name__ == "__main__":
    main()
    sys.exit(0)
