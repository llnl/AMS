import torch
import os
import sys
import numpy as np
from torch.autograd import Variable
from torch import jit
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
        self.linear = torch.nn.Linear(inputSize, outputSize)
        self.linear.weight.data.fill_(0.0)
        self.linear.bias.data.fill_(0.0)
        self.fake_uq = torch.nn.Parameter(fake_uq, requires_grad=False)

    def forward(self, x):
        y = self.linear(x)
        return to_tupple(y, self.fake_uq)


def main(args):
    inputDim = int(args[1])
    outputDim = int(args[2])
    device = args[3]
    uq_type = args[4]
    precision = args[5]
    enable_cuda = True
    if device == "cuda":
        enable_cuda = True
        suffix = "_gpu"
    elif device == "cpu":
        enable_cuda = False
        suffix = "_cpu"
    prec = torch.float32
    if precision == "double":
        prec = torch.double

    fake_uq = torch.rand(2, outputDim, dtype=prec)
    if uq_type == "mean":
        # This sets odd uq to less than 0.5
        fake_uq[0, ...] *= 0.5
        # This sets even uq to larger than 0.5
        fake_uq[1, ...] = 0.5 + 0.5 * (fake_uq[1, ...])
    elif uq_type == "max":
        max_val = torch.max(fake_uq, axis=1).values
        scale = 0.49 / max_val
        fake_uq *= scale.unsqueeze(0).T
        fake_uq[0, int(outputDim / 2)] = 0.51
    else:
        print("Unknown uq type")
        sys.exit()
    if precision == "double":
        model = TuppleModel(inputDim, outputDim, fake_uq).double()
    else:
        model = TuppleModel(inputDim, outputDim, fake_uq)

    if torch.cuda.is_available() and enable_cuda:
        model = model.cuda()

    model.eval()

    data = torch.randn(1023, inputDim, dtype=prec)

    with torch.jit.optimized_execution(True):
        traced = torch.jit.trace(model, (torch.randn(inputDim, dtype=prec).to(device),))
        traced.save(f"uq_{uq_type}_{precision}{suffix}.pt")

    data = torch.zeros(2, inputDim, dtype=prec)
    inputs = Variable(data.to(device))
    model = jit.load(f"uq_{uq_type}_{precision}{suffix}.pt")
    model.eval()
    with torch.no_grad():
        print("Ouput", model(inputs))


if __name__ == "__main__":
    main(sys.argv)
