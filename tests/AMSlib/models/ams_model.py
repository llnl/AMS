from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch import Tensor

class AMSModel(nn.Module):
    _ams_dtype: torch.dtype
    _ams_device: torch.device

    def __init__(self, model: nn.Module, dtype: torch.dtype, device: torch.device):
        super().__init__()
        self._model = model
        self._ams_dtype = dtype
        self._ams_device = device

    @torch.jit.export
    def get_ams_dtype(self) -> torch.dtype:
        return self._ams_dtype

    @torch.jit.export
    def get_ams_device(self) -> torch.device:
        return self._ams_device

    def forward(self, x: Tensor):
        return self._model(x)


def create_ams_model(
    model: nn.Module,
    device: torch.device,
    precision: torch.dtype,
    trace_input: Tensor | None = None,
):
    if not isinstance(device, torch.device):
        raise RuntimeError(f"Expected device to be torch.device, got {type(device)}")

    if not isinstance(precision, torch.dtype):
        raise RuntimeError(f"Expected precision to be torch.dtype, got {type(precision)}")

    model = model.eval().to(device=device, dtype=precision)

    # inner TS module: traced OR scripted
    if trace_input is not None:
        inp = trace_input.to(device=device, dtype=precision)
        inner = torch.jit.trace(model, inp)
    else:
        inner = torch.jit.script(model)

    # wrap inner TS module
    ams = AMSModel(inner, precision, device)

    # IMPORTANT: script the wrapper so get_ams_* become TorchScript methods
    scripted = torch.jit.script(ams)
    return scripted


class AMSModelOld(nn.Module):
    ams_info: Dict[str, str]

    def __init__(self, model, meta: Dict[str, str]):
        super(AMSModelOld, self).__init__()
        self._model = model
        self.ams_info = meta

    @torch.jit.export
    def get_ams_info(self) -> Dict[str, str]:
        return self.ams_info

    def forward(self, x):
        return self._model(x)


def create_ams_model_old(model, device, precision, trace_input=None):
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
        ams_model = AMSModelOld(model, meta={"ams_type": ams_dtype, "ams_device": ams_device})

        if trace_input is None:
            return torch.jit.script(ams_model)

        inp = trace_input.to(device, dtype=precision)
        return torch.jit.trace(ams_model, inp)
