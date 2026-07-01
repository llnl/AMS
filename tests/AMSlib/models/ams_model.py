from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from torch import Tensor


# ==============================================================================
# Tensor model wrapper
# ==============================================================================

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


# ==============================================================================
# Graph model wrappers
# ==============================================================================

class AMSHomogeneousGraphModel(nn.Module):
    """AMS wrapper for homogeneous graph models.

    This wrapper exposes AMS metadata methods and forwards graph dictionaries
    directly to the wrapped model without signature mismatch.
    """
    ams_info: Dict[str, str]

    def __init__(self, model: nn.Module, dtype: torch.dtype, device: torch.device):
        super().__init__()
        self._model = model

        # Convert dtype to string
        if dtype == torch.float32:
            ams_dtype = "float32"
        elif dtype == torch.float64:
            ams_dtype = "float64"
        else:
            raise RuntimeError(f"AMS library does not support dtype {dtype}")

        # Device type as string
        ams_device = device.type

        # Store in old-style format for compatibility
        self.ams_info = {"ams_type": ams_dtype, "ams_device": ams_device}

    @torch.jit.export
    def get_ams_info(self) -> Dict[str, str]:
        return self.ams_info

    def forward(self, graph: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Forward pass for homogeneous graph.

        Args:
            graph: Dict[str, Tensor] representing homogeneous graph

        Returns:
            Dict of named graph output fields
        """
        return self._model(graph)


def create_ams_homogeneous_graph_model(
    model: nn.Module,
    device: torch.device,
    precision: torch.dtype,
):
    """Create AMS-wrapped homogeneous graph model.

    Args:
        model: PyTorch model with forward(graph: Dict[str, Tensor]) -> Dict[str, Tensor]
        device: Target device
        precision: Target dtype

    Returns:
        TorchScript scripted module ready for AMS
    """
    if not isinstance(device, torch.device):
        raise RuntimeError(f"Expected device to be torch.device, got {type(device)}")

    if not isinstance(precision, torch.dtype):
        raise RuntimeError(f"Expected precision to be torch.dtype, got {type(precision)}")

    model = model.eval().to(device=device, dtype=precision)

    # Script the inner model
    inner = torch.jit.script(model)

    # Wrap in AMS metadata wrapper
    ams = AMSHomogeneousGraphModel(inner, precision, device)

    # Script the wrapper
    scripted = torch.jit.script(ams)
    return scripted


class AMSHeterogeneousGraphModel(nn.Module):
    """AMS wrapper for heterogeneous graph models.

    This wrapper exposes AMS metadata methods and forwards heterogeneous graph
    structures to the wrapped model.

    Note: The forward signature uses a generic Dict type to work around
    TorchScript limitations with deeply nested dict structures.
    """
    ams_info: Dict[str, str]

    def __init__(self, model: nn.Module, dtype: torch.dtype, device: torch.device):
        super().__init__()
        self._model = model

        # Convert dtype to string
        if dtype == torch.float32:
            ams_dtype = "float32"
        elif dtype == torch.float64:
            ams_dtype = "float64"
        else:
            raise RuntimeError(f"AMS library does not support dtype {dtype}")

        # Device type as string
        ams_device = device.type

        # Store in old-style format for compatibility
        self.ams_info = {"ams_type": ams_dtype, "ams_device": ams_device}

    @torch.jit.export
    def get_ams_info(self) -> Dict[str, str]:
        return self.ams_info

    def forward(self, graph: Dict[str, Any]) -> Dict[str, Tensor]:
        """Forward pass for heterogeneous graph.

        Args:
            graph: Nested dict with node_stores/edge_stores/global_store
                   Uses Dict[str, Any] to handle mixed value types at top level

        Returns:
            Dict of named graph output fields
        """
        return self._model(graph)


def create_ams_heterogeneous_graph_model(
    model: nn.Module,
    device: torch.device,
    precision: torch.dtype,
):
    """Create AMS-wrapped heterogeneous graph model.

    Args:
        model: PyTorch model with forward(graph: Dict[str, Any]) -> Dict[str, Tensor]
               Uses Dict[str, Any] to handle mixed top-level value types
        device: Target device
        precision: Target dtype

    Returns:
        TorchScript scripted module ready for AMS
    """
    if not isinstance(device, torch.device):
        raise RuntimeError(f"Expected device to be torch.device, got {type(device)}")

    if not isinstance(precision, torch.dtype):
        raise RuntimeError(f"Expected precision to be torch.dtype, got {type(precision)}")

    model = model.eval().to(device=device, dtype=precision)

    # Script the inner model
    inner = torch.jit.script(model)

    # Wrap in AMS metadata wrapper
    ams = AMSHeterogeneousGraphModel(inner, precision, device)

    # Script the wrapper
    scripted = torch.jit.script(ams)
    return scripted


# ==============================================================================
# Legacy model wrapper (deprecated)
# ==============================================================================

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
