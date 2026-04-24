#!/usr/bin/env python3
"""Generate test TorchScript models for graph surrogate execution.

This script creates minimal test models for homogeneous and heterogeneous graphs
that can be used to test the graph surrogate execution path in AMS.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn

# Add current directory to path to import ams_model helper
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)
from ams_model import (
    create_ams_homogeneous_graph_model,
    create_ams_heterogeneous_graph_model,
)


class HomogeneousGraphModel(nn.Module):
    """Simple model for homogeneous graphs.

    Accepts a Dict[str, Tensor] representing a homogeneous graph.
    Reads the 'x' field (node features) and applies a linear transformation.
    Returns (prediction, uncertainty) tuple where uncertainty is fixed low value.
    """

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(16, 8)

    def forward(
        self, graph: Dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Read 'x' field from graph (node features)
        x = graph['x']

        # Simple prediction: linear transform
        prediction = self.linear(x)

        # Low fixed uncertainty (always accept for testing)
        uncertainty = torch.full(
            (x.shape[0], 1), 0.01, dtype=x.dtype, device=x.device
        )

        return prediction, uncertainty


class HeterogeneousGraphModel(nn.Module):
    """Simple test model for heterogeneous graphs.

    This is a narrow test fixture designed to be TorchScript-scriptable.
    It expects a specific node store named "node" with an 'x' field.

    Input structure:
    {
      'node_stores': {'node': Dict[str, Tensor], ...},
      'edge_stores': {...},
      'global_store': Dict[str, Tensor]
    }

    Reads the 'x' field from the 'node' node store and applies transformation.
    Returns (prediction, uncertainty) tuple where uncertainty is fixed low value.
    """

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(16, 8)

    def forward(
        self, graph: Dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Extract node_stores with proper type recovery for TorchScript
        # The top-level dict has mixed types (node_stores/edge_stores are dicts,
        # global_store is also a dict but with different structure)
        # Use Any and isinstance to work around TorchScript limitations
        node_stores_any = graph['node_stores']

        # Recover proper type using torch.jit.isinstance
        assert torch.jit.isinstance(node_stores_any, Dict[str, Dict[str, torch.Tensor]])
        node_stores = torch.jit.annotate(Dict[str, Dict[str, torch.Tensor]], node_stores_any)

        # Use fixed node store name "node" (test fixture, not generic)
        node_store = node_stores['node']
        x = node_store['x']

        # Simple prediction
        prediction = self.linear(x)

        # Low fixed uncertainty
        uncertainty = torch.full(
            (x.shape[0], 1), 0.01, dtype=x.dtype, device=x.device
        )

        return prediction, uncertainty


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate test graph models for AMS graph surrogate execution"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory where models will be written",
    )
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    dtype = torch.float32

    # Generate homogeneous graph model
    print("[info] Generating homogeneous graph model...")
    homo_model = HomogeneousGraphModel().to(device=device, dtype=dtype)
    homo_wrapped = create_ams_homogeneous_graph_model(homo_model, device, dtype)
    homo_path = out_dir / "homogeneous_graph.pt"
    homo_wrapped.save(str(homo_path))
    print(f"[info] Saved homogeneous graph model: {homo_path}")

    # Generate heterogeneous graph model
    print("[info] Generating heterogeneous graph model...")
    hetero_model = HeterogeneousGraphModel().to(device=device, dtype=dtype)
    hetero_wrapped = create_ams_heterogeneous_graph_model(hetero_model, device, dtype)
    hetero_path = out_dir / "heterogeneous_graph.pt"
    hetero_wrapped.save(str(hetero_path))
    print(f"[info] Saved heterogeneous graph model: {hetero_path}")

    print("[info] Done generating graph models")


if __name__ == "__main__":
    main()
