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
    """Deterministic message-passing model for homogeneous graphs."""

    def forward(
        self, graph: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        node_features = graph["node_features"]
        edge_index = graph["edge_index"].to(torch.int64)
        edge_features = graph["edge_features"]

        src = edge_index[0]
        dst = edge_index[1]

        messages = edge_features[:, 0:1] * node_features.index_select(0, src)[:, 0:1]
        aggregated = torch.zeros(
            (node_features.shape[0], 1),
            dtype=node_features.dtype,
            device=node_features.device,
        )
        aggregated = aggregated.index_add(0, dst, messages)

        prediction = node_features[:, 0:1] + aggregated
        if "global_features" in graph:
            prediction = prediction + graph["global_features"][0] * 0.0

        return {"node:prediction": prediction}


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
    Returns named node fields.
    """

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(16, 8)

    def forward(
        self, graph: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
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

        return {"node:node:prediction": prediction}


class MalformedHomogeneousGraphModel(nn.Module):
    def forward(
        self, graph: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        node_features = graph["node_features"]
        return {"bad:prediction": node_features[:, 0:1]}


class WrongShapeHomogeneousGraphModel(nn.Module):
    def forward(
        self, graph: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        node_features = graph["node_features"]
        return {"node:prediction": node_features[0:1, 0:1]}


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

    print("[info] Generating malformed homogeneous graph model...")
    bad_key_model = MalformedHomogeneousGraphModel().to(device=device, dtype=dtype)
    bad_key_wrapped = create_ams_homogeneous_graph_model(
        bad_key_model, device, dtype
    )
    bad_key_path = out_dir / "homogeneous_graph_bad_key.pt"
    bad_key_wrapped.save(str(bad_key_path))
    print(f"[info] Saved malformed graph model: {bad_key_path}")

    print("[info] Generating wrong-shape homogeneous graph model...")
    bad_shape_model = WrongShapeHomogeneousGraphModel().to(
        device=device, dtype=dtype
    )
    bad_shape_wrapped = create_ams_homogeneous_graph_model(
        bad_shape_model, device, dtype
    )
    bad_shape_path = out_dir / "homogeneous_graph_bad_shape.pt"
    bad_shape_wrapped.save(str(bad_shape_path))
    print(f"[info] Saved wrong-shape graph model: {bad_shape_path}")

    print("[info] Done generating graph models")


if __name__ == "__main__":
    main()
