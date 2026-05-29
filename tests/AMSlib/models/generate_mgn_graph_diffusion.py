#!/usr/bin/env python3
"""Train/export a tiny pure-Torch MGN-like graph diffusion surrogate.

This Stage 2 utility intentionally stays pure PyTorch. It generates synthetic
homogeneous graphs, trains a small message-passing model, exports an
AMS-wrapped TorchScript model, and writes C++ fixtures whose reference outputs
come from the exported TorchScript model.
"""

import argparse
import copy
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

try:
    import torch
    import torch.nn as nn
    from torch import Tensor
except ModuleNotFoundError as exc:
    raise SystemExit(
        "PyTorch is required for Stage 2 MGN graph diffusion generation. "
        "Install PyTorch or run this script on a Torch-enabled machine."
    ) from exc


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)
from ams_model import create_ams_homogeneous_graph_model


NODE_FEATURE_DIM = 4
EDGE_FEATURE_DIM = 4
GLOBAL_FEATURE_DIM = 1
REFERENCE_OUTPUT_DIM = 1

FIXTURE_GRAPH_SIZES = (24, 73)
FEASIBILITY_SEEDS = (70024, 70073)
FIXTURE_SEEDS = (90024, 90073)

MODEL_SEED = 314159
TRAIN_DATA_SEED = 271828
VAL_DATA_SEED = 161803

LATENT_DIM = 64
NUM_PROCESSOR_BLOCKS = 2
K_NEIGHBORS = 6
TRAIN_STEPS = 3000
VAL_GRAPHS = 32
LEARNING_RATE = 1.0e-3
WEIGHT_DECAY = 1.0e-6
PRIMARY_BASELINE_FACTOR = 0.1
SECONDARY_ABSOLUTE_MSE = 1.0e-4

CHECKPOINT_NAME = "mgn_graph_diffusion_checkpoint.pt"
MODEL_NAME = "mgn_graph_diffusion.pt"
FIXTURE_HEADER_NAME = "mgn_graph_fixtures.hpp"


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class ProcessorBlock(nn.Module):
    def __init__(self, latent_dim: int, global_dim: int):
        super().__init__()
        self.edge_mlp = MLP(latent_dim * 3 + global_dim, latent_dim, latent_dim)
        self.node_mlp = MLP(latent_dim * 2 + global_dim, latent_dim, latent_dim)

    def forward(
        self,
        node_latent: Tensor,
        edge_latent: Tensor,
        edge_index: Tensor,
        global_features: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        src = edge_index[0]
        dst = edge_index[1]

        global_edges = global_features.expand(edge_latent.shape[0], global_features.shape[1])
        edge_input = torch.cat(
            (
                edge_latent,
                node_latent.index_select(0, src),
                node_latent.index_select(0, dst),
                global_edges,
            ),
            dim=1,
        )
        edge_latent = edge_latent + self.edge_mlp(edge_input)

        aggregated = torch.zeros(
            (node_latent.shape[0], edge_latent.shape[1]),
            dtype=node_latent.dtype,
            device=node_latent.device,
        )
        aggregated = aggregated.index_add(0, dst, edge_latent)

        global_nodes = global_features.expand(node_latent.shape[0], global_features.shape[1])
        node_input = torch.cat((node_latent, aggregated, global_nodes), dim=1)
        node_latent = node_latent + self.node_mlp(node_input)
        return node_latent, edge_latent


class TinyGraphDiffusionMGN(nn.Module):
    def __init__(
        self,
        node_dim: int = NODE_FEATURE_DIM,
        edge_dim: int = EDGE_FEATURE_DIM,
        global_dim: int = GLOBAL_FEATURE_DIM,
        latent_dim: int = LATENT_DIM,
    ):
        super().__init__()
        self.node_encoder = MLP(node_dim, latent_dim, latent_dim)
        self.edge_encoder = MLP(edge_dim, latent_dim, latent_dim)
        self.processor1 = ProcessorBlock(latent_dim, global_dim)
        self.processor2 = ProcessorBlock(latent_dim, global_dim)
        self.node_decoder = MLP(latent_dim, latent_dim, REFERENCE_OUTPUT_DIM)

    def forward(self, graph: Dict[str, Tensor]) -> Dict[str, Tensor]:
        node_features = graph["node_features"]
        edge_index = graph["edge_index"].to(torch.int64)
        edge_features = graph["edge_features"]
        global_features = graph["global_features"]

        node_latent = self.node_encoder(node_features)
        edge_latent = self.edge_encoder(edge_features)
        node_latent, edge_latent = self.processor1(
            node_latent, edge_latent, edge_index, global_features
        )
        node_latent, edge_latent = self.processor2(
            node_latent, edge_latent, edge_index, global_features
        )
        delta_u = self.node_decoder(node_latent)
        return {"node:delta_u": delta_u}


def model_config() -> Dict[str, int]:
    return {
        "node_feature_dim": NODE_FEATURE_DIM,
        "edge_feature_dim": EDGE_FEATURE_DIM,
        "global_feature_dim": GLOBAL_FEATURE_DIM,
        "reference_output_dim": REFERENCE_OUTPUT_DIM,
        "latent_dim": LATENT_DIM,
        "num_processor_blocks": NUM_PROCESSOR_BLOCKS,
        "k_neighbors": K_NEIGHBORS,
    }


def make_model() -> TinyGraphDiffusionMGN:
    torch.manual_seed(MODEL_SEED)
    model = TinyGraphDiffusionMGN()
    return model.to(device=torch.device("cpu"), dtype=torch.float32)


def make_generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def _unique_directed_edges(knn: Tensor) -> Tensor:
    pairs = set()
    num_nodes = int(knn.shape[0])
    for dst in range(num_nodes):
        for n in range(int(knn.shape[1])):
            src = int(knn[dst, n].item())
            if src == dst:
                continue
            pairs.add((src, dst))
            pairs.add((dst, src))
    ordered = sorted(pairs)
    return torch.tensor(ordered, dtype=torch.int64).t().contiguous()


def generate_graph(num_nodes: int, seed: int) -> Tuple[Dict[str, Tensor], Tensor]:
    generator = make_generator(seed)
    positions = torch.rand((num_nodes, 2), generator=generator, dtype=torch.float32)
    phases = torch.rand((1, 4), generator=generator, dtype=torch.float32) * (2.0 * torch.pi)
    amps = torch.rand((1, 4), generator=generator, dtype=torch.float32) * 0.5 + 0.5
    x = positions[:, 0:1]
    y = positions[:, 1:2]
    u_raw = (
        amps[:, 0:1] * torch.sin(2.0 * torch.pi * x + phases[:, 0:1])
        + amps[:, 1:2] * torch.cos(2.0 * torch.pi * y + phases[:, 1:2])
        + amps[:, 2:3] * torch.sin(2.0 * torch.pi * (x + y) + phases[:, 2:3])
        + amps[:, 3:4] * torch.cos(2.0 * torch.pi * (x - y) + phases[:, 3:4])
    )
    u = torch.tanh(0.5 * u_raw)
    kappa = torch.rand((num_nodes, 1), generator=generator, dtype=torch.float32) + 0.5
    dt = torch.rand((1, 1), generator=generator, dtype=torch.float32) * 0.06 + 0.02

    distances = torch.cdist(positions, positions)
    masked = distances + torch.eye(num_nodes, dtype=torch.float32) * 1.0e6
    knn = torch.topk(masked, k=K_NEIGHBORS, largest=False, dim=1).indices
    edge_index = _unique_directed_edges(knn)
    src = edge_index[0]
    dst = edge_index[1]

    delta_pos = positions.index_select(0, src) - positions.index_select(0, dst)
    distance = delta_pos.norm(dim=1, keepdim=True).clamp_min(1.0e-6)
    conductivity = 0.5 * (kappa.index_select(0, src) + kappa.index_select(0, dst))
    raw_weight = conductivity / (distance + 0.05)
    incoming_sum = torch.zeros((num_nodes, 1), dtype=torch.float32)
    incoming_sum = incoming_sum.index_add(0, dst, raw_weight)
    weight = raw_weight / incoming_sum.index_select(0, dst).clamp_min(1.0e-8)

    node_features = torch.cat((positions, u, kappa), dim=1).contiguous()
    messages = weight * (u.index_select(0, src) - u.index_select(0, dst))
    edge_features = torch.cat((delta_pos, distance, messages), dim=1).contiguous()
    target = torch.zeros((num_nodes, 1), dtype=torch.float32)
    target = target.index_add(0, dst, messages)
    target = dt * target

    graph = {
        "node_features": node_features,
        "edge_index": edge_index,
        "edge_features": edge_features,
        "global_features": dt.contiguous(),
    }
    return graph, target.contiguous()


def graph_to_model_input(graph: Dict[str, Tensor]) -> Dict[str, Tensor]:
    return {
        "node_features": graph["node_features"].to(dtype=torch.float32),
        "edge_index": graph["edge_index"].to(dtype=torch.int64),
        "edge_features": graph["edge_features"].to(dtype=torch.float32),
        "global_features": graph["global_features"].to(dtype=torch.float32),
    }


def assert_close(name: str, actual: Tensor, expected: Tensor, atol: float = 1.0e-6, rtol: float = 1.0e-6) -> None:
    if not torch.allclose(actual, expected, atol=atol, rtol=rtol):
        max_diff = (actual - expected).abs().max().item()
        raise RuntimeError(f"{name} mismatch: max abs diff={max_diff}")


def run_feasibility(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print("[info] Stage 2 feasibility gate")
    print(f"[info] model_seed={MODEL_SEED}, fixture_sizes={FIXTURE_GRAPH_SIZES}, config={model_config()}")

    model = make_model().eval()
    graphs = [graph_to_model_input(generate_graph(n, seed)[0]) for n, seed in zip(FIXTURE_GRAPH_SIZES, FEASIBILITY_SEEDS)]

    with torch.no_grad():
        eager = [model(graph)["node:delta_u"].detach() for graph in graphs]

    scripted = torch.jit.script(model)
    with torch.no_grad():
        scripted_out = [scripted(graph)["node:delta_u"].detach() for graph in graphs]
    for i, (actual, expected) in enumerate(zip(scripted_out, eager)):
        assert_close(f"scripted graph {i}", actual, expected)

    script_path = out_dir / "mgn_graph_diffusion_feasibility.pt"
    scripted.save(str(script_path))
    reloaded = torch.jit.load(str(script_path)).eval()
    with torch.no_grad():
        reloaded_out = [reloaded(graph)["node:delta_u"].detach() for graph in graphs]
    for i, (actual, expected) in enumerate(zip(reloaded_out, eager)):
        assert_close(f"reloaded graph {i}", actual, expected)

    wrapped = create_ams_homogeneous_graph_model(model, torch.device("cpu"), torch.float32)
    with torch.no_grad():
        wrapped_out = [wrapped(graph)["node:delta_u"].detach() for graph in graphs]
    for i, (actual, expected) in enumerate(zip(wrapped_out, eager)):
        assert_close(f"AMS-wrapped graph {i}", actual, expected)

    print(f"[info] Feasibility passed for dynamic graph sizes {FIXTURE_GRAPH_SIZES}")


def validation_graphs() -> List[Tuple[Dict[str, Tensor], Tensor]]:
    graphs = []
    for i in range(VAL_GRAPHS):
        num_nodes = 16 + ((i * 37) % 113)
        graphs.append(generate_graph(num_nodes, VAL_DATA_SEED + i))
    return graphs


def evaluate(model: nn.Module, cases: Iterable[Tuple[Dict[str, Tensor], Tensor]]) -> Tuple[float, float, Tensor]:
    total_loss = 0.0
    total_baseline = 0.0
    total_nodes = 0
    targets = []
    model.eval()
    with torch.no_grad():
        for graph, target in cases:
            pred = model(graph_to_model_input(graph))["node:delta_u"]
            loss_sum = (pred - target).pow(2).sum().item()
            baseline_sum = target.pow(2).sum().item()
            total_loss += loss_sum
            total_baseline += baseline_sum
            total_nodes += int(target.shape[0])
            targets.append(target)
    return total_loss / total_nodes, total_baseline / total_nodes, torch.cat(targets, dim=0)


def run_train(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print("[info] Stage 2 training")
    print(f"[info] model_seed={MODEL_SEED}, train_data_seed={TRAIN_DATA_SEED}, val_data_seed={VAL_DATA_SEED}")
    print(f"[info] config={model_config()}")
    print(
        "[info] training "
        f"steps={TRAIN_STEPS}, lr={LEARNING_RATE}, weight_decay={WEIGHT_DECAY}, "
        f"primary_pass=validation_mse <= {PRIMARY_BASELINE_FACTOR} * zero_baseline_mse"
    )

    model = make_model().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=(TRAIN_STEPS // 2, (TRAIN_STEPS * 5) // 6),
        gamma=0.3,
    )
    val_cases = validation_graphs()
    best_val_mse = float("inf")
    best_zero_baseline_mse = float("inf")
    best_step = 0
    best_state_dict = copy.deepcopy(model.state_dict())

    for step in range(1, TRAIN_STEPS + 1):
        num_nodes = 16 + ((step * 53) % 113)
        graph, target = generate_graph(num_nodes, TRAIN_DATA_SEED + step)
        pred = model(graph_to_model_input(graph))["node:delta_u"]
        loss = torch.nn.functional.mse_loss(pred, target)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step == 1 or step % 200 == 0 or step == TRAIN_STEPS:
            val_mse, zero_baseline_mse, _ = evaluate(model, val_cases)
            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_zero_baseline_mse = zero_baseline_mse
                best_step = step
                best_state_dict = copy.deepcopy(model.state_dict())
            print(
                f"[info] step={step:04d} train_mse={loss.item():.8e} "
                f"val_mse={val_mse:.8e} zero_baseline_mse={zero_baseline_mse:.8e} "
                f"best_step={best_step:04d} best_val_mse={best_val_mse:.8e}"
            )

    model.load_state_dict(best_state_dict)
    val_mse, zero_baseline_mse, val_targets = evaluate(model, val_cases)
    target_mean = val_targets.mean().item()
    target_max_abs = val_targets.abs().max().item()
    target_std = val_targets.std(unbiased=False).item()
    primary_threshold = PRIMARY_BASELINE_FACTOR * zero_baseline_mse

    print(
        "[info] target_delta_u stats "
        f"mean={target_mean:.8e} max_abs={target_max_abs:.8e} std={target_std:.8e}"
    )
    print(
        f"[info] selected best_step={best_step}, validation_mse={val_mse:.8e}, "
        f"zero_baseline_mse={zero_baseline_mse:.8e}, primary_threshold={primary_threshold:.8e}"
    )
    if val_mse > SECONDARY_ABSOLUTE_MSE:
        print(
            f"[warn] validation_mse={val_mse:.8e} is above secondary diagnostic "
            f"absolute MSE {SECONDARY_ABSOLUTE_MSE:.8e}"
        )
    if val_mse > primary_threshold:
        raise RuntimeError(
            "Training failed primary Stage 2 criterion: "
            f"validation_mse={val_mse:.8e} > {primary_threshold:.8e}"
        )

    metadata = {
        "model_config": model_config(),
        "model_seed": MODEL_SEED,
        "train_data_seed": TRAIN_DATA_SEED,
        "val_data_seed": VAL_DATA_SEED,
        "fixture_graph_sizes": list(FIXTURE_GRAPH_SIZES),
        "fixture_seeds": list(FIXTURE_SEEDS),
        "train_steps": TRAIN_STEPS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "validation_mse": val_mse,
        "best_step": best_step,
        "best_validation_mse": best_val_mse,
        "best_zero_baseline_mse": best_zero_baseline_mse,
        "zero_baseline_mse": zero_baseline_mse,
        "primary_baseline_factor": PRIMARY_BASELINE_FACTOR,
        "secondary_absolute_mse": SECONDARY_ABSOLUTE_MSE,
        "target_mean": target_mean,
        "target_max_abs": target_max_abs,
        "target_std": target_std,
        "torch_version": torch.__version__,
    }
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "metadata": metadata,
    }
    checkpoint_path = out_dir / CHECKPOINT_NAME
    torch.save(checkpoint, checkpoint_path)
    print(f"[info] Saved checkpoint: {checkpoint_path}")


def format_float(value: float) -> str:
    text = f"{float(value):.9g}"
    if "e" not in text and "." not in text:
        text += ".0"
    return text + "f"


def format_array(name: str, c_type: str, values: List[float], values_per_line: int = 6) -> str:
    lines = [f"inline constexpr {c_type} {name}[] = {{"]
    for i in range(0, len(values), values_per_line):
        chunk = values[i : i + values_per_line]
        if c_type == "float":
            rendered = ", ".join(format_float(v) for v in chunk)
        else:
            rendered = ", ".join(str(int(v)) for v in chunk)
        lines.append(f"  {rendered},")
    lines.append("};")
    return "\n".join(lines)


def flatten_float(tensor: Tensor) -> List[float]:
    return [float(x) for x in tensor.detach().cpu().contiguous().view(-1).tolist()]


def flatten_int(tensor: Tensor) -> List[int]:
    return [int(x) for x in tensor.detach().cpu().contiguous().view(-1).tolist()]


def write_fixture_header(path: Path, cases: List[Dict[str, object]], metadata: Dict[str, object]) -> None:
    lines = [
        "#pragma once",
        "",
        "#include <cstddef>",
        "#include <cstdint>",
        "",
        "// Generated by tests/AMSlib/models/generate_mgn_graph_diffusion.py --mode fixtures.",
        f"// model_seed: {metadata['model_seed']}",
        f"// train_data_seed: {metadata['train_data_seed']}",
        f"// val_data_seed: {metadata['val_data_seed']}",
        f"// fixture_graph_sizes: {metadata['fixture_graph_sizes']}",
        f"// fixture_seeds: {metadata['fixture_seeds']}",
        f"// model_config: {metadata['model_config']}",
        f"// validation_mse: {metadata['validation_mse']:.9e}",
        f"// zero_baseline_mse: {metadata['zero_baseline_mse']:.9e}",
        f"// primary criterion: validation_mse <= {metadata['primary_baseline_factor']} * zero_baseline_mse",
        "",
        "namespace ams::test::mgn",
        "{",
        "",
        "struct GraphFixture {",
        "  const char* name;",
        "  std::int64_t num_nodes;",
        "  std::int64_t num_edges;",
        "  std::int64_t node_feature_dim;",
        "  std::int64_t edge_feature_dim;",
        "  std::int64_t global_feature_dim;",
        "  std::int64_t reference_output_dim;",
        "  const float* node_features;",
        "  const std::int64_t* edge_index;",
        "  const float* edge_features;",
        "  const float* global_features;",
        "  const float* reference_delta_u;",
        "};",
        "",
        "inline constexpr float kComparisonRtol = 2.0e-5f;",
        "inline constexpr float kComparisonAtol = 2.0e-5f;",
        "",
    ]

    for case in cases:
        prefix = str(case["prefix"])
        lines.append(f"// {case['name']}: N={case['num_nodes']}, E={case['num_edges']}")
        lines.append(format_array(f"{prefix}_node_features", "float", case["node_features"]))
        lines.append(format_array(f"{prefix}_edge_index", "std::int64_t", case["edge_index"], values_per_line=8))
        lines.append(format_array(f"{prefix}_edge_features", "float", case["edge_features"]))
        lines.append(format_array(f"{prefix}_global_features", "float", case["global_features"]))
        lines.append(format_array(f"{prefix}_reference_delta_u", "float", case["reference_delta_u"]))
        lines.append("")

    lines.append("inline constexpr GraphFixture kGraphFixtures[] = {")
    for case in cases:
        prefix = str(case["prefix"])
        lines.extend(
            [
                "  {",
                f"    \"{case['name']}\",",
                f"    {case['num_nodes']},",
                f"    {case['num_edges']},",
                f"    {NODE_FEATURE_DIM},",
                f"    {EDGE_FEATURE_DIM},",
                f"    {GLOBAL_FEATURE_DIM},",
                f"    {REFERENCE_OUTPUT_DIM},",
                f"    {prefix}_node_features,",
                f"    {prefix}_edge_index,",
                f"    {prefix}_edge_features,",
                f"    {prefix}_global_features,",
                f"    {prefix}_reference_delta_u,",
                "  },",
            ]
        )
    lines.extend(
        [
            "};",
            "",
            "inline constexpr std::size_t kNumGraphFixtures =",
            "    sizeof(kGraphFixtures) / sizeof(kGraphFixtures[0]);",
            "",
            "}  // namespace ams::test::mgn",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_fixtures(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / CHECKPOINT_NAME
    if not checkpoint_path.exists():
        raise RuntimeError(
            f"Missing checkpoint {checkpoint_path}. Run --mode train before --mode fixtures."
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    metadata = checkpoint["metadata"]
    print(f"[info] Loading checkpoint: {checkpoint_path}")
    print(f"[info] metadata={metadata}")

    model = make_model().eval()
    model.load_state_dict(checkpoint["model_state_dict"])
    wrapped = create_ams_homogeneous_graph_model(model, torch.device("cpu"), torch.float32)

    model_path = out_dir / MODEL_NAME
    wrapped.save(str(model_path))
    print(f"[info] Saved AMS TorchScript model: {model_path}")

    reloaded = torch.jit.load(str(model_path)).eval()
    cases: List[Dict[str, object]] = []
    with torch.no_grad():
        for index, (num_nodes, seed) in enumerate(zip(FIXTURE_GRAPH_SIZES, FIXTURE_SEEDS)):
            graph, _target = generate_graph(num_nodes, seed)
            graph = graph_to_model_input(graph)
            output = reloaded(graph)["node:delta_u"].detach().cpu().contiguous()
            case = {
                "name": f"mgn_diffusion_n{num_nodes}",
                "prefix": f"kCase{index}",
                "num_nodes": int(num_nodes),
                "num_edges": int(graph["edge_index"].shape[1]),
                "node_features": flatten_float(graph["node_features"]),
                "edge_index": flatten_int(graph["edge_index"]),
                "edge_features": flatten_float(graph["edge_features"]),
                "global_features": flatten_float(graph["global_features"]),
                "reference_delta_u": flatten_float(output),
            }
            print(
                f"[info] fixture {case['name']} N={case['num_nodes']} "
                f"E={case['num_edges']} reference_shape={tuple(output.shape)}"
            )
            cases.append(case)

    header_path = out_dir / FIXTURE_HEADER_NAME
    write_fixture_header(header_path, cases, metadata)
    print(f"[info] Saved C++ fixture header: {header_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for Stage 2 artifacts")
    parser.add_argument(
        "--mode",
        choices=("feasibility", "train", "fixtures", "all"),
        required=True,
        help="Separated Stage 2 generation mode",
    )
    args = parser.parse_args()

    if args.mode in ("feasibility", "all"):
        run_feasibility(args.out_dir)
    if args.mode in ("train", "all"):
        run_train(args.out_dir)
    if args.mode in ("fixtures", "all"):
        run_fixtures(args.out_dir)


if __name__ == "__main__":
    main()
