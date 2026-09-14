"""One-round client heterogeneity diagnostic for the active PA-STFed backbone.

This is an analysis artifact, not an experiment registered in the result ledger.
It creates train loaders only and never constructs validation or test loaders.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import networkx as nx
import numpy as np
import torch

from config import configure_torch_runtime, resolve_device, set_seed
from data import archive_sha256, make_data_loader
from experiment_runtime import (
    graph_tensors,
    load_project_config,
    load_smartds,
    make_dataset,
    make_model,
    experiment_config,
)
from federated import build_client_model, train_local
from models import shared_state_dict


MODULE_PREFIXES = {
    "physical": ("physical.",),
    "temporal": ("temporal.",),
    "horizon_decoder": ("horizon_decoder.",),
    "head": ("head.",),
}


def _rank(values: np.ndarray) -> np.ndarray:
    """Average-rank ties, matching the usual Spearman definition."""

    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2:
        return None
    xr, yr = _rank(np.asarray(x, dtype=np.float64)), _rank(np.asarray(y, dtype=np.float64))
    xdev, ydev = xr - xr.mean(), yr - yr.mean()
    denominator = float(np.linalg.norm(xdev) * np.linalg.norm(ydev))
    return None if denominator == 0.0 else float(np.dot(xdev, ydev) / denominator)


def _cosine_matrix(updates: list[np.ndarray]) -> np.ndarray:
    matrix = np.zeros((len(updates), len(updates)), dtype=np.float64)
    for left, first in enumerate(updates):
        first_norm = float(np.linalg.norm(first))
        for right, second in enumerate(updates):
            second_norm = float(np.linalg.norm(second))
            denominator = first_norm * second_norm
            matrix[left, right] = (
                float(np.dot(first, second) / denominator) if denominator else 0.0
            )
    return matrix


def _matrix_stats(matrix: np.ndarray, spatial_similarity: np.ndarray) -> dict[str, object]:
    pair_indices = np.triu_indices(matrix.shape[0], k=1)
    values = matrix[pair_indices]
    spatial_values = spatial_similarity[pair_indices]
    return {
        "pair_count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
        "negative_pair_count": int(np.count_nonzero(values < 0.0)),
        "client2_mean_cosine": float(np.delete(matrix[1], 1).mean()),
        "client6_mean_cosine": float(np.delete(matrix[5], 5).mean()),
        "spearman_spatial_similarity": _spearman(
            spatial_values.tolist(), values.tolist()
        ),
    }


def _json_matrix(matrix: np.ndarray) -> list[list[float]]:
    return [[float(value) for value in row] for row in matrix]


def _client_spatial_matrices(data, partitions: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    full_graph = nx.from_numpy_array((data.adj > 0).astype(np.uint8))
    shortest = dict(nx.all_pairs_shortest_path_length(full_graph))
    hop = np.zeros((len(partitions), len(partitions)), dtype=np.float64)
    for left, first in enumerate(partitions):
        for right, second in enumerate(partitions):
            distances = [
                shortest[int(source)][int(target)]
                for source in first.tolist()
                for target in second.tolist()
            ]
            hop[left, right] = float(np.mean(distances))
    raw_similarity = 1.0 / (1.0 + hop)
    raw_off_diagonal = raw_similarity[~np.eye(len(partitions), dtype=bool)]
    low, high = float(raw_off_diagonal.min()), float(raw_off_diagonal.max())
    similarity = np.zeros_like(raw_similarity)
    if math.isclose(low, high):
        similarity.fill(1.0)
    else:
        similarity = (raw_similarity - low) / (high - low)
    np.fill_diagonal(similarity, 1.0)
    return hop, raw_similarity, similarity


def run_diagnostic(seed: int = 2026, device_name: str = "auto") -> dict[str, object]:
    set_seed(seed)
    base_cfg = load_project_config()
    cfg, _ = experiment_config(base_cfg, "stattn_moduleala_wl1_screen", seed)
    cfg["seed"] = int(seed)
    cfg["device"] = device_name
    cfg["federated"]["algorithm"] = "FedAvg"
    cfg["federated"]["mu"] = 0.0
    cfg["federated"]["rounds"] = 1
    cfg["federated"]["local_epochs"] = 1
    cfg["training"]["max_train_windows"] = 2048
    cfg["training"]["evaluate_test"] = False
    cfg["training"]["federated_batch_size"] = 512
    cfg["training"]["eval_batch_size"] = 1024
    configure_torch_runtime(cfg.get("runtime", {}))
    device = resolve_device(device_name)

    data = load_smartds(cfg)
    bounds = data.split_bounds(cfg["data"]["train_ratio"], cfg["data"]["val_ratio"])
    partitions = data.client_partitions(int(cfg["federated"]["clients"]))
    hop, raw_similarity, similarity = _client_spatial_matrices(data, partitions)

    template = make_model(cfg, len(partitions[0]), device)
    initial_template = {
        name: value.detach().cpu().clone()
        for name, value in template.named_parameters()
    }
    global_initial = shared_state_dict(template, personalized_head=False)
    train_sets = [make_dataset(data, nodes, "train", cfg) for nodes in partitions]
    graphs = [
        data.graph_view(
            nodes,
            cfg["data"]["graph"],
            int(cfg["data"].get("hop_radius", 2)),
            int(cfg["data"].get("target_knn_k", 6)),
        )
        for nodes in partitions
    ]

    module_updates: dict[str, list[np.ndarray]] = {name: [] for name in MODULE_PREFIXES}
    client_train_metrics: list[dict[str, float]] = []
    for nodes, dataset, graph in zip(partitions, train_sets, graphs):
        model = build_client_model(template, len(nodes)).to(device)
        # Force a common initialization for every shape-compatible parameter.
        with torch.no_grad():
            current = dict(model.named_parameters())
            for name, value in initial_template.items():
                if name in current and current[name].shape == value.shape:
                    current[name].copy_(value.to(device=current[name].device, dtype=current[name].dtype))
        loader = make_data_loader(
            dataset,
            int(cfg["training"]["federated_batch_size"]),
            True,
            cfg["training"],
        )
        graph_device = graph_tensors(graph, device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(cfg["training"]["learning_rate"]),
            weight_decay=float(cfg["training"]["weight_decay"]),
        )
        state, metrics = train_local(
            model,
            dataset,
            graph,
            cfg,
            global_state=global_initial,
            loader=loader,
            graph_tensors_device=graph_device,
            optimizer=optimizer,
        )
        client_train_metrics.append(metrics)
        trained = dict(model.named_parameters())
        for module, prefixes in MODULE_PREFIXES.items():
            pieces: list[np.ndarray] = []
            for name, initial in initial_template.items():
                if name.startswith(prefixes) and name in trained and trained[name].shape == initial.shape:
                    pieces.append((trained[name].detach().cpu() - initial).float().reshape(-1).numpy())
            module_updates[module].append(np.concatenate(pieces) if pieces else np.zeros(1, dtype=np.float32))

    module_results: dict[str, object] = {}
    matrices: dict[str, list[list[float]]] = {}
    for module, updates in module_updates.items():
        matrix = _cosine_matrix(updates)
        matrices[module] = _json_matrix(matrix)
        module_results[module] = _matrix_stats(matrix, similarity)

    output = {
        "analysis": "client_heterogeneity_diagnostic",
        "seed": int(seed),
        "device": str(device),
        "data_source_sha256": archive_sha256(data.source),
        "experiment_backbone": "stattn_moduleala_wl1_screen",
        "graph_mode": cfg["data"]["graph"],
        "target_knn_k": int(cfg["data"].get("target_knn_k", 6)),
        "history": int(cfg["data"]["history"]),
        "horizon": int(cfg["data"]["horizon"]),
        "loss_mode": cfg["training"].get("loss_mode"),
        "scale_source": cfg["training"].get("scale_source"),
        "train_windows_per_client": [int(len(dataset)) for dataset in train_sets],
        "split_bounds": {"train_end": int(bounds.train_end), "val_end": int(bounds.val_end), "total": int(bounds.total)},
        "test_evaluated": False,
        "client_partitions": [nodes.tolist() for nodes in partitions],
        "client_node_counts": [int(len(nodes)) for nodes in partitions],
        "spatial_relation": {
            "average_shortest_topology_hop": _json_matrix(hop),
            "raw_similarity": _json_matrix(raw_similarity),
            "normalized_similarity": _json_matrix(similarity),
        "similarity_definition": "raw=1/(1+average ordered-pair official-tree hop); raw off-diagonal min-max normalized; diagonal=1",
            "source": "official 273-node graph and frozen topology partition",
        },
        "local_training": {
            "rounds": 1,
            "local_epochs": 1,
            "max_train_windows": 2048,
            "metrics": client_train_metrics,
        },
        "update_definition": "local trained named_parameter tensor minus common shape-compatible initial tensor; no validation/test data",
        "update_cosine": matrices,
        "module_statistics": module_results,
        "module_cosine_correlations": {
            "physical_vs_temporal": _spearman(
                [matrices["physical"][i][j] for i in range(8) for j in range(i + 1, 8)],
                [matrices["temporal"][i][j] for i in range(8) for j in range(i + 1, 8)],
            ),
            "pairwise_upper_triangle": {
                left + "_vs_" + right: _spearman(
                    [matrices[left][i][j] for i in range(8) for j in range(i + 1, 8)],
                    [matrices[right][i][j] for i in range(8) for j in range(i + 1, 8)],
                )
                for left in MODULE_PREFIXES
                for right in MODULE_PREFIXES
                if left < right
            },
        },
    }
    output_path = Path(__file__).resolve().parents[1] / "results" / f"client_heterogeneity_diagnostic_seed{seed}.json"
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output_path), "clients": len(partitions), "train_windows": output["train_windows_per_client"], "test_evaluated": False}, ensure_ascii=False, indent=2))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="One-round client heterogeneity diagnostic")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    run_diagnostic(args.seed, args.device)


if __name__ == "__main__":
    main()
