"""Recover a centralized development result from an already completed checkpoint.

This utility deliberately performs inference only.  It never constructs a test
dataset and never calls the training entry point.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data import archive_sha256, make_data_loader
from run import (
    _dataset_alignment_metadata,
    _hash_array,
    _make_train_diagnostic_loader,
    _selection_metric,
    evaluate,
    evaluate_wape_blocks,
    graph_tensors,
    load_smartds,
    make_dataset,
    make_model,
    model_diagnostics,
    multiscale_patch_metadata,
    output_path,
)
from config import autocast_context, configure_torch_runtime, resolve_device, set_seed


def recover(checkpoint_path: Path) -> Path:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("model_state"), dict):
        raise ValueError("checkpoint does not contain a model_state mapping")
    cfg = payload.get("config")
    if not isinstance(cfg, dict):
        raise ValueError("checkpoint does not contain its training config")
    if bool(cfg.get("training", {}).get("evaluate_test", False)):
        raise AssertionError("checkpoint config requests test evaluation")
    nodes = np.asarray(payload.get("active_nodes"), dtype=np.int64)
    if nodes.ndim != 1 or nodes.size == 0:
        raise ValueError("checkpoint active_nodes is missing or invalid")
    state = payload["model_state"]
    if not all(torch.is_tensor(value) and torch.isfinite(value).all() for value in state.values()):
        raise ValueError("checkpoint contains non-finite model parameters")

    set_seed(int(cfg["seed"]))
    configure_torch_runtime(cfg.get("runtime", {}))
    device = resolve_device(cfg.get("device", "auto"))
    data = load_smartds(cfg)
    bounds = data.split_bounds(cfg["data"]["train_ratio"], cfg["data"]["val_ratio"])
    train_set = make_dataset(data, nodes, "train", cfg)
    val_set = make_dataset(data, nodes, "val", cfg)
    graph = data.graph_view(
        nodes,
        cfg["data"]["graph"],
        int(cfg["data"].get("hop_radius", 2)),
        int(cfg["data"].get("target_knn_k", 6)),
    )
    model = make_model(cfg, len(nodes), device)
    model.load_state_dict(state, strict=True)
    model.eval()
    eval_batch_size = int(cfg["training"].get("eval_batch_size", 64))
    val_loader = make_data_loader(val_set, eval_batch_size, False, cfg["training"])
    train_loader = _make_train_diagnostic_loader(
        train_set, eval_batch_size, cfg["training"]
    )
    validation = evaluate(model, val_set, graph, device, eval_batch_size, cfg["training"], val_loader)
    train_metrics = evaluate(model, train_set, graph, device, eval_batch_size, cfg["training"], train_loader)
    source_sha256 = archive_sha256(data.source)
    global_target_adjacency, _, _, _ = data.topology_knn_graph(int(cfg["data"].get("target_knn_k", 6)))
    global_edges = int(np.count_nonzero(np.triu(global_target_adjacency > 0, k=1)))
    output = output_path(cfg, "centralized_result.json")
    result = {
        "mode": "centralized",
        "code_revision": cfg.get("code_revision"),
        "experiment_name": cfg.get("experiment_name"),
        "config_signature": __import__("run").config_signature(cfg),
        "seed": int(cfg["seed"]),
        "device": str(device),
        "data_source_sha256": source_sha256,
        "node_indices_sha256": _hash_array(nodes),
        "split_bounds": {"train_end": int(bounds.train_end), "val_end": int(bounds.val_end), "total": int(bounds.total)},
        "architecture": str(cfg["model"].get("architecture", "pa_stfed")),
        "temporal_architecture": str(cfg["model"].get("temporal_architecture", "transformer")),
        "tcn_config": None,
        "functional_graph_mode": str(cfg["model"].get("functional_graph_mode", "static")),
        "dynamic_context_steps": int(cfg["model"].get("dynamic_context_steps", 12)),
        "dynamic_gain_init": float(cfg["model"].get("dynamic_gain_init", 0.0)),
        "multiscale_patch_config": multiscale_patch_metadata(model),
        "loss_mode": str(cfg["training"].get("loss_mode", "charbonnier")),
        "scale_source": cfg["training"].get("scale_source"),
        "feeder_loss_weight": float(cfg["training"].get("feeder_loss_weight", 0.0)),
        "graph_mode": str(cfg["data"].get("graph", "topology_knn")),
        "target_knn_k": int(cfg["data"].get("target_knn_k", 6)),
        "global_target_graph_edges": global_edges,
        "graph_effective_nodes": int(len(graph.node_indices)),
        "graph_effective_undirected_edges": int(np.count_nonzero(np.triu(graph.adjacency > 0, k=1))),
        "graph_inferred_bridge_metadata": int(len(graph.bridge_edges)),
        "model_ablation": {key: bool(cfg["model"].get(key, True)) for key in ("use_physical", "use_functional", "use_spatial_gate", "use_temporal_gate", "use_residual_anchor")},
        "best_epoch": None,
        "selection_metric": _selection_metric(cfg["training"]),
        "best_selection_score": float(validation[_selection_metric(cfg["training"])]),
        "best_train": train_metrics,
        "best_validation": validation,
        "test_evaluated": False,
        "test": None,
        "evaluation_metadata": {
            "source_sha256": source_sha256,
            "history": int(val_set.history),
            "horizon": int(val_set.horizon),
            "node_count": int(len(nodes)),
            "node_indices_sha256": _hash_array(nodes),
            "validation_alignment": _dataset_alignment_metadata(val_set, source_sha256),
            "test_alignment": None,
            "train_windows_total": int(len(train_set)),
            "validation_windows": int(len(val_set)),
            "test_windows": None,
            "mape_floor": "nodewise 0.01 * training-split mean absolute load; values below floor excluded",
            "metric_scale": "all reported error metrics are computed after nodewise inverse normalization",
        },
        "bootstrap_block_windows": int(cfg["training"].get("bootstrap_block_windows", 96)),
        "validation_wape_blocks": evaluate_wape_blocks(model, val_set, graph, device, eval_batch_size, cfg["training"], val_loader, block_windows=int(cfg["training"].get("bootstrap_block_windows", 96)), aggregation_level="micro_global"),
        "test_wape_blocks": None,
        "diagnostics": model_diagnostics(model, val_set, graph, device, eval_batch_size, cfg["training"], val_loader),
        "history": [],
        "recovered_from_checkpoint": True,
        "checkpoint_path": str(checkpoint_path),
        "recovery_note": "Validation-only recovery; best epoch history was not persisted in the checkpoint.",
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"result": str(output), "validation": validation, "patch": result["multiscale_patch_config"], "test_evaluated": False}, ensure_ascii=False, indent=2))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    recover(args.checkpoint)
