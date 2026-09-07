"""Centralized training runner."""

from __future__ import annotations

from experiment_runtime import *  # noqa: F401,F403
from experiment_runtime import (
    _assert_active_nodes_train_stable,
    _batch_size,
    _dataset_alignment_metadata,
    _make_scheduler,
    _make_train_diagnostic_loader,
    _non_blocking,
    _selection_metric,
    _hash_array,
    aggregate_wape_blocks,
    evaluate,
    evaluate_wape_blocks,
    graph_tensors,
    make_dataset,
    make_model,
    model_diagnostics,
    output_path,
    load_smartds,
    config_signature,
 )

def centralized(cfg: dict, device: torch.device) -> dict:
    """训练集中式对照模型，并按预先声明的验证指标保存最佳参数。"""

    data = load_smartds(cfg)
    source_sha256 = archive_sha256(data.source)
    bounds = data.split_bounds(cfg["data"]["train_ratio"], cfg["data"]["val_ratio"])
    _assert_active_nodes_train_stable(data, bounds.train_end)
    nodes = data.active_indices
    train_set = make_dataset(data, nodes, "train", cfg)
    val_set = make_dataset(data, nodes, "val", cfg)
    loss_mode = str(cfg["training"].get("loss_mode", "charbonnier")).lower()
    if loss_mode not in {"charbonnier", "scale_aware", "scale_aware_l1", "multilevel"}:
        raise ValueError(f"Unsupported training.loss_mode={loss_mode!r}")
    configured_scale_source = cfg["training"].get("scale_source", "train_iqr")
    if loss_mode in {"scale_aware", "scale_aware_l1", "multilevel"} and configured_scale_source != "train_iqr":
        raise ValueError(
            "scale-aware losses require training.loss scale_source='train_iqr'"
        )
    scale_source = "train_iqr" if loss_mode in {"scale_aware", "scale_aware_l1", "multilevel"} else None
    feeder_loss_weight = float(
        cfg["training"].get("feeder_loss_weight", 0.1 if loss_mode == "multilevel" else 0.0)
    )
    if loss_mode != "multilevel" and feeder_loss_weight != 0.0:
        raise ValueError("feeder_loss_weight is only valid with loss_mode=multilevel")
    # 测试集默认关闭；只有 centralized_test 等显式任务才打开，防止配置缺省
    # 时误把测试集用于普通验证矩阵。
    evaluate_test = bool(cfg["training"].get("evaluate_test", False))
    test_set = make_dataset(data, nodes, "test", cfg) if evaluate_test else None
    graph = data.graph_view(
        nodes,
        cfg["data"]["graph"],
        int(cfg["data"].get("hop_radius", 2)),
        int(cfg["data"].get("target_knn_k", 6)),
    )
    model = make_model(cfg, len(nodes), device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    adjacency, edge_features = graph_tensors(graph, device)
    batch_size = _batch_size(cfg["training"], "centralized")
    eval_batch_size = _batch_size(cfg["training"], "centralized", evaluation=True)
    train_loader = make_data_loader(train_set, batch_size, True, cfg["training"])
    val_loader = make_data_loader(
        val_set, eval_batch_size, False, cfg["training"]
    )
    test_loader = (
        make_data_loader(test_set, eval_batch_size, False, cfg["training"])
        if evaluate_test and test_set is not None
        else None
    )
    train_diagnostic_loader = _make_train_diagnostic_loader(
        train_set, eval_batch_size, cfg["training"]
    )
    transfer_non_blocking = _non_blocking(cfg["training"], device)
    scaler = make_grad_scaler(cfg["training"], device)
    scheduler = _make_scheduler(optimizer, cfg["training"])
    patience = int(cfg["training"].get("patience", 5))
    min_delta = float(cfg["training"].get("early_stop_min_delta", 0.0))
    selection_metric = _selection_metric(cfg["training"])

    history: list[dict] = []
    best_score = float("inf")
    best_epoch = 0
    stale_epochs = 0
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, int(cfg["training"]["epochs"]) + 1):
        model.train()
        loss_sum = torch.zeros((), device=device)
        batch_count = 0
        for inputs, targets in train_loader:
            target_device = targets.to(device, non_blocking=transfer_non_blocking)
            with autocast_context(cfg["training"], device):
                prediction = model(
                    inputs.to(device, non_blocking=transfer_non_blocking),
                    adjacency,
                    edge_features,
                )["prediction"]
                if loss_mode == "charbonnier":
                    loss = charbonnier_loss(
                        prediction,
                        target_device,
                        float(cfg["model"]["robust_kappa"]),
                    )
                elif loss_mode == "scale_aware_l1":
                    loss = scale_aware_l1_loss(
                        prediction,
                        target_device,
                        train_set.scale,
                    )
                else:
                    loss = scale_aware_charbonnier_loss(
                        prediction,
                        target_device,
                        float(cfg["model"]["robust_kappa"]),
                        train_set.scale,
                        feeder_loss_weight=feeder_loss_weight,
                    )
            if bool(cfg["training"].get("smoke_checks", False)):
                if prediction.shape != target_device.shape:
                    raise RuntimeError(
                        f"smoke shape mismatch: prediction={prediction.shape}, "
                        f"target={target_device.shape}"
                    )
                if not torch.isfinite(prediction).all() or not torch.isfinite(loss):
                    raise RuntimeError("smoke detected NaN/Inf in prediction or loss")
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(cfg["training"]["grad_clip_norm"]),
            )
            if bool(cfg["training"].get("smoke_checks", False)) and not torch.isfinite(
                gradient_norm
            ):
                raise RuntimeError("smoke detected NaN/Inf in gradients")
            scaler.step(optimizer)
            scaler.update()
            # 不在每个 batch 调用 .cpu()/.item()，避免强制 CUDA 同步。
            loss_sum += loss.detach()
            batch_count += 1

        validation = evaluate(
            model,
            val_set,
            graph,
            device,
            eval_batch_size,
            cfg["training"],
            val_loader,
        )
        record = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_loss_normalized": float(
                (loss_sum / max(batch_count, 1)).cpu()
            ),
            "validation": validation,
        }
        history.append(record)
        print(
            f"[Centralized] epoch={epoch:03d} "
            f"loss={record['train_loss_normalized']:.6f} "
            f"val_RMSE={validation['rmse']:.6f} "
            f"val_WAPE={validation['wape']:.2f}% "
            f"val_MAPE={validation['mape']:.2f}%"
        )

        selection_score = float(validation[selection_metric])
        if not np.isfinite(selection_score):
            raise RuntimeError(
                f"Validation {selection_metric} is non-finite at epoch {epoch}"
            )
        if selection_score < best_score - min_delta:
            best_score = selection_score
            best_epoch = epoch
            stale_epochs = 0
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        else:
            stale_epochs += 1
        if scheduler is not None:
            scheduler.step(selection_score)
        if stale_epochs >= patience:
            print(
                f"[Centralized] early stop at epoch {epoch}; "
                f"selection={selection_metric}"
            )
            break

    if best_state is None:
        raise RuntimeError("Centralized training produced no valid checkpoint")
    model.load_state_dict(best_state)

    # 训练指标只在恢复最佳 checkpoint 后计算一次，避免每轮重复推理拖慢训练。
    best_train = evaluate(
        model,
        train_set,
        graph,
        device,
        eval_batch_size,
        cfg["training"],
        train_diagnostic_loader,
    )
    bootstrap_block_windows = int(
        cfg["training"].get("bootstrap_block_windows", 96)
    )
    validation_wape_blocks = evaluate_wape_blocks(
        model,
        val_set,
        graph,
        device,
        eval_batch_size,
        cfg["training"],
        val_loader,
        block_windows=bootstrap_block_windows,
        aggregation_level="micro_global",
    )
    test_metrics = (
        evaluate(
            model,
            test_set,
            graph,
            device,
            eval_batch_size,
            cfg["training"],
            test_loader,
        )
        if evaluate_test and test_set is not None and test_loader is not None
        else None
    )
    test_wape_blocks = (
        evaluate_wape_blocks(
            model,
            test_set,
            graph,
            device,
            eval_batch_size,
            cfg["training"],
            test_loader,
            block_windows=bootstrap_block_windows,
            aggregation_level="micro_global",
        )
        if evaluate_test and test_set is not None and test_loader is not None
        else None
    )
    global_target_adjacency, _, _, _ = data.topology_knn_graph(
        int(cfg["data"].get("target_knn_k", 6))
    )
    global_target_graph_edges = int(
        np.count_nonzero(np.triu(global_target_adjacency > 0, k=1))
    )

    OUTPUTS.mkdir(exist_ok=True)
    torch.save(
        {"model_state": best_state, "active_nodes": nodes, "config": cfg},
        output_path(cfg, "centralized_model.pt"),
    )
    result = {
        "mode": "centralized",
        "code_revision": cfg.get("code_revision"),
        "experiment_name": cfg.get("experiment_name", "manual"),
        "config_signature": config_signature(cfg),
        "seed": int(cfg["seed"]),
        "device": str(device),
        "data_source_sha256": source_sha256,
        "node_indices_sha256": _hash_array(np.asarray(nodes, dtype=np.int64)),
        "split_bounds": {
            "train_end": int(bounds.train_end),
            "val_end": int(bounds.val_end),
            "total": int(bounds.total),
        },
        "architecture": str(cfg["model"].get("architecture", "pa_stfed")),
        "temporal_architecture": str(cfg["model"].get("temporal_architecture", "transformer")),
        "tcn_config": (
            {"layers": 2, "kernel_size": int(cfg["model"].get("tcn_kernel_size", 3)), "dilations": [1, 2], "causal": True, "residual": True}
            if str(cfg["model"].get("temporal_architecture", "transformer")).lower() == "tcn_transformer"
            else None
        ),
        "functional_graph_mode": str(cfg["model"].get("functional_graph_mode", "static")),
        "dynamic_context_steps": int(cfg["model"].get("dynamic_context_steps", 12)),
        "dynamic_gain_init": float(cfg["model"].get("dynamic_gain_init", 0.0)),
        "multiscale_patch_config": multiscale_patch_metadata(model),
        "horizon_decoder": horizon_decoder_metadata(model),
        "loss_mode": loss_mode,
        "scale_source": scale_source,
        "metric_alignment": (
            "raw_absolute_error_numerator" if loss_mode == "scale_aware_l1" else None
        ),
        "feeder_loss_weight": feeder_loss_weight,
        "graph_mode": str(cfg["data"].get("graph", "topology_knn")),
        "target_knn_k": int(cfg["data"].get("target_knn_k", 6)),
        "global_target_graph_edges": global_target_graph_edges,
        "graph_effective_nodes": int(len(graph.node_indices)),
        "graph_effective_undirected_edges": int(np.count_nonzero(np.triu(graph.adjacency > 0, k=1))),
        "graph_inferred_bridge_metadata": int(len(graph.bridge_edges)),
        "model_ablation": {
            "use_physical": bool(cfg["model"].get("use_physical", True)),
            "use_functional": bool(cfg["model"].get("use_functional", True)),
            "use_spatial_gate": bool(cfg["model"].get("use_spatial_gate", True)),
            "use_temporal_gate": bool(cfg["model"].get("use_temporal_gate", True)),
            "use_residual_anchor": bool(
                cfg["model"].get("use_residual_anchor", False)
            ),
        },
        "best_epoch": best_epoch,
        "selection_metric": selection_metric,
        "best_selection_score": best_score,
        "best_train": best_train,
        "best_validation": history[best_epoch - 1]["validation"],
        "test_evaluated": evaluate_test,
        "test": test_metrics,
        "evaluation_metadata": {
            "source_sha256": source_sha256,
            "history": int(train_set.history),
            "horizon": int(train_set.horizon),
            "node_count": int(len(nodes)),
            "node_indices_sha256": _hash_array(np.asarray(nodes, dtype=np.int64)),
            "validation_alignment": _dataset_alignment_metadata(val_set, source_sha256),
            "test_alignment": (
                _dataset_alignment_metadata(test_set, source_sha256)
                if test_set is not None
                else None
            ),
            "train_windows_total": int(len(train_set)),
            "validation_windows": int(len(val_set)),
            "test_windows": int(len(test_set)) if test_set is not None else None,
            "mape_floor": "nodewise 0.01 * training-split mean absolute load; values below floor excluded",
            "metric_scale": "all reported error metrics are computed after nodewise inverse normalization",
        },
        "bootstrap_block_windows": bootstrap_block_windows,
        "validation_wape_blocks": validation_wape_blocks,
        "test_wape_blocks": test_wape_blocks,
        "diagnostics": (
            model_diagnostics(
                model,
                val_set,
                graph,
                device,
                eval_batch_size,
                cfg["training"],
                val_loader,
            )
            if isinstance(model, PA_STFed)
            else None
        ),
        "history": history,
    }
    output_path(cfg, "centralized_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result
