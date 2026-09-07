"""Federated training runner."""

from __future__ import annotations

import json
import time
from copy import deepcopy

import numpy as np
import torch
from torch.func import functional_call
from torch.utils.data import DataLoader, Subset

from config import autocast_context, make_grad_scaler
from data import GraphView, LoadWindowDataset, archive_sha256, make_data_loader
from federated import (
    aggregate_private_updates, build_client_model, charbonnier_loss,
    train_local, weighted_average,
)
from privacy import gaussian_rdp_epsilon
from models import (
    PA_STFed,
    ala_parameter_prefixes,
    load_shared_state,
    local_parameter_prefixes,
    shared_state_dict,
    vanilla_ala_parameter_names,
)
from experiment_runtime import (
    _assert_active_nodes_train_stable,
    _batch_size,
    _dataset_alignment_metadata,
    _hash_array,
    _hash_partitions,
    _macro_average,
    _make_scheduler,
    _non_blocking,
    _selection_metric,
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
    OUTPUTS,
)

def _client_metric_stats(client_metrics: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """汇总每个客户端的均值、标准差和尾部误差分位数。

    客户端级指标先分别计算，再对客户端分布求统计量；这与把所有节点和
    时间点直接拼接后的 micro 平均不同，可显式暴露联邦 Non-IID 下的尾部
    客户端表现。
    """

    if not client_metrics:
        return {}
    names = tuple(client_metrics[0].keys())
    return {
        name: {
            "mean": float(np.mean(values := [metrics[name] for metrics in client_metrics])),
            "std": float(np.std(values)),
            "p90": float(np.quantile(values, 0.90)),
            "p95": float(np.quantile(values, 0.95)),
            "max": float(np.max(values)),
            "worst_client": int(np.argmax(values)),
        }
        for name in names
    }


def _is_prefixed(name: str, prefixes: tuple[str, ...]) -> bool:
    """判断 state_dict 名称是否属于指定参数组。"""

    return name.startswith(prefixes)


def _load_moduleala_initial_state(
    model: PA_STFed,
    local_state: dict[str, torch.Tensor],
    global_state: dict[str, torch.Tensor],
    alpha_state: dict[str, torch.Tensor] | None,
    eligible_prefixes: tuple[str, ...] | None = None,
    eligible_names: tuple[str, ...] | None = None,
) -> dict[str, float]:
    """构造 ModuleALA 本轮初始化：shared 取 global，ALA 参数逐元素插值。

    该模块机制 inspired by / adapted from FedALA 的 ALA 更新方式，
    不声称复现原始 FedALA。
    """

    ala_prefixes = eligible_prefixes or ala_parameter_prefixes()
    ala_name_set = set(eligible_names or ())
    local_prefixes = local_parameter_prefixes(False)
    current = model.state_dict()
    adapted: dict[str, torch.Tensor] = {}
    alpha_values: list[torch.Tensor] = []
    changed_values: list[torch.Tensor] = []
    for name, value in current.items():
        if _is_prefixed(name, local_prefixes):
            source = local_state[name]
        elif (
            name in ala_name_set
            if eligible_names is not None
            else _is_prefixed(name, ala_prefixes)
        ):
            local_value = local_state[name].to(device=value.device, dtype=value.dtype)
            global_value = global_state[name].to(device=value.device, dtype=value.dtype)
            alpha = (
                alpha_state[name].to(device=value.device, dtype=value.dtype)
                if alpha_state is not None and name in alpha_state
                else torch.ones_like(value)
            ).clamp(0.0, 1.0)
            mixed = local_value + alpha * (global_value - local_value)
            adapted[name] = mixed
            alpha_values.append(alpha.detach().float().reshape(-1))
            changed_values.append((mixed - local_value).detach().float().abs().reshape(-1))
            continue
        elif name in global_state:
            source = global_state[name]
        else:
            source = local_state[name]
        adapted[name] = source.to(device=value.device, dtype=value.dtype)
    model.load_state_dict(adapted, strict=True)
    if alpha_values:
        all_alpha = torch.cat(alpha_values)
        all_changed = torch.cat(changed_values)
        return {
            "alpha_min": float(all_alpha.min().cpu()),
            "alpha_max": float(all_alpha.max().cpu()),
            "alpha_mean": float(all_alpha.mean().cpu()),
            "nonzero_initialization": float((all_changed > 1e-12).sum().cpu()),
        }
    return {"alpha_min": 1.0, "alpha_max": 1.0, "alpha_mean": 1.0, "nonzero_initialization": 0.0}


def _load_modulelocal_state(
    model: PA_STFed,
    local_state: dict[str, torch.Tensor],
    global_state: dict[str, torch.Tensor],
    eligible_prefixes: tuple[str, ...],
    eligible_names: tuple[str, ...] | None = None,
) -> None:
    """ModuleLocal 初始化：functional 与指定高层模块本地，其余参数取 global。"""

    local_prefixes = local_parameter_prefixes(False)
    current = model.state_dict()
    adapted: dict[str, torch.Tensor] = {}
    ala_name_set = set(eligible_names or ())
    for name, value in current.items():
        if name.startswith(local_prefixes) or name in ala_name_set or name.startswith(eligible_prefixes):
            source = local_state[name]
        elif name in global_state:
            source = global_state[name]
        else:
            source = local_state[name]
        adapted[name] = source.to(device=value.device, dtype=value.dtype)
    model.load_state_dict(adapted, strict=True)


def _alpha_module_statistics(
    alpha_state: dict[str, torch.Tensor], eligible_prefixes: tuple[str, ...], eligible_names: tuple[str, ...] | None = None
) -> dict[str, dict[str, float]]:
    """将完整 alpha 压缩为按模块统计，数组本身只进入 checkpoint。"""

    groups: dict[str, list[torch.Tensor]] = {}
    for name, value in alpha_state.items():
        if eligible_names is not None:
            module = name.split(".", 1)[0]
        else:
            module = next((prefix[:-1] for prefix in eligible_prefixes if name.startswith(prefix)), "other")
        groups.setdefault(module, []).append(value.detach().float().reshape(-1))
    statistics: dict[str, dict[str, float]] = {}
    for module, values in groups.items():
        merged = torch.cat(values)
        statistics[module] = {
            "mean": float(merged.mean()),
            "std": float(merged.std(unbiased=False)),
            "min": float(merged.min()),
            "max": float(merged.max()),
            "non_one_ratio": float((merged - 1.0).abs().gt(1e-7).float().mean()),
        }
    return statistics


def _ala_window_indices(
    length: int,
    ratio: float,
    max_windows: int,
    seed: int,
) -> list[int]:
    """按整个 train 时间范围确定性分层抽取 ALA 窗口索引。

    滑窗之间高度重叠，随机抽取会把样本集中在局部时间段。这里把 train
    窗口划成等宽时间桶，每桶取一个由 ``seed`` 确定的中心位置；seed 仍由
    实验、客户端和通信轮次共同派生，因此每轮可复现地重采样，同时覆盖
    整个 train split。返回顺序按时间排序，便于审计。
    """

    if length < 1:
        return []
    if not 0.0 < ratio <= 1.0:
        raise ValueError("federated.ala_sample_ratio must be in (0, 1]")
    if max_windows < 1:
        raise ValueError("federated.ala_max_windows must be positive")
    count = min(length, max(1, int(np.ceil(length * ratio))), int(max_windows))
    if count == length:
        return list(range(length))
    # 每个时间桶只选一个位置。用整数哈希产生桶内确定性偏移，避免
    # Python hash 的进程随机化，同时保证首尾时间段都被覆盖。
    seed_u = int(seed) & 0xFFFFFFFFFFFFFFFF
    positions: list[int] = []
    for bucket in range(count):
        start = (bucket * length) // count
        end = ((bucket + 1) * length) // count
        width = max(1, end - start)
        value = (seed_u + 0x9E3779B97F4A7C15 * (bucket + 1)) & 0xFFFFFFFFFFFFFFFF
        value ^= value >> 30
        value = (value * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        value ^= value >> 27
        offset = int(value % width)
        positions.append(start + offset)
    return positions


def _learn_moduleala_weights(
    model: PA_STFed,
    previous_local_state: dict[str, torch.Tensor],
    global_state: dict[str, torch.Tensor],
    alpha_state: dict[str, torch.Tensor],
    dataset: LoadWindowDataset,
    graph: GraphView,
    config: dict,
    loader: DataLoader,
    graph_tensors_device: tuple[torch.Tensor, torch.Tensor],
    initial_adaptation: bool = False,
    eligible_prefixes: tuple[str, ...] | None = None,
    eligible_names: tuple[str, ...] | None = None,
) -> dict[str, float]:
    """只用 train 窗口学习 ModuleALA alpha，再写回客户端模型。"""

    device = next(model.parameters()).device
    training_config = config["training"]
    ala_prefixes = eligible_prefixes or ala_parameter_prefixes()
    ala_name_set = set(eligible_names or ())
    ala_names = tuple(
        name for name in model.state_dict()
        if (name in ala_name_set if eligible_names is not None else _is_prefixed(name, ala_prefixes))
    )
    if not ala_names:
        raise RuntimeError("ModuleALA found no eligible spatial_gate/temporal_gate/head parameters")
    alpha_parameters = {
        name: torch.nn.Parameter(
            alpha_state[name].to(device=device, dtype=torch.float32).clone().clamp(0.0, 1.0)
        )
        for name in ala_names
    }
    optimizer = torch.optim.Adam(alpha_parameters.values(), lr=float(config["federated"].get("ala_weight_lr", 1.0)))
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    adjacency, edge_features = graph_tensors_device
    # 每个 client/round 只做一次 CPU->device 拷贝和静态 state 构造。
    # functional_call 仍接收完整 state，但 batch 内只重新计算 ALA 张量。
    previous_local_state = {
        name: value.to(device=device)
        for name, value in previous_local_state.items()
    }
    global_device_state = {
        name: value.to(device=device)
        for name, value in global_state.items()
    }
    current_state = model.state_dict()
    local_prefixes = local_parameter_prefixes(False)
    static_state: dict[str, torch.Tensor] = {}
    local_values: dict[str, torch.Tensor] = {}
    global_values: dict[str, torch.Tensor] = {}
    for name, value in current_state.items():
        if name in alpha_parameters:
            local_values[name] = previous_local_state[name].to(dtype=value.dtype)
            global_values[name] = global_device_state[name].to(dtype=value.dtype)
        elif name.startswith(local_prefixes):
            static_state[name] = previous_local_state[name].to(dtype=value.dtype)
        elif name in global_device_state:
            static_state[name] = global_device_state[name].to(dtype=value.dtype)
        else:
            static_state[name] = previous_local_state[name].to(dtype=value.dtype)
    model.eval()
    steps = 0
    ala_started = time.perf_counter()
    last_loss = torch.zeros((), device=device)
    max_epochs = int(
        config["federated"].get(
            "ala_initial_epochs" if initial_adaptation else "ala_adapt_epochs", 1
        )
    )
    for _ in range(max_epochs):
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=device.type == "cuda")
            targets = targets.to(device, non_blocking=device.type == "cuda")
            # 只为 ALA 参数计算 local + alpha * (global-local)，其余张量复用缓存。
            state = dict(static_state)
            for name in alpha_parameters:
                local_value = local_values[name]
                global_value = global_values[name]
                mixed_fp32 = local_value.float() + alpha_parameters[name] * (
                    global_value.float() - local_value.float()
                )
                state[name] = mixed_fp32.to(dtype=current_state[name].dtype)
            with autocast_context(training_config, device):
                output = functional_call(model, state, (inputs, adjacency, edge_features))["prediction"]
                loss = charbonnier_loss(output, targets, float(config["model"]["robust_kappa"]))
            if not torch.isfinite(loss):
                raise RuntimeError("ModuleALA alpha learning produced NaN/Inf loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                for parameter in alpha_parameters.values():
                    parameter.clamp_(0.0, 1.0)
            last_loss = loss.detach()
            steps += 1
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    learned = {name: parameter.detach().cpu().clone().clamp(0.0, 1.0) for name, parameter in alpha_parameters.items()}
    alpha_state.update(learned)
    stats = _load_moduleala_initial_state(
        model, previous_local_state, global_state, alpha_state,
        eligible_prefixes=ala_prefixes, eligible_names=eligible_names,
    )
    all_alpha = torch.cat([value.reshape(-1) for value in alpha_state.values()])
    stats.update({
        "loss": float(last_loss.cpu()),
        "steps": float(steps),
        "ala_batches": float(steps),
        "ala_seconds": float(time.perf_counter() - ala_started),
        "alpha_non_one": float((all_alpha - 1.0).abs().gt(1e-7).sum().item()),
    })
    return stats


def federated(cfg: dict, device: torch.device) -> dict:
    """执行 LocalOnly、FedAvg、FedProx 或独立的 ModuleALA smoke/正式任务。"""

    if str(cfg["model"].get("architecture", "pa_stfed")).lower() != "pa_stfed":
        raise ValueError("federated training currently supports architecture=pa_stfed only")
    data = load_smartds(cfg)
    source_sha256 = archive_sha256(data.source)
    bounds = data.split_bounds(cfg["data"]["train_ratio"], cfg["data"]["val_ratio"])
    _assert_active_nodes_train_stable(data, bounds.train_end)
    partitions = data.client_partitions(int(cfg["federated"]["clients"]))
    template = make_model(cfg, len(partitions[0]), device)
    models = [build_client_model(template, len(nodes)).to(device) for nodes in partitions]
    personalized_head = bool(cfg["federated"].get("personalized_head", False))
    algorithm = str(cfg["federated"].get("algorithm", "FedAvg")).lower()
    local_only = algorithm == "localonly"
    is_moduleala = algorithm == "moduleala"
    is_modulelocal = algorithm == "modulelocal"
    is_vanilla_ala = algorithm in {"vanillafedala", "fedala"}
    is_ala = is_moduleala or is_vanilla_ala
    is_personalized_module = is_moduleala or is_modulelocal or is_vanilla_ala
    if algorithm not in {"localonly", "fedavg", "fedprox", "moduleala", "modulelocal", "vanillafedala", "fedala"}:
        raise ValueError("federated.algorithm must be LocalOnly, FedAvg, FedProx, ModuleALA, ModuleLocal, or VanillaFedALA")
    if (is_moduleala or is_modulelocal) and personalized_head:
        raise ValueError("ModuleALA/ModuleLocal own head.* personalization; personalized_head must be false")
    mu = float(cfg["federated"].get("mu", 0.0))
    if algorithm != "fedprox" and abs(mu) > 0.0:
        raise ValueError(f"proximal mu must be 0 for algorithm={cfg['federated'].get('algorithm')}; got {mu}")
    ala_prefixes = ala_parameter_prefixes()
    vanilla_eligible_names: tuple[str, ...] = ()
    if is_vanilla_ala:
        layer_idx = int(cfg["federated"].get("vanilla_ala_layer_idx", 2))
        vanilla_eligible_names = vanilla_ala_parameter_names(models[0], layer_idx)
        print(f"[VanillaFedALA] layer_idx={layer_idx}; eligible={list(vanilla_eligible_names)}")
    ala_eligible_names = vanilla_eligible_names if is_vanilla_ala else None
    global_state = shared_state_dict(models[0], personalized_head=personalized_head)

    # 数据集与图在各轮之间不变，提前构造可避免重复拓扑计算。
    train_sets = [make_dataset(data, nodes, "train", cfg) for nodes in partitions]
    val_sets = [make_dataset(data, nodes, "val", cfg) for nodes in partitions]
    # 测试集默认关闭；只有显式 *_test 任务才读取。
    evaluate_test = bool(cfg["training"].get("evaluate_test", False))
    test_sets = (
        [make_dataset(data, nodes, "test", cfg) for nodes in partitions]
        if evaluate_test
        else []
    )
    graphs = [
        data.graph_view(
            nodes,
            cfg["data"]["graph"],
            int(cfg["data"].get("hop_radius", 2)),
            int(cfg["data"].get("target_knn_k", 6)),
        )
        for nodes in partitions
    ]
    # 客户端子图和 DataLoader 在轮次之间不变，提前构造并复用，避免每轮
    # 重复执行 NumPy->CUDA 拷贝以及 worker 进程启动。
    graph_tensors_device = [graph_tensors(graph, device) for graph in graphs]
    train_batch_size = _batch_size(cfg["training"], "federated")
    eval_batch_size = _batch_size(cfg["training"], "federated", evaluation=True)
    train_loaders = [
        make_data_loader(train_set, train_batch_size, True, cfg["training"])
        for train_set in train_sets
    ]
    val_loaders = [
        make_data_loader(val_set, eval_batch_size, False, cfg["training"])
        for val_set in val_sets
    ]
    test_loaders = (
        [
            make_data_loader(test_set, eval_batch_size, False, cfg["training"])
            for test_set in test_sets
        ]
        if evaluate_test
        else []
    )

    batch_size = eval_batch_size
    patience = int(cfg["federated"].get("patience", 5))
    min_delta = float(cfg["training"].get("early_stop_min_delta", 0.0))
    selection_metric = _selection_metric(cfg["training"])
    privacy_cfg = cfg["privacy"]
    privacy_enabled = bool(privacy_cfg.get("enabled", False))
    if local_only and privacy_enabled:
        raise ValueError("LocalOnly has no server aggregation and cannot enable central DP")
    uniform_mean = bool(cfg["federated"].get("uniform_mean", True))
    if privacy_enabled and not uniform_mean:
        raise ValueError("Client-level DP requires uniform_mean=true for this implementation")
    if privacy_enabled and privacy_cfg.get("max_epsilon") is not None:
        planned_epsilon = gaussian_rdp_epsilon(
            noise_multiplier=float(privacy_cfg["noise_multiplier"]),
            rounds=int(cfg["federated"]["rounds"]),
            delta=float(privacy_cfg["delta"]),
        )
        if planned_epsilon > float(privacy_cfg["max_epsilon"]):
            raise ValueError(
                "Configured federated rounds exceed the DP epsilon budget: "
                f"planned epsilon={planned_epsilon:.4f}, "
                f"limit={float(privacy_cfg['max_epsilon']):.4f}"
            )

    history: list[dict] = []
    best_round = 0
    best_score = float("inf")
    stale_rounds = 0
    best_global_state: dict[str, torch.Tensor] | None = None
    best_client_states: list[dict[str, torch.Tensor]] | None = None
    best_ala_weights: list[dict[str, torch.Tensor]] | None = None
    # LocalOnly 需要跨轮保留 AdamW 动量；联邦客户端则按每轮本地任务重建优化器。
    local_optimizers = (
        [
            torch.optim.AdamW(
                model.parameters(),
                lr=float(cfg["training"]["learning_rate"]),
                weight_decay=float(cfg["training"]["weight_decay"]),
            )
            for model in models
        ]
        if local_only
        else [None] * len(models)
    )
    local_scalers = (
        [make_grad_scaler(cfg["training"], device) for _ in models]
        if local_only
        else [None] * len(models)
    )
    # 仅 ModuleALA/VanillaFedALA 持久化逐元素 alpha；ModuleLocal 不创建该状态。
    ala_weights: list[dict[str, torch.Tensor]] = [
        {
            name: torch.ones_like(value, dtype=torch.float32, device="cpu")
            for name, value in model.state_dict().items()
            if (
                name in set(ala_eligible_names or ())
                if is_vanilla_ala
                else _is_prefixed(name, ala_prefixes)
            )
        }
        for model in models
    ] if is_ala else []
    previous_local_states: list[dict[str, torch.Tensor]] = [
        {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        for model in models
    ] if is_personalized_module else []
    if is_ala:
        ratio = float(cfg["federated"].get("ala_sample_ratio", 0.05))
        if not 0.0 < ratio <= 1.0:
            raise ValueError("federated.ala_sample_ratio must be in (0, 1]")
        ala_max_windows = int(cfg["federated"].get("ala_max_windows", 2048))
        if ala_max_windows < 1:
            raise ValueError("federated.ala_max_windows must be positive")

    for round_index in range(1, int(cfg["federated"]["rounds"]) + 1):
        local_states: list[dict[str, torch.Tensor]] = []
        train_metrics: list[dict[str, float]] = []
        sample_weights: list[float] = []
        ala_round_stats: list[dict[str, float]] = []
        global_ala_before = {
            name: value.detach().clone()
            for name, value in global_state.items()
            if (name in set(ala_eligible_names) if is_vanilla_ala else name.startswith(ala_prefixes))
        }
        ala_sample_loaders: list[DataLoader] = []
        ala_window_counts: list[int] = []
        if is_ala:
            for client_index, dataset in enumerate(train_sets):
                count = min(len(dataset), max(1, int(np.ceil(len(dataset) * ratio))), ala_max_windows)
                round_seed = int(cfg["seed"]) + 1009 * (client_index + 1) + 9176 * round_index
                indices = _ala_window_indices(len(dataset), ratio, ala_max_windows, round_seed)
                ala_sample_loaders.append(make_data_loader(Subset(dataset, indices), train_batch_size, True, cfg["training"]))
                ala_window_counts.append(len(indices))

        for client_index, (model, train_set, graph) in enumerate(
            zip(models, train_sets, graphs)
        ):
            if is_ala:
                if round_index == 1:
                    # 第一轮严格按 FedAvg 初始化，ALA 从第二轮才启用。
                    load_shared_state(model, global_state)
                    ala_round_stats.append({"executed": 0.0, "alpha_min": 1.0, "alpha_max": 1.0, "alpha_mean": 1.0, "nonzero_initialization": 0.0, "ala_windows": float(ala_window_counts[client_index]), "ala_batches": 0.0, "ala_seconds": 0.0})
                else:
                    stats = _learn_moduleala_weights(
                        model,
                        previous_local_states[client_index],
                        global_state,
                        ala_weights[client_index],
                        train_set,
                        graph,
                        cfg,
                        ala_sample_loaders[client_index],
                        graph_tensors_device[client_index],
                        initial_adaptation=round_index == 2,
                        eligible_prefixes=ala_prefixes,
                        eligible_names=ala_eligible_names,
                    )
                    stats["executed"] = 1.0
                    ala_round_stats.append(stats)
            elif is_modulelocal and round_index > 1:
                _load_modulelocal_state(model, previous_local_states[client_index], global_state, ala_prefixes, ala_eligible_names)
            elif not local_only:
                load_shared_state(model, global_state)
            local_started = time.perf_counter()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            state, metrics = train_local(
                model,
                train_set,
                graph,
                cfg,
                global_state,
                loader=train_loaders[client_index],
                graph_tensors_device=graph_tensors_device[client_index],
                optimizer=local_optimizers[client_index],
                scaler=local_scalers[client_index],
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            local_seconds = float(time.perf_counter() - local_started)
            metrics["local_train_seconds"] = local_seconds
            if is_ala:
                ala_round_stats[client_index].update({
                    "ala_windows": float(ala_window_counts[client_index]),
                    "local_train_seconds": local_seconds,
                })
            if is_personalized_module:
                local_states.append({name: state[name] for name in global_state})
            else:
                local_states.append(state)
            if is_personalized_module:
                previous_local_states[client_index] = {
                    name: value.detach().cpu().clone() for name, value in model.state_dict().items()
                }
            train_metrics.append(metrics)
            sample_weights.append(metrics["samples"])

        # 只有 functional embeddings 永久不在 global_state；ModuleALA/VanillaFedALA
        # 的 ALA 参数和 ModuleLocal 的 gate/head 均参与 shape-compatible 聚合。
        server_names = tuple(global_state)
        if local_only:
            aggregation_audit = {
                "privacy_enabled": False,
                "clients": float(len(local_states)),
                "aggregation": "none",
            }
        elif privacy_enabled:
            private_states = [{name: state[name] for name in server_names} for state in local_states]
            private_global = {name: global_state[name] for name in server_names}
            aggregated, aggregation_audit = aggregate_private_updates(
                private_states,
                private_global,
                clip_norm=float(privacy_cfg["clip_norm"]),
                noise_multiplier=float(privacy_cfg["noise_multiplier"]),
            )
        else:
            aggregated = weighted_average(
                [{name: state[name] for name in server_names} for state in local_states],
                None if uniform_mean else sample_weights,
            )
            aggregation_audit = {
                "privacy_enabled": False,
                "clients": float(len(local_states)),
            }
        if not local_only:
            global_state.update(aggregated)
        global_ala_delta = float(max(
            [
                (global_state[name].float() - value.float()).abs().max().item()
                for name, value in global_ala_before.items()
                if name in global_state
            ] or [0.0]
        ))

        # ModuleALA 评估 global shared + client-local ALA/functional 状态，不能整体回载 global。
        validation_clients: list[dict[str, float]] = []
        for client_index, (model, val_set, graph) in enumerate(
            zip(models, val_sets, graphs)
        ):
            if not local_only and not is_personalized_module:
                load_shared_state(model, global_state)
            elif is_personalized_module:
                # 聚合后的 global 只覆盖非 ALA、非 functional 参数，保留个性化状态。
                current = model.state_dict()
                for name, value in global_state.items():
                    if (name in set(ala_eligible_names or ()) or (not is_vanilla_ala and name.startswith(ala_prefixes))
                            or name.startswith(local_parameter_prefixes(False))):
                        continue
                    if name in current:
                        current[name].copy_(value.to(device=current[name].device, dtype=current[name].dtype))
            validation_clients.append(
                evaluate(
                    model,
                    val_set,
                    graph,
                    device,
                    batch_size,
                    cfg["training"],
                    val_loaders[client_index],
                )
            )

        validation_macro = _macro_average(validation_clients)
        client_rmses = np.asarray(
            [metrics["rmse"] for metrics in validation_clients],
            dtype=np.float64,
        )
        record = {
            "round": round_index,
            "train_clients": train_metrics,
            "validation_clients": validation_clients,
            "validation_macro": validation_macro,
            "validation_client_stats": _client_metric_stats(validation_clients),
            "validation_rmse_p90": float(np.quantile(client_rmses, 0.90)),
            "worst_client": int(np.argmax(client_rmses)),
            "aggregation": aggregation_audit,
            "ala": {
                "executed": bool(is_ala and round_index >= 2),
                "clients": ala_round_stats,
                "alpha_min": float(min((item.get("alpha_min", 1.0) for item in ala_round_stats), default=1.0)),
                "alpha_max": float(max((item.get("alpha_max", 1.0) for item in ala_round_stats), default=1.0)),
                "nonzero_initialization_total": float(sum(item.get("nonzero_initialization", 0.0) for item in ala_round_stats)),
                "alpha_non_one_total": float(sum(item.get("alpha_non_one", 0.0) for item in ala_round_stats)),
                "global_ala_max_abs_delta": global_ala_delta,
                "sample_ratio": float(ratio) if is_ala else None,
                "max_windows": int(ala_max_windows) if is_ala else None,
                "ala_windows_total": int(sum(item.get("ala_windows", 0.0) for item in ala_round_stats)),
                "ala_batches_total": int(sum(item.get("ala_batches", 0.0) for item in ala_round_stats)),
                "ala_seconds_total": float(sum(item.get("ala_seconds", 0.0) for item in ala_round_stats)),
                "local_train_seconds_total": float(sum(item.get("local_train_seconds", 0.0) for item in ala_round_stats)),
            } if is_ala else None,
        }
        history.append(record)
        print(
            f"[Federated] round={round_index:03d} "
            f"val_macro_RMSE={validation_macro['rmse']:.6f} "
            f"val_RMSE_p90={record['validation_rmse_p90']:.6f} "
            f"val_macro_WAPE={validation_macro['wape']:.2f}% "
            f"val_macro_MAPE={validation_macro['mape']:.2f}%"
        )

        selection_score = float(validation_macro[selection_metric])
        if not np.isfinite(selection_score):
            raise RuntimeError(
                f"Validation macro {selection_metric} is non-finite "
                f"at round {round_index}"
            )
        if selection_score < best_score - min_delta:
            best_score = selection_score
            best_round = round_index
            stale_rounds = 0
            best_global_state = (
                {}
                if local_only
                else {
                    name: value.detach().cpu().clone()
                    for name, value in global_state.items()
                }
            )
            best_client_states = [
                {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
                for model in models
            ]
            best_ala_weights = [
                {name: value.detach().cpu().clone() for name, value in weights.items()}
                for weights in ala_weights
            ] if is_ala else None
        else:
            stale_rounds += 1
            if stale_rounds >= patience:
                print(
                    f"[Federated] early stop at round {round_index}; "
                    f"selection={selection_metric}"
                )
                break

    if best_global_state is None or best_client_states is None:
        raise RuntimeError("Federated training produced no valid checkpoint")
    for model, state in zip(models, best_client_states):
        model.load_state_dict(state)

    diagnostics = [
        model_diagnostics(
            model,
            val_set,
            graph,
            device,
            eval_batch_size,
            cfg["training"],
            val_loaders[client_index],
        )
        for client_index, (model, val_set, graph) in enumerate(
            zip(models, val_sets, graphs)
        )
    ]

    bootstrap_block_windows = int(
        cfg["training"].get("bootstrap_block_windows", 96)
    )
    validation_wape_blocks_clients = [
        evaluate_wape_blocks(
            model,
            val_set,
            graph,
            device,
            eval_batch_size,
            cfg["training"],
            val_loaders[client_index],
            block_windows=bootstrap_block_windows,
            aggregation_level="micro_client",
        )
        for client_index, (model, val_set, graph) in enumerate(
            zip(models, val_sets, graphs)
        )
    ]
    # 同一时间块内先累加所有客户端的绝对误差和目标绝对值和，
    # 再计算全局 micro-WAPE；该量才可与集中式全局 WAPE 配对比较。
    validation_wape_blocks_micro = aggregate_wape_blocks(
        validation_wape_blocks_clients
    )

    test_clients = None
    if evaluate_test:
        test_clients = [
            evaluate(
                model,
                test_set,
                graph,
                device,
                batch_size,
                cfg["training"],
                test_loaders[client_index],
            )
            for client_index, (model, test_set, graph) in enumerate(
                zip(models, test_sets, graphs)
            )
        ]
    test_wape_blocks_clients = (
        [
            evaluate_wape_blocks(
                model,
                test_set,
                graph,
                device,
                batch_size,
                cfg["training"],
                test_loaders[client_index],
                block_windows=bootstrap_block_windows,
                aggregation_level="micro_client",
            )
            for client_index, (model, test_set, graph) in enumerate(
                zip(models, test_sets, graphs)
            )
        ]
        if evaluate_test
        else None
    )
    test_wape_blocks_micro = (
        aggregate_wape_blocks(test_wape_blocks_clients)
        if test_wape_blocks_clients
        else None
    )
    rounds_executed = len(history)
    if privacy_enabled:
        epsilon = gaussian_rdp_epsilon(
            noise_multiplier=float(privacy_cfg["noise_multiplier"]),
            rounds=rounds_executed,
            delta=float(privacy_cfg["delta"]),
        )
    else:
        epsilon = None

    privacy_report = {
        "enabled": privacy_enabled,
        "C": float(privacy_cfg["clip_norm"]),
        "sigma": float(privacy_cfg["noise_multiplier"]),
        "q": 1.0,
        "R": rounds_executed,
        "delta": float(privacy_cfg["delta"]),
        "epsilon": epsilon,
        "planned_epsilon": (
            gaussian_rdp_epsilon(
                noise_multiplier=float(privacy_cfg["noise_multiplier"]),
                rounds=int(cfg["federated"]["rounds"]),
                delta=float(privacy_cfg["delta"]),
            )
            if privacy_enabled
            else None
        ),
        "max_epsilon": privacy_cfg.get("max_epsilon"),
        "accountant_version": "PA-STFed Gaussian RDP q=1 v1",
        "accountant_source": "privacy.py:68-99 (gaussian_rdp_epsilon); analytic full-participation q=1 accountant",
        "adjacency_definition": "client add/remove with fixed denominator K",
        "mean_noise_variance": (
            float(privacy_cfg["noise_multiplier"]) ** 2
            * float(privacy_cfg["clip_norm"]) ** 2
            / len(partitions) ** 2
            if privacy_enabled
            else 0.0
        ),
    }
    global_target_adjacency, _, _, _ = data.topology_knn_graph(
        int(cfg["data"].get("target_knn_k", 6))
    )
    global_target_graph_edges = int(
        np.count_nonzero(np.triu(global_target_adjacency > 0, k=1))
    )

    OUTPUTS.mkdir(exist_ok=True)
    torch.save(
        {
            "global_shared_state": best_global_state,
            "client_states": best_client_states,
            "ala_weights": best_ala_weights if is_ala else None,
            "client_nodes": partitions,
            "config": cfg,
        },
        output_path(cfg, "federated_model.pt"),
    )
    result = {
        "mode": "federated",
        "code_revision": cfg.get("code_revision"),
        "experiment_name": cfg.get("experiment_name", "manual"),
        "config_signature": config_signature(cfg),
        "seed": int(cfg["seed"]),
        "device": str(device),
        "data_source_sha256": source_sha256,
        "node_indices_sha256": _hash_array(np.asarray(data.active_indices, dtype=np.int64)),
        "client_partition_sha256": _hash_partitions(partitions),
        "split_bounds": {
            "train_end": int(bounds.train_end),
            "val_end": int(bounds.val_end),
            "total": int(bounds.total),
        },
        "graph_mode": str(cfg["data"].get("graph", "topology_knn")),
        "target_knn_k": int(cfg["data"].get("target_knn_k", 6)),
        "global_target_graph_edges": global_target_graph_edges,
        "graph_client_effective_undirected_edges": [
            int(np.count_nonzero(np.triu(graph.adjacency > 0, k=1))) for graph in graphs
        ],
        "graph_inferred_bridge_metadata": int(len(graphs[0].bridge_edges)) if graphs else 0,
        "model_ablation": {
            "use_physical": bool(cfg["model"].get("use_physical", True)),
            "use_functional": bool(cfg["model"].get("use_functional", True)),
            "use_spatial_gate": bool(cfg["model"].get("use_spatial_gate", True)),
            "use_temporal_gate": bool(cfg["model"].get("use_temporal_gate", True)),
            "use_residual_anchor": bool(
                cfg["model"].get("use_residual_anchor", False)
            ),
            "personalized_head": personalized_head,
        },
        "federated_algorithm": str(cfg["federated"].get("algorithm", "FedAvg")),
        "fedprox_mu": float(cfg["federated"].get("mu", 0.0)),
        "effective_mu": float(mu if algorithm == "fedprox" else 0.0),
        "proximal_enabled": bool(algorithm == "fedprox" and mu > 0.0),
        "client_node_counts": [len(nodes) for nodes in partitions],
        "evaluation_metadata": {
            "source_sha256": source_sha256,
            "history": int(train_sets[0].history),
            "horizon": int(train_sets[0].horizon),
            "node_count": int(len(data.active_indices)),
            "node_indices_sha256": _hash_array(
                np.asarray(data.active_indices, dtype=np.int64)
            ),
            "client_count": int(len(partitions)),
            "client_ids": list(range(len(partitions))),
            "client_partition_sha256": _hash_partitions(partitions),
            "train_windows_per_client": [int(len(dataset)) for dataset in train_sets],
            "validation_windows_per_client": [int(len(dataset)) for dataset in val_sets],
            "test_windows_per_client": (
                [int(len(dataset)) for dataset in test_sets] if evaluate_test else None
            ),
            "mape_floor": "nodewise 0.01 * training-split mean absolute load; values below floor excluded",
            "metric_scale": "all reported error metrics are computed after nodewise inverse normalization",
        },
        "best_round": best_round,
        "selection_metric": selection_metric,
        "best_selection_score": best_score,
        "best_validation": history[best_round - 1],
        "test_evaluated": evaluate_test,
        "test_clients": test_clients,
        "test_macro": _macro_average(test_clients) if test_clients else None,
        "bootstrap_block_windows": bootstrap_block_windows,
        "validation_wape_blocks_micro": validation_wape_blocks_micro,
        "validation_wape_blocks_clients": validation_wape_blocks_clients,
        "test_wape_blocks_micro": test_wape_blocks_micro,
        "test_wape_blocks_clients": test_wape_blocks_clients,
        "test_client_stats": _client_metric_stats(test_clients) if test_clients else None,
        "test_rmse_p90": (
            float(np.quantile([metrics["rmse"] for metrics in test_clients], 0.90))
            if test_clients
            else None
        ),
        "diagnostics": diagnostics,
        "privacy": privacy_report,
        "history": history,
        "ala": {
            "enabled": is_ala,
            "mode": "ModuleALA" if is_moduleala else ("VanillaFedALA" if is_vanilla_ala else ("ModuleLocal" if is_modulelocal else None)),
            "eligible_prefixes": list(ala_prefixes) if is_moduleala else [],
            "eligible_names": list(ala_eligible_names) if is_vanilla_ala else [],
            "sample_ratio": float(cfg["federated"].get("ala_sample_ratio", 0.0)) if is_ala else None,
            "weight_lr": float(cfg["federated"].get("ala_weight_lr", 0.0)) if is_ala else None,
            "statistics": [
                _alpha_module_statistics(weights, ala_prefixes, ala_eligible_names)
                for weights in (best_ala_weights if is_ala and best_ala_weights is not None else [])
            ],
        },
    }
    output_path(cfg, "federated_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result
