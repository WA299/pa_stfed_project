"""PA-STFed 正式实验入口。

外部只暴露完整矩阵、数据审计和配置预览；具体训练任务统一由
experiments.yaml 调度，避免绕过正式矩阵产生不可比结果。
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import traceback
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from torch.func import functional_call
from torch.utils.data import DataLoader, Subset

from config import (
    autocast_context,
    configure_torch_runtime,
    load_config,
    make_grad_scaler,
    resolve_device,
    set_seed,
)
from data import (
    GraphView,
    LoadWindowDataset,
    SmartDS,
    archive_sha256,
    make_data_loader,
)
from federated import (
    aggregate_private_updates,
    build_client_model,
    charbonnier_loss,
    metric_summary,
    train_local,
    weighted_average,
)
from metrics import gate_diagnostics, linear_cka
from model import (
    AGCRNBaseline,
    GraphWaveNetBaseline,
    ITransformerBaseline,
    LSTMBaseline,
    PA_STFed,
    ala_parameter_prefixes,
    load_shared_state,
    local_parameter_prefixes,
    shared_state_dict,
)
from privacy import gaussian_rdp_epsilon


CODE_DIR = Path(__file__).resolve().parent
ROOT = CODE_DIR.parent
CONFIG = CODE_DIR / "config.yaml"
EXPERIMENTS_CONFIG = CODE_DIR / "experiments.yaml"
OUTPUTS = ROOT / "results"


def load_project_config() -> dict:
    """读取 YAML，并把相对数据路径解析为项目内绝对路径。"""

    cfg = load_config(CONFIG)
    source = Path(cfg["data"]["source"])
    if not source.is_absolute():
        cfg["data"]["source"] = str((ROOT / source).resolve())
    calendar_source = cfg["data"].get("calendar_source")
    if calendar_source:
        calendar_path = Path(calendar_source)
        if not calendar_path.is_absolute():
            cfg["data"]["calendar_source"] = str((ROOT / calendar_path).resolve())
    return cfg


def load_smartds(cfg: dict) -> SmartDS:
    """按同一配置同时加载 canonical 图和已验证 calendar sidecar。"""

    return SmartDS.load(
        cfg["data"]["source"],
        calendar_source=cfg["data"].get("calendar_source"),
    )


def config_signature(cfg: dict) -> str:
    """生成稳定配置指纹，用于校验断点续跑结果是否属于同一配置。"""

    payload = json.dumps(cfg, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def deep_merge(base: dict, overrides: dict) -> dict:
    """递归合并实验覆盖项，避免每个实验复制整份主配置。"""

    merged = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def output_path(cfg: dict, filename: str) -> Path:
    """按实验标签生成独立输出文件名，避免不同实验互相覆盖。"""

    tag = str(cfg.get("experiment_tag", "")).strip()
    prefix = f"{tag}_" if tag else ""
    OUTPUTS.mkdir(exist_ok=True)
    return OUTPUTS / f"{prefix}{filename}"


def _pairwise_hop_summary(graph: nx.Graph, node_indices: np.ndarray) -> dict[str, float | int | dict[str, int]]:
    """统计指定节点对在图中的最短跳数分布。

    统计只针对无序节点对，排除自环；不可达节点对单独计数。该量用于审计
    ``d_hop`` 的图论含义，不把跳数解释为导线长度或电气阻抗。
    """

    nodes = [int(node) for node in np.asarray(node_indices, dtype=np.int64).tolist()]
    lengths = dict(nx.all_pairs_shortest_path_length(graph))
    values: list[int] = []
    unreachable = 0
    for left_position, source in enumerate(nodes):
        source_lengths = lengths.get(source, {})
        for target in nodes[left_position + 1 :]:
            distance = source_lengths.get(target)
            if distance is None:
                unreachable += 1
            else:
                values.append(int(distance))
    if values:
        array = np.asarray(values, dtype=np.float64)
        histogram: dict[str, int] = {}
        for value in sorted(set(values)):
            histogram[str(value)] = int(np.sum(array == value))
        summary: dict[str, float | int | dict[str, int]] = {
            "pairs_total": int(len(nodes) * (len(nodes) - 1) // 2),
            "connected_pairs": int(len(values)),
            "unreachable_pairs": int(unreachable),
            "coverage": float(len(values) / max(len(values) + unreachable, 1)),
            "min": int(np.min(array)),
            "median": float(np.median(array)),
            "p90": float(np.quantile(array, 0.90)),
            "p95": float(np.quantile(array, 0.95)),
            "max": int(np.max(array)),
            "histogram": histogram,
        }
    else:
        summary = {
            "pairs_total": int(len(nodes) * (len(nodes) - 1) // 2),
            "connected_pairs": 0,
            "unreachable_pairs": int(unreachable),
            "coverage": 0.0,
            "min": None,
            "median": None,
            "p90": None,
            "p95": None,
            "max": None,
            "histogram": {},
        }
    return summary


def _hash_array(values: np.ndarray) -> str:
    """对数组的形状、类型和值计算稳定哈希，供结果配对审计使用。"""

    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(repr(array.shape).encode("utf-8"))
    digest.update(array.view(np.uint8).tobytes())
    return digest.hexdigest()


def _hash_partitions(partitions: list[np.ndarray]) -> str:
    """对联邦客户端节点划分计算顺序敏感哈希，不在结果中暴露完整节点列表。"""

    canonical = [
        np.asarray(nodes, dtype=np.int64).reshape(-1).tolist() for nodes in partitions
    ]
    return hashlib.sha256(
        json.dumps(canonical, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _dataset_alignment_metadata(
    dataset: LoadWindowDataset,
    source_sha256: str | None = None,
) -> dict[str, object]:
    """返回可用于严格配对比较的数据集、节点和时间窗口元数据。"""

    origins = np.asarray(dataset.origins, dtype=np.int64)
    return {
        "source_sha256": source_sha256 or archive_sha256(dataset.data.source),
        "history": int(dataset.history),
        "horizon": int(dataset.horizon),
        "node_count": int(len(dataset.node_indices)),
        "node_indices_sha256": _hash_array(
            np.asarray(dataset.node_indices, dtype=np.int64)
        ),
        "origin_sha256": _hash_array(origins),
        "first_origin": int(origins[0]) if origins.size else None,
        "last_origin": int(origins[-1]) if origins.size else None,
        "n_windows": int(origins.size),
    }


def _federated_parameter_audit(cfg: dict, node_count: int) -> dict[str, object]:
    """从实际 PA-STFed state_dict 生成可审计的联邦参数分组。"""

    model = make_model(cfg, node_count, torch.device("cpu"))
    state_names = tuple(model.state_dict().keys())
    local_prefixes = local_parameter_prefixes(False)
    local_functional = tuple(
        name for name in state_names if name.startswith(local_prefixes)
    )
    head = tuple(name for name in state_names if name.startswith("head."))
    shared_fedavg = tuple(name for name in state_names if name not in local_functional)
    shared_personalized = tuple(name for name in shared_fedavg if name not in head)
    return {
        "code_locations": {
            "prefix_definition": "model.py:17-23",
            "state_extraction": "model.py:603-612",
            "state_loading": "model.py:615-621",
            "local_training": "federated.py:81-156",
            "aggregation": "federated.py:159-219",
            "server_loop": "run.py:1275-1650",
        },
        "shared_fedavg_fedprox": list(shared_fedavg),
        "local_functional_embeddings_all_federated_modes": list(local_functional),
        "local_prediction_head_only_personalized": list(head),
        "shared_personalized_encoder_and_non_head": list(shared_personalized),
        "counts": {
            "state_entries_total": len(state_names),
            "shared_fedavg_fedprox": len(shared_fedavg),
            "local_functional_embeddings": len(local_functional),
            "local_head_personalized": len(head),
            "shared_personalized": len(shared_personalized),
        },
        "non_parameter_local_objects": [
            "client graph adjacency and edge_features",
            "client node index partition",
            "raw load windows and normalization statistics",
        ],
        "interpretation": (
            "FedAvg/FedProx 聚合共享编码器、物理图层、功能图 value、门控、"
            "Transformer 和默认预测头；功能图节点嵌入因客户端节点数不同始终本地保留；"
            "personalized_head=true 时额外把 head.* 保留在客户端。"
        ),
    }


def _assert_active_nodes_train_stable(data: SmartDS, train_end: int) -> None:
    """拒绝使用仅在验证/测试阶段才出现的节点，避免节点筛选泄漏。"""

    training_mask = np.any(data.load_ts[:train_end] != 0.0, axis=0)
    if not np.array_equal(training_mask, data.active_mask):
        raise RuntimeError(
            "active node set changes after the training split; rebuild topology and "
            "client partitions from training-only activity before running experiments"
        )


def make_model(cfg: dict, node_count: int, device: torch.device) -> torch.nn.Module:
    """根据 architecture 创建主模型或公开基线。

    LSTM 和 iTransformer 仅用于集中式预测对照；联邦训练入口仍严格限制为
    PA-STFed，以免把不同节点数客户端错误地接入固定节点参数模型。
    """

    architecture = str(cfg["model"].get("architecture", "pa_stfed")).lower()
    common = {
        "node_count": node_count,
        "input_dim": int(cfg["model"]["input_dim"]),
        "hidden_dim": int(cfg["model"]["hidden_dim"]),
        "horizon": int(cfg["data"]["horizon"]),
    }
    if architecture == "agcrn":
        return AGCRNBaseline(
            **common,
            embedding_dim=int(cfg.get("baselines", {}).get("agcrn_embedding_dim", 10)),
            cheb_order=int(cfg.get("baselines", {}).get("agcrn_cheb_order", 2)),
        ).to(device)
    if architecture == "gwnet":
        return GraphWaveNetBaseline(
            **common,
            layers=int(cfg.get("baselines", {}).get("gwnet_layers", 4)),
            embedding_dim=int(cfg.get("baselines", {}).get("gwnet_embedding_dim", 10)),
        ).to(device)
    if architecture == "lstm":
        # Kong et al.（IEEE TSG, 2019）将 LSTM 用作短期住宅负荷预测的
        # 直接序列基线；layers/hidden_dim/dropout 是本项目统一预算下的适配项。
        return LSTMBaseline(
            **common,
            layers=int(cfg.get("baselines", {}).get("lstm_layers", 2)),
            dropout=float(cfg["model"].get("dropout", 0.1)),
        ).to(device)
    if architecture == "itransformer":
        # Liu et al.（ICLR, 2024）的 iTransformer 将变量作为 token，
        # 在节点×特征变量维度执行注意力；其输入仍与其他模型完全一致。
        return ITransformerBaseline(
            **common,
            history=int(cfg["data"]["history"]),
            heads=int(
                cfg.get("baselines", {}).get(
                    "itransformer_heads", cfg["model"].get("transformer_heads", 4)
                )
            ),
            layers=int(
                cfg.get("baselines", {}).get(
                    "itransformer_layers", cfg["model"].get("transformer_layers", 2)
                )
            ),
            dropout=float(cfg["model"].get("dropout", 0.1)),
        ).to(device)
    if architecture != "pa_stfed":
        raise ValueError(f"Unsupported model architecture: {architecture}")
    return PA_STFed(
        node_count=node_count,
        history=int(cfg["data"]["history"]),
        horizon=int(cfg["data"]["horizon"]),
        input_dim=int(cfg["model"]["input_dim"]),
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        functional_dim=int(cfg["model"]["functional_dim"]),
        spatial_heads=int(cfg["model"]["spatial_heads"]),
        transformer_layers=int(cfg["model"]["transformer_layers"]),
        transformer_heads=int(cfg["model"]["transformer_heads"]),
        dropout=float(cfg["model"]["dropout"]),
        use_physical=bool(cfg["model"].get("use_physical", True)),
        use_functional=bool(cfg["model"].get("use_functional", True)),
        use_spatial_gate=bool(cfg["model"].get("use_spatial_gate", True)),
        use_temporal_gate=bool(cfg["model"].get("use_temporal_gate", True)),
        use_residual_anchor=bool(cfg["model"].get("use_residual_anchor", False)),
    ).to(device)


def make_dataset(
    data: SmartDS,
    nodes: np.ndarray,
    split: str,
    cfg: dict,
) -> LoadWindowDataset:
    """构造按时间顺序切分的数据窗口，避免未来样本泄漏到训练集。"""

    bounds = data.split_bounds(cfg["data"]["train_ratio"], cfg["data"]["val_ratio"])
    sampling_minutes = float(cfg["data"].get("sampling_interval_minutes", 15))
    if sampling_minutes <= 0:
        raise ValueError("sampling_interval_minutes must be positive")
    daily_period_float = 24.0 * 60.0 / sampling_minutes
    daily_period = int(round(daily_period_float))
    if not np.isclose(daily_period, daily_period_float):
        raise ValueError("sampling_interval_minutes must divide a 24-hour day")
    return LoadWindowDataset(
        data,
        nodes,
        split,
        history=int(cfg["data"]["history"]),
        horizon=int(cfg["data"]["horizon"]),
        train_end=bounds.train_end,
        val_end=bounds.val_end,
        daily_period=daily_period,
        weekly_period=7 * daily_period,
        max_windows=cfg["training"].get(f"max_{split}_windows"),
    )


def graph_tensors(graph: GraphView, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """在一个位置完成静态图张量到目标设备的转移。"""

    return (
        torch.from_numpy(graph.adjacency).to(device),
        torch.from_numpy(graph.edge_features).to(device),
    )


def _non_blocking(training_config: dict, device: torch.device) -> bool:
    """判断主机到 GPU 的数据拷贝是否可以使用 pinned memory 异步重叠。"""

    return bool(training_config.get("pin_memory", False)) and device.type == "cuda"


def _batch_size(training_config: dict, mode: str, evaluation: bool = False) -> int:
    """根据任务选择显存安全的 batch size。

    集中式 PA-STFed 的空间注意力张量随 ``历史长度 × 节点数²`` 增长；
    联邦客户端只有 11--12 个有效节点，可以安全使用更大的本地 batch。
    两种任务共用一个过大的 batch 会导致集中式显存溢出，共用一个过小
    batch 又会使联邦客户端的 GPU 计算单元长期空闲。
    """

    if evaluation:
        value = training_config.get("eval_batch_size", training_config["batch_size"])
    elif mode == "federated":
        value = training_config.get(
            "federated_batch_size", training_config["batch_size"]
        )
    else:
        value = training_config.get(
            "centralized_batch_size", training_config["batch_size"]
        )
    value = int(value)
    if value < 1:
        raise ValueError("batch sizes must be positive")
    return value


def _selection_metric(training_config: dict) -> str:
    """返回 checkpoint 选择指标，并限制为已实现的误差指标。"""

    name = str(training_config.get("selection_metric", "rmse")).lower()
    supported = {"mae", "rmse", "wape", "smape", "mape"}
    if name not in supported:
        raise ValueError(
            f"training.selection_metric must be one of {sorted(supported)}, got {name!r}"
        )
    return name


def _make_scheduler(
    optimizer: torch.optim.Optimizer,
    training_config: dict,
):
    """构造验证平台期学习率调度器；name=none 时关闭。"""

    scheduler_config = training_config.get("scheduler", {}) or {}
    name = str(scheduler_config.get("name", "none")).lower()
    if name in {"none", "off", "disabled"}:
        return None
    if name != "reduce_on_plateau":
        raise ValueError(f"Unsupported training.scheduler.name={name!r}")
    factor = float(scheduler_config.get("factor", 0.5))
    patience = int(scheduler_config.get("patience", 2))
    min_lr = float(scheduler_config.get("min_lr", 1e-5))
    if not 0.0 < factor < 1.0 or patience < 0 or min_lr < 0.0:
        raise ValueError("Invalid ReduceLROnPlateau scheduler parameters")
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=factor,
        patience=patience,
        min_lr=min_lr,
    )


def _make_train_diagnostic_loader(
    dataset: LoadWindowDataset,
    batch_size: int,
    training_config: dict,
) -> DataLoader:
    """在时间轴上等间隔抽取训练窗口，仅用于最佳模型的泛化差距审计。"""

    limit = int(training_config.get("max_train_diagnostic_windows", 1024))
    if limit < 1:
        raise ValueError("training.max_train_diagnostic_windows must be positive")
    count = min(limit, len(dataset))
    indices = np.linspace(0, len(dataset) - 1, num=count, dtype=np.int64)
    subset = Subset(dataset, indices.tolist())
    return make_data_loader(subset, batch_size, False, training_config)


def audit(cfg: dict) -> dict:
    """复核数据形状、零负荷节点影响、两种拓扑方案和客户端节点数。"""

    data = load_smartds(cfg)
    # ``legacy_inf`` 是历史方案：在全部273个节点上补边，可能产生大量
    # 零负荷--零负荷桥接。``forest``/``mst_tag`` 是修正方案：先把零负荷
    # 节点作为拓扑中继投影，再只用有效负荷节点作为 MST 端点。
    legacy_imputed_adj, legacy_bridges = data.heuristic_topology_imputation()
    projected_adj, projected_hops, projected_components = data.projected_topology()
    corrected_imputed_adj, corrected_hops, corrected_bridges = (
        data.heuristic_projected_topology_imputation()
    )
    bounds = data.split_bounds(cfg["data"]["train_ratio"], cfg["data"]["val_ratio"])
    training_active_mask = np.any(data.load_ts[: bounds.train_end] != 0.0, axis=0)
    partitions = data.client_partitions(int(cfg["federated"]["clients"]))
    duplicate_groups = data.duplicate_groups
    node_to_client = {
        int(node): client_index
        for client_index, nodes in enumerate(partitions)
        for node in nodes.tolist()
    }
    duplicate_cross_client_groups = [
        {
            "nodes": [int(node) for node in group.tolist()],
            "clients": sorted({node_to_client[int(node)] for node in group.tolist()}),
        }
        for group in duplicate_groups
        if len({node_to_client[int(node)] for node in group.tolist()}) > 1
    ]
    client_profiles = []
    for client_index, nodes in enumerate(partitions):
        client_load = data.load_ts[:, nodes]
        aggregate_load = client_load.sum(axis=1)
        mean_load = float(aggregate_load.mean())
        client_profiles.append(
            {
                "client": client_index,
                "node_count": int(len(nodes)),
                "mean_total_load": mean_load,
                "std_total_load": float(aggregate_load.std()),
                "peak_total_load": float(aggregate_load.max()),
                "peak_to_mean": float(aggregate_load.max() / max(mean_load, 1e-6)),
            }
        )

    # 仅用训练时段量化客户端分布差异，避免测试集参与 Non-IID 诊断。
    # Wasserstein 距离用固定分位点近似；Jensen--Shannon 散度使用全部
    # 客户端共享的直方图区间，数值采用自然对数（上界 ln 2）。
    client_train_values = [
        data.load_ts[: bounds.train_end, nodes].astype(np.float64).reshape(-1)
        for nodes in partitions
    ]
    quantile_grid = np.linspace(0.0, 1.0, 257)
    client_quantiles = [np.quantile(values, quantile_grid) for values in client_train_values]
    all_train_values = np.concatenate(client_train_values)
    hist_low, hist_high = np.quantile(all_train_values, [0.005, 0.995])
    if hist_high <= hist_low:
        hist_high = hist_low + 1.0
    histogram_edges = np.linspace(hist_low, hist_high, 65)
    client_histograms = []
    for values in client_train_values:
        clipped = np.clip(values, hist_low, hist_high)
        counts, _ = np.histogram(clipped, bins=histogram_edges)
        probabilities = counts.astype(np.float64) + 1e-12
        client_histograms.append(probabilities / probabilities.sum())
    pairwise_non_iid = []
    for left in range(len(partitions)):
        for right in range(left + 1, len(partitions)):
            p = client_histograms[left]
            q = client_histograms[right]
            midpoint = 0.5 * (p + q)
            js_divergence = 0.5 * (
                np.sum(p * np.log(p / midpoint)) + np.sum(q * np.log(q / midpoint))
            )
            pairwise_non_iid.append(
                {
                    "clients": [left, right],
                    "quantile_wasserstein_approx": float(
                        np.mean(np.abs(client_quantiles[left] - client_quantiles[right]))
                    ),
                    "jensen_shannon_nats": float(js_divergence),
                }
            )

    raw_edge_count = int(np.count_nonzero(np.triu(data.adj > 0, k=1)))
    full_raw_graph = nx.from_numpy_array((data.adj > 0).astype(np.uint8))
    active_indices = data.active_indices
    active_node_set = set(int(node) for node in active_indices.tolist())
    active_induced_graph = full_raw_graph.subgraph(active_indices.tolist()).copy()
    projected_graph = nx.from_numpy_array((projected_adj > 0).astype(np.uint8))
    corrected_graph = nx.from_numpy_array((corrected_imputed_adj > 0).astype(np.uint8))

    # 对有效节点对统计图论最短跳数。raw/legacy 使用完整 273 节点图，
    # projected/mst_projected 使用 92 节点有效节点图；不可达对不强行填充，
    # 以 coverage 和 unreachable_pairs 明确记录。该诊断不代表电气距离。
    hop_distance_summary = {
        "G_raw_active_pairs": _pairwise_hop_summary(full_raw_graph, active_indices),
        "G_legacy_inf_active_pairs": _pairwise_hop_summary(
            nx.from_numpy_array((legacy_imputed_adj > 0).astype(np.uint8)),
            active_indices,
        ),
        "G_projected_raw_active_pairs": _pairwise_hop_summary(
            projected_graph, np.arange(len(active_indices), dtype=np.int64)
        ),
        "G_mst_projected_active_pairs": _pairwise_hop_summary(
            corrected_graph, np.arange(len(active_indices), dtype=np.int64)
        ),
    }

    parameter_audit = _federated_parameter_audit(
        cfg, node_count=int(len(partitions[0]))
    )
    parameter_csv = OUTPUTS / "federated_parameter_groups.csv"
    with parameter_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["group", "state_name", "FedAvg/FedProx", "Personalized"])
        local_functional = set(parameter_audit["local_functional_embeddings_all_federated_modes"])
        head_names = set(parameter_audit["local_prediction_head_only_personalized"])
        for state_name in parameter_audit["shared_fedavg_fedprox"]:
            writer.writerow([
                "shared_encoder_or_head",
                state_name,
                "aggregate",
                "aggregate" if state_name not in head_names else "local",
            ])
        for state_name in parameter_audit["local_functional_embeddings_all_federated_modes"]:
            writer.writerow(["functional_node_embedding", state_name, "local", "local"])

    def graph_row(name: str, node_count: int, edge_count: int, component_count: int, inferred: int, note: str) -> dict:
        return {
            "graph": name,
            "nodes": int(node_count),
            "undirected_edges": int(edge_count),
            "components": int(component_count),
            "inferred_edges": int(inferred),
            "note": note,
        }

    topology_comparison = [
        graph_row(
            "raw_full",
            data.node_count,
            raw_edge_count,
            nx.number_connected_components(full_raw_graph),
            0,
            "原始273节点图；包含零负荷节点",
        ),
        graph_row(
            "active_induced",
            len(active_indices),
            int(active_induced_graph.number_of_edges()),
            nx.number_connected_components(active_induced_graph),
            0,
            "删除零负荷节点后的诱导子图，仅用于诊断，不作为正式图",
        ),
        graph_row(
            "projected_raw",
            len(active_indices),
            int(np.count_nonzero(np.triu(projected_adj > 0, k=1))),
            len(projected_components),
            0,
            "零负荷节点中继投影后的原始拓扑图；投影可能形成环，不预设为森林",
        ),
        graph_row(
            "legacy_inf",
            data.node_count,
            int(np.count_nonzero(np.triu(legacy_imputed_adj > 0, k=1))),
            nx.number_connected_components(nx.from_numpy_array(legacy_imputed_adj)),
            len(legacy_bridges),
            "历史全节点MST；桥接端点可能为零负荷节点",
        ),
        graph_row(
            "mst_projected",
            len(active_indices),
            int(np.count_nonzero(np.triu(corrected_imputed_adj > 0, k=1))),
            nx.number_connected_components(corrected_graph),
            len(corrected_bridges),
            "零负荷中继投影图后仅在有效节点之间执行MST",
        ),
    ]
    with (OUTPUTS / "topology_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(topology_comparison[0]))
        writer.writeheader()
        writer.writerows(topology_comparison)

    legacy_bridge_endpoint_types = {
        "active_active": int(
            sum(int(source in active_node_set and target in active_node_set) for source, target in legacy_bridges)
        ),
        "active_zero": int(
            sum(int((source in active_node_set) != (target in active_node_set)) for source, target in legacy_bridges)
        ),
        "zero_zero": int(
            sum(int(source not in active_node_set and target not in active_node_set) for source, target in legacy_bridges)
        ),
    }
    split_ranges = {
        "train": (0, bounds.train_end),
        "validation": (bounds.train_end, bounds.val_end),
        "test": (bounds.val_end, bounds.total),
    }
    split_load_statistics = {}
    for split_name, (start, stop) in split_ranges.items():
        values = data.load_ts[start:stop, data.active_indices]
        split_load_statistics[split_name] = {
            "time_steps": int(stop - start),
            "days": float((stop - start) / (24 * 60 / float(cfg["data"]["sampling_interval_minutes"]))),
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "std": float(values.std()),
            "p01": float(np.percentile(values, 1.0)),
            "p99": float(np.percentile(values, 99.0)),
        }

    OUTPUTS.mkdir(exist_ok=True)
    with (OUTPUTS / "bridge_edges.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["bridge_index", "source_index", "target_index", "scheme"])
        for index, (source, target) in enumerate(corrected_bridges):
            writer.writerow([index, source, target, "projected_active_mst"])
    with (OUTPUTS / "legacy_bridge_edges.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["bridge_index", "source_index", "target_index", "scheme"])
        for index, (source, target) in enumerate(legacy_bridges):
            writer.writerow([index, source, target, "legacy_all_node_mst"])

    report = {
        "code_revision": cfg.get("code_revision"),
        "config_signature": config_signature(cfg),
        "audit_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(data.source),
        "sha256": archive_sha256(data.source),
        "fields": {
            "node_coords": list(data.node_coords.shape),
            "adj": list(data.adj.shape),
            "edge_index": list(data.edge_index.shape),
            "load_ts": list(data.load_ts.shape),
            "node_ids": list(data.node_ids.shape),
        },
        "active_load_nodes": int(data.active_mask.sum()),
        "zero_only_nodes": int((~data.active_mask).sum()),
        "training_active_load_nodes": int(training_active_mask.sum()),
        "active_mask_stable_across_full_series": bool(
            np.array_equal(training_active_mask, data.active_mask)
        ),
        "raw_components": len(data.raw_components()),
        "raw_undirected_edges": raw_edge_count,
        "legacy_imputed_bridge_edges": len(legacy_bridges),
        "legacy_imputed_undirected_edges": int(np.count_nonzero(np.triu(legacy_imputed_adj > 0, k=1))),
        "active_induced_components": nx.number_connected_components(active_induced_graph),
        "active_induced_undirected_edges": int(active_induced_graph.number_of_edges()),
        "projected_raw_components": len(projected_components),
        "projected_raw_undirected_edges": int(np.count_nonzero(np.triu(projected_adj > 0, k=1))),
        "corrected_mst_bridge_edges": len(corrected_bridges),
        "corrected_mst_undirected_edges": int(np.count_nonzero(np.triu(corrected_imputed_adj > 0, k=1))),
        "topology_comparison": topology_comparison,
        "hop_distance_summary": hop_distance_summary,
        "federated_parameter_audit": parameter_audit,
        "federated_parameter_groups_csv": str(parameter_csv),
        "legacy_bridge_endpoint_types": legacy_bridge_endpoint_types,
        "topology_origin_interpretation": {
            "raw_graph_is_forest": bool(raw_edge_count - data.node_count + len(data.raw_components()) == 0),
            "zero_only_raw_components": int(sum(1 for component in data.raw_components() if not np.any(data.active_mask[component]))),
            "raw_components_with_active_load": int(sum(1 for component in data.raw_components() if np.any(data.active_mask[component]))),
            "statement": (
                "57个断连分量存在于273节点原始邻接图；删除181个零负荷节点后，"
                "有效节点诱导子图退化为92个孤立点。投影零负荷中继后保留56个含负荷分量，"
                "因此MST应增加55条有效节点候选桥接边，而非历史方案的56条全节点桥接边。"
            ),
        },
        "unique_nonzero_curves": len(duplicate_groups),
        "largest_duplicate_group": max(len(group) for group in duplicate_groups),
        "split_bounds": {
            "train_end": bounds.train_end,
            "val_end": bounds.val_end,
            "total": bounds.total,
        },
        "split_protocol": "strict chronological holdout without random shuffling",
        "split_ratios": {
            "train": float(cfg["data"]["train_ratio"]),
            "validation": float(cfg["data"]["val_ratio"]),
            "test": float(
                1.0 - cfg["data"]["train_ratio"] - cfg["data"]["val_ratio"]
            ),
        },
        "split_load_statistics": split_load_statistics,
        "client_node_counts": [int(len(nodes)) for nodes in partitions],
        "client_partition_protocol": (
            "official-tree topology partition; duplicate load-curve groups are "
            "recorded for audit but not constrained to one client"
        ),
        "duplicate_curve_groups_crossing_clients": duplicate_cross_client_groups,
        "duplicate_curve_group_split_count": len(duplicate_cross_client_groups),
        "client_load_profiles": client_profiles,
        "client_mean_load_cv": float(
            np.std([profile["mean_total_load"] for profile in client_profiles])
            / max(np.mean([profile["mean_total_load"] for profile in client_profiles]), 1e-6)
        ),
        "client_non_iid_training_only": {
            "pair_count": len(pairwise_non_iid),
            "wasserstein_approx_mean": float(
                np.mean([item["quantile_wasserstein_approx"] for item in pairwise_non_iid])
            ),
            "wasserstein_approx_max": float(
                np.max([item["quantile_wasserstein_approx"] for item in pairwise_non_iid])
            ),
            "jensen_shannon_nats_mean": float(
                np.mean([item["jensen_shannon_nats"] for item in pairwise_non_iid])
            ),
            "jensen_shannon_nats_max": float(
                np.max([item["jensen_shannon_nats"] for item in pairwise_non_iid])
            ),
            "pairwise": pairwise_non_iid,
        },
        "absolute_timestamp_available": data.timestamp is not None,
        "calendar_features": [
            "relative_daily_sin",
            "relative_daily_cos",
            "relative_weekly_sin",
            "relative_weekly_cos",
        ],
        "weather_available": False,
        "topology_rule": (
            "formal: official Line+Transformer target topology-kNN with global k=6; "
            "legacy MST modes are retained only for historical comparison"
        ),
    }
    output_path(cfg, "audit_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def evaluate(
    model: torch.nn.Module,
    dataset: LoadWindowDataset,
    graph: GraphView,
    device: torch.device,
    batch_size: int,
    training_config: dict | None = None,
    loader: DataLoader | None = None,
) -> dict[str, float]:
    """在反归一化后的负荷尺度上计算 MAE、RMSE、WAPE、SMAPE 和 MAPE。"""

    model.eval()
    adjacency, edge_features = graph_tensors(graph, device)
    predictions: list[torch.Tensor] = []
    targets_all: list[torch.Tensor] = []

    training_config = training_config or {}
    transfer_non_blocking = _non_blocking(training_config, device)
    eval_loader = (
        loader
        if loader is not None
        else make_data_loader(dataset, batch_size, False, training_config)
    )
    with torch.inference_mode():
        for inputs, targets in eval_loader:
            with autocast_context(training_config, device):
                prediction = model(
                    inputs.to(device, non_blocking=transfer_non_blocking),
                    adjacency,
                    edge_features,
                )["prediction"]
            predictions.append(prediction)
            targets_all.append(targets.to(device, non_blocking=transfer_non_blocking))

    if not predictions:
        return {
            name: float("nan")
            for name in ("mae", "rmse", "wape", "smape", "mape", "mape_valid_ratio")
        }

    prediction = torch.cat(predictions, dim=0).cpu()
    target = torch.cat(targets_all, dim=0).cpu()
    return metric_summary(prediction, target, dataset)


def evaluate_wape_blocks(
    model: torch.nn.Module,
    dataset: LoadWindowDataset,
    graph: GraphView,
    device: torch.device,
    batch_size: int,
    training_config: dict | None = None,
    loader: DataLoader | None = None,
    block_windows: int = 96,
    aggregation_level: str = "micro_global",
) -> dict[str, object]:
    """保存固定时间块的 WAPE 充分统计量，供配对 Bootstrap 使用。

    每个块记录所有窗口、预测步和节点上的绝对误差和目标绝对值和；不保存
    原始预测，避免结果文件膨胀。块按 ``shuffle=False`` 的时间顺序构造，
    因此不同模型在同一 split 上可以进行配对重采样。
    """

    if block_windows < 1:
        raise ValueError("block_windows must be positive")
    if aggregation_level not in {"micro_global", "micro_client"}:
        raise ValueError(
            "aggregation_level must be 'micro_global' or 'micro_client'"
        )
    model.eval()
    adjacency, edge_features = graph_tensors(graph, device)
    training_config = training_config or {}
    transfer_non_blocking = _non_blocking(training_config, device)
    eval_loader = loader or make_data_loader(dataset, batch_size, False, training_config)
    error_windows: list[np.ndarray] = []
    target_windows: list[np.ndarray] = []
    with torch.inference_mode():
        for inputs, targets in eval_loader:
            with autocast_context(training_config, device):
                prediction = model(
                    inputs.to(device, non_blocking=transfer_non_blocking),
                    adjacency,
                    edge_features,
                )["prediction"]
            prediction = dataset.denormalize(prediction).float()
            targets = dataset.denormalize(
                targets.to(device, non_blocking=transfer_non_blocking)
            ).float()
            error_windows.append(
                (prediction - targets).abs().sum(dim=(1, 2)).cpu().numpy()
            )
            target_windows.append(targets.abs().sum(dim=(1, 2)).cpu().numpy())
    if not error_windows:
        metadata = _dataset_alignment_metadata(dataset)
        return {
            **metadata,
            "aggregation_level": aggregation_level,
            "block_windows": int(block_windows),
            "block_n_windows": [],
            "wape": [],
            "error_sum": [],
            "target_sum": [],
        }
    errors = np.concatenate(error_windows)
    targets = np.concatenate(target_windows)
    if errors.ndim != 1 or targets.ndim != 1 or errors.shape != targets.shape:
        raise RuntimeError("WAPE sufficient statistics must be one-dimensional and aligned")
    if not np.isfinite(errors).all() or not np.isfinite(targets).all():
        raise RuntimeError("WAPE sufficient statistics contain NaN or Inf")
    error_sum = [
        float(errors[start : start + block_windows].sum())
        for start in range(0, len(errors), block_windows)
    ]
    target_sum = [
        float(targets[start : start + block_windows].sum())
        for start in range(0, len(targets), block_windows)
    ]
    block_n_windows = [
        int(min(block_windows, len(errors) - start))
        for start in range(0, len(errors), block_windows)
    ]
    if any(target <= 0.0 for target in target_sum):
        raise RuntimeError("WAPE target_sum must be positive in every time block")
    metadata = _dataset_alignment_metadata(dataset)
    return {
        **metadata,
        "aggregation_level": aggregation_level,
        "block_windows": int(block_windows),
        "n_windows": int(len(errors)),
        "block_n_windows": block_n_windows,
        "error_sum": error_sum,
        "target_sum": target_sum,
        "wape": [
            float(100.0 * error / max(target, 1e-6))
            for error, target in zip(error_sum, target_sum)
        ],
    }


def aggregate_wape_blocks(
    client_blocks: list[dict[str, object]],
) -> dict[str, object]:
    """把联邦客户端的区块充分统计量合成为全局 micro-WAPE。"""

    if not client_blocks:
        raise ValueError("client_blocks must not be empty")
    reference = client_blocks[0]
    required = (
        "source_sha256",
        "history",
        "horizon",
        "node_count",
        "origin_sha256",
        "first_origin",
        "last_origin",
        "n_windows",
        "block_windows",
        "block_n_windows",
        "error_sum",
        "target_sum",
    )
    for key in required:
        if key not in reference:
            raise ValueError(f"client WAPE block metadata missing {key!r}")
    for index, record in enumerate(client_blocks[1:], start=1):
        for key in (
            "source_sha256",
            "history",
            "horizon",
            "origin_sha256",
            "first_origin",
            "last_origin",
            "n_windows",
            "block_windows",
            "block_n_windows",
        ):
            if record.get(key) != reference.get(key):
                raise ValueError(
                    f"client block alignment differs at client {index}, field {key!r}"
                )
    error_arrays = [np.asarray(record["error_sum"], dtype=np.float64) for record in client_blocks]
    target_arrays = [np.asarray(record["target_sum"], dtype=np.float64) for record in client_blocks]
    if any(array.ndim != 1 for array in error_arrays + target_arrays):
        raise ValueError("client block sufficient statistics must be one-dimensional")
    if len({array.size for array in error_arrays + target_arrays}) != 1:
        raise ValueError("client block counts are inconsistent")
    error_sum = np.sum(np.stack(error_arrays), axis=0)
    target_sum = np.sum(np.stack(target_arrays), axis=0)
    if not np.isfinite(error_sum).all() or not np.isfinite(target_sum).all():
        raise ValueError("aggregated WAPE sufficient statistics contain NaN or Inf")
    if np.any(target_sum <= 0.0):
        raise ValueError("aggregated WAPE target_sum must be positive")
    return {
        **{key: reference[key] for key in required if key not in {"error_sum", "target_sum", "node_count"}},
        "aggregation_level": "micro_global",
        "client_count": int(len(client_blocks)),
        "node_count": int(sum(int(record["node_count"]) for record in client_blocks)),
        "error_sum": error_sum.tolist(),
        "target_sum": target_sum.tolist(),
        "wape": (100.0 * error_sum / target_sum).tolist(),
    }


def model_diagnostics(
    model: PA_STFed,
    dataset: LoadWindowDataset,
    graph: GraphView,
    device: torch.device,
    batch_size: int,
    training_config: dict | None = None,
    loader: DataLoader | None = None,
    max_gate_values: int = 200_000,
    max_cka_tokens: int = 4_096,
) -> dict:
    """遍历验证集，诊断双级门控饱和与双图表征冗余。

    门控张量按上限抽样后统计，避免把整个验证集的高维状态长期保存在内存中；
    CKA 使用相同的有限 token 子集，保证诊断可复现且不改变模型参数。
    """

    training_config = training_config or {}
    transfer_non_blocking = _non_blocking(training_config, device)
    diagnostic_loader = (
        loader
        if loader is not None
        else make_data_loader(dataset, batch_size, False, training_config)
    )
    adjacency, edge_features = graph_tensors(graph, device)
    model.eval()
    gamma_values: list[torch.Tensor] = []
    temporal_values: list[torch.Tensor] = []
    physical_tokens: list[torch.Tensor] = []
    functional_tokens: list[torch.Tensor] = []
    gate_count = 0
    cka_count = 0

    with torch.inference_mode():
        for inputs, _ in diagnostic_loader:
            with autocast_context(training_config, device):
                output = model(
                    inputs.to(device, non_blocking=transfer_non_blocking),
                    adjacency,
                    edge_features,
                )
            if gate_count < max_gate_values:
                remaining = max_gate_values - gate_count
                gamma = output["gamma"].detach().float().reshape(-1).cpu()[:remaining]
                temporal = output["temporal_gate"].detach().float().reshape(-1).cpu()[:remaining]
                gamma_values.append(gamma)
                temporal_values.append(temporal)
                gate_count += int(max(gamma.numel(), temporal.numel()))
            if cka_count < max_cka_tokens:
                remaining = max_cka_tokens - cka_count
                physical = output["physical"].detach().float().reshape(-1, output["physical"].shape[-1])
                functional = output["functional"].detach().float().reshape(-1, output["functional"].shape[-1])
                physical_tokens.append(physical[:remaining].cpu())
                functional_tokens.append(functional[:remaining].cpu())
                cka_count += min(remaining, physical.shape[0])

    if not gamma_values or not temporal_values:
        raise ValueError("validation dataset is empty; cannot compute gate diagnostics")
    physical_all = torch.cat(physical_tokens, dim=0)
    functional_all = torch.cat(functional_tokens, dim=0)
    return {
        "spatial_gate": gate_diagnostics(torch.cat(gamma_values)),
        "temporal_gate": gate_diagnostics(torch.cat(temporal_values)),
        "physical_functional_cka": linear_cka(physical_all, functional_all),
        "diagnostic_windows": len(dataset),
        "diagnostic_gate_values": int(torch.cat(gamma_values).numel()),
        "diagnostic_cka_tokens": int(physical_all.shape[0]),
    }


def evaluate_naive_baseline(
    dataset: LoadWindowDataset,
    kind: str,
    batch_size: int,
    daily_period: int,
    training_config: dict | None = None,
) -> dict[str, float]:
    """在调用方指定的数据窗口上评估持久性或相对日周期基线。"""

    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    loader = make_data_loader(dataset, batch_size, False, training_config or {})
    for inputs, target in loader:
        if kind == "persistence":
            prediction = inputs[:, -1, :, 0].unsqueeze(1).expand(
                -1, target.shape[1], -1
            )
        elif kind == "daily_naive":
            if inputs.shape[1] < daily_period:
                raise ValueError("daily_naive requires history >= daily_period")
            start = inputs.shape[1] - daily_period
            stop = start + target.shape[1]
            if stop > inputs.shape[1]:
                raise ValueError("daily_naive horizon exceeds the available history")
            # 从历史窗口中取出与预测起点相同相位的相对日周期片段。
            prediction = inputs[:, start:stop, :, 0]
        else:
            raise ValueError(f"Unknown baseline kind: {kind}")
        predictions.append(prediction)
        targets.append(target)
    if not predictions:
        return {name: float("nan") for name in ("mae", "rmse", "wape", "smape", "mape")}
    return metric_summary(torch.cat(predictions), torch.cat(targets), dataset)


def baselines(cfg: dict, device: torch.device) -> dict:
    """计算集中式与客户端宏平均的简单时间序列基线。

    锁参与模型选择阶段使用验证集；只有独立最终评估任务显式设置
    ``evaluate_test=true`` 时才允许访问测试集。
    """

    data = load_smartds(cfg)
    bounds = data.split_bounds(cfg["data"]["train_ratio"], cfg["data"]["val_ratio"])
    _assert_active_nodes_train_stable(data, bounds.train_end)
    partitions = data.client_partitions(int(cfg["federated"]["clients"]))
    batch_size = _batch_size(cfg["training"], "baseline", evaluation=True)
    sampling_minutes = float(cfg["data"].get("sampling_interval_minutes", 15))
    if sampling_minutes <= 0:
        raise ValueError("sampling_interval_minutes must be positive")
    daily_period_float = 24.0 * 60.0 / sampling_minutes
    daily_period = int(round(daily_period_float))
    if not np.isclose(daily_period, daily_period_float):
        raise ValueError("sampling_interval_minutes must divide a 24-hour day")
    evaluation_split = (
        "test" if bool(cfg["training"].get("evaluate_test", False)) else "val"
    )
    result = {
        "mode": "baseline",
        "code_revision": cfg.get("code_revision"),
        "experiment_name": cfg.get("experiment_name", "baselines"),
        "config_signature": config_signature(cfg),
        "seed": int(cfg["seed"]),
        "device": str(device),
        "split_bounds": {
            "train_end": int(bounds.train_end),
            "val_end": int(bounds.val_end),
            "total": int(bounds.total),
        },
        "evaluation_split": evaluation_split,
        "evaluation_metadata": {
            "mape_floor": "nodewise 0.01 * training-split mean absolute load; values below floor excluded",
            "metric_scale": "all reported error metrics are computed after nodewise inverse normalization",
        },
        "centralized": {},
        "federated_macro": {},
        "federated_clients": {},
    }
    for kind in ("persistence", "daily_naive"):
        centralized_set = make_dataset(
            data, data.active_indices, evaluation_split, cfg
        )
        result["centralized"][kind] = evaluate_naive_baseline(
            centralized_set, kind, batch_size, daily_period, cfg["training"]
        )
        client_sets = [
            make_dataset(data, nodes, evaluation_split, cfg) for nodes in partitions
        ]
        client_metrics = [
            evaluate_naive_baseline(
                dataset, kind, batch_size, daily_period, cfg["training"]
            )
            for dataset in client_sets
        ]
        result["federated_clients"][kind] = client_metrics
        result["federated_macro"][kind] = _macro_average(client_metrics)

    output_path(cfg, "baseline_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def centralized(cfg: dict, device: torch.device) -> dict:
    """训练集中式对照模型，并按预先声明的验证指标保存最佳参数。"""

    data = load_smartds(cfg)
    source_sha256 = archive_sha256(data.source)
    bounds = data.split_bounds(cfg["data"]["train_ratio"], cfg["data"]["val_ratio"])
    _assert_active_nodes_train_stable(data, bounds.train_end)
    nodes = data.active_indices
    train_set = make_dataset(data, nodes, "train", cfg)
    val_set = make_dataset(data, nodes, "val", cfg)
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
                loss = charbonnier_loss(
                    prediction,
                    target_device,
                    float(cfg["model"]["robust_kappa"]),
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


def _macro_average(client_metrics: list[dict[str, float]]) -> dict[str, float]:
    """对客户端等权平均，避免大客户端掩盖小客户端性能。"""

    names = client_metrics[0].keys()
    return {
        name: float(np.mean([metrics[name] for metrics in client_metrics]))
        for name in names
    }


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
) -> dict[str, float]:
    """构造 ModuleALA 本轮初始化：shared 取 global，ALA 参数逐元素插值。

    该模块机制 inspired by / adapted from FedALA 的 ALA 更新方式，
    不声称复现原始 FedALA。
    """

    ala_prefixes = ala_parameter_prefixes()
    local_prefixes = local_parameter_prefixes(False)
    current = model.state_dict()
    adapted: dict[str, torch.Tensor] = {}
    alpha_values: list[torch.Tensor] = []
    changed_values: list[torch.Tensor] = []
    for name, value in current.items():
        if _is_prefixed(name, local_prefixes):
            source = local_state[name]
        elif _is_prefixed(name, ala_prefixes):
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
) -> dict[str, float]:
    """只用 train 窗口学习 ModuleALA alpha，再写回客户端模型。"""

    device = next(model.parameters()).device
    training_config = config["training"]
    ala_names = tuple(
        name for name in model.state_dict()
        if _is_prefixed(name, ala_parameter_prefixes())
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
    previous_local_state = {
        name: value.to(device=device)
        for name, value in previous_local_state.items()
    }
    global_device_state = {
        name: value.to(device=device)
        for name, value in global_state.items()
    }
    model.eval()
    steps = 0
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
            state: dict[str, torch.Tensor] = {}
            for name, value in model.state_dict().items():
                if name in alpha_parameters:
                    local_value = previous_local_state[name].to(dtype=value.dtype)
                    global_value = global_device_state[name].to(dtype=value.dtype)
                    state[name] = local_value + alpha_parameters[name].to(dtype=value.dtype) * (global_value - local_value)
                elif name.startswith(local_parameter_prefixes(False)):
                    state[name] = previous_local_state[name].to(dtype=value.dtype)
                elif name in global_device_state:
                    state[name] = global_device_state[name].to(dtype=value.dtype)
                else:
                    state[name] = previous_local_state[name].to(dtype=value.dtype)
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
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    learned = {name: parameter.detach().cpu().clone().clamp(0.0, 1.0) for name, parameter in alpha_parameters.items()}
    alpha_state.update(learned)
    stats = _load_moduleala_initial_state(model, previous_local_state, global_state, alpha_state)
    all_alpha = torch.cat([value.reshape(-1) for value in alpha_state.values()])
    stats.update({
        "loss": float(last_loss.cpu()),
        "steps": float(steps),
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
    is_ala = algorithm == "moduleala"
    if algorithm not in {"localonly", "fedavg", "fedprox", "moduleala"}:
        raise ValueError("federated.algorithm must be LocalOnly, FedAvg, FedProx, or ModuleALA")
    if is_ala and personalized_head:
        raise ValueError("ModuleALA owns head.* personalization; personalized_head must be false")
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
    # ModuleALA 的客户端完整状态和逐元素 alpha 跨轮持久化；服务器只看 shared state。
    ala_weights: list[dict[str, torch.Tensor]] = [
        {
            name: torch.ones_like(value, dtype=torch.float32)
            for name, value in model.state_dict().items()
            if _is_prefixed(name, ala_parameter_prefixes())
        }
        for model in models
    ] if is_ala else []
    previous_local_states: list[dict[str, torch.Tensor]] = [
        {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        for model in models
    ] if is_ala else []
    ala_sample_loaders: list[DataLoader] = []
    if is_ala:
        ratio = float(cfg["federated"].get("ala_sample_ratio", 0.05))
        if not 0.0 < ratio <= 1.0:
            raise ValueError("federated.ala_sample_ratio must be in (0, 1]")
        for client_index, dataset in enumerate(train_sets):
            count = max(1, int(np.ceil(len(dataset) * ratio)))
            generator = torch.Generator().manual_seed(int(cfg["seed"]) + 1009 * (client_index + 1))
            indices = torch.randperm(len(dataset), generator=generator)[:count].tolist()
            subset = Subset(dataset, indices)
            ala_sample_loaders.append(make_data_loader(subset, train_batch_size, True, cfg["training"]))

    for round_index in range(1, int(cfg["federated"]["rounds"]) + 1):
        local_states: list[dict[str, torch.Tensor]] = []
        train_metrics: list[dict[str, float]] = []
        sample_weights: list[float] = []
        ala_round_stats: list[dict[str, float]] = []

        for client_index, (model, train_set, graph) in enumerate(
            zip(models, train_sets, graphs)
        ):
            if is_ala:
                if round_index == 1:
                    # 第一轮严格按 FedAvg 初始化，ALA 从第二轮才启用。
                    load_shared_state(model, global_state)
                    ala_round_stats.append({"executed": 0.0, "alpha_min": 1.0, "alpha_max": 1.0, "alpha_mean": 1.0, "nonzero_initialization": 0.0})
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
                    )
                    stats["executed"] = 1.0
                    ala_round_stats.append(stats)
            elif not local_only:
                load_shared_state(model, global_state)
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
            local_states.append(state)
            if is_ala:
                previous_local_states[client_index] = {
                    name: value.detach().cpu().clone() for name, value in model.state_dict().items()
                }
            train_metrics.append(metrics)
            sample_weights.append(metrics["samples"])

        if local_only:
            aggregation_audit = {
                "privacy_enabled": False,
                "clients": float(len(local_states)),
                "aggregation": "none",
            }
        elif privacy_enabled:
            global_state, aggregation_audit = aggregate_private_updates(
                local_states,
                global_state,
                clip_norm=float(privacy_cfg["clip_norm"]),
                noise_multiplier=float(privacy_cfg["noise_multiplier"]),
            )
        else:
            global_state = weighted_average(
                local_states,
                None if uniform_mean else sample_weights,
            )
            aggregation_audit = {
                "privacy_enabled": False,
                "clients": float(len(local_states)),
            }

        # ModuleALA 评估 global shared + client-local ALA/functional 状态，不能整体回载 global。
        validation_clients: list[dict[str, float]] = []
        for client_index, (model, val_set, graph) in enumerate(
            zip(models, val_sets, graphs)
        ):
            if not local_only and not is_ala:
                load_shared_state(model, global_state)
            elif is_ala:
                # 聚合后的 global 只覆盖非 ALA、非 functional 参数，保留个性化状态。
                current = model.state_dict()
                for name, value in global_state.items():
                    if name.startswith(ala_parameter_prefixes()) or name.startswith(local_parameter_prefixes(False)):
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
            "eligible_prefixes": list(ala_parameter_prefixes()) if is_ala else [],
            "sample_ratio": float(cfg["federated"].get("ala_sample_ratio", 0.0)) if is_ala else None,
            "weight_lr": float(cfg["federated"].get("ala_weight_lr", 0.0)) if is_ala else None,
            "weights": [
                {name: value.tolist() for name, value in weights.items()}
                for weights in (best_ala_weights if is_ala and best_ala_weights is not None else [])
            ],
        },
    }
    output_path(cfg, "federated_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def load_experiment_catalog() -> dict:
    """读取实验矩阵定义，所有覆盖项均可审计地写在 YAML 中。"""

    return load_config(EXPERIMENTS_CONFIG)


def experiment_config(base_cfg: dict, name: str, seed: int) -> tuple[dict, str]:
    """按实验名生成独立配置，并固定结果标签。"""

    catalog = load_experiment_catalog()
    jobs = catalog.get("experiments", {})
    if name not in jobs:
        raise ValueError(f"Unknown experiment {name!r}; available={sorted(jobs)}")
    job = jobs[name]
    overrides = deepcopy(job.get("overrides", {}))
    cfg = deep_merge(base_cfg, overrides)
    cfg["seed"] = int(seed)
    cfg["experiment_name"] = name
    tag_template = str(job.get("tag", f"{name}_seed{{seed}}"))
    cfg["experiment_tag"] = tag_template.format(name=name, seed=seed)
    return cfg, str(job.get("task", "federated"))


def run_task(task: str, cfg: dict) -> dict:
    """统一执行一个任务；实验配置和随机种子在此之前已经确定。"""

    set_seed(int(cfg["seed"]))
    # 审计只读取数据和拓扑，不应因为训练解释器没有 CUDA 而失败。
    if task == "audit":
        print(f"[PA-STFed] task={task}, device=not-required, seed={cfg['seed']}")
        return audit(cfg)

    device = resolve_device(cfg.get("device", "auto"))
    configure_torch_runtime(cfg.get("runtime", {}))
    print(f"[PA-STFed] task={task}, device={device}, seed={cfg['seed']}")
    if device.type == "cuda":
        print(
            f"[PA-STFed] torch={torch.__version__}; "
            f"torch_cuda={torch.version.cuda}; "
            f"gpu={torch.cuda.get_device_name(device)}"
        )
        props = torch.cuda.get_device_properties(device)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        print(
            f"[PA-STFed] capability={props.major}.{props.minor}; "
            f"vram_free={free_bytes / 2**30:.2f}GiB/"
            f"{total_bytes / 2**30:.2f}GiB; "
            f"amp={cfg['training'].get('amp', True)} "
            f"dtype={cfg['training'].get('amp_dtype', 'bf16')}"
        )
    else:
        print(
            f"[PA-STFed] torch={torch.__version__}; "
            "CUDA unavailable in this interpreter; training will run on CPU"
        )

    if task == "centralized":
        return centralized(cfg, device)
    if task == "federated":
        return federated(cfg, device)
    if task == "baselines":
        return baselines(cfg, device)
    raise ValueError(f"Unsupported task {task}")


def result_brief(result: dict) -> dict:
    """从完整历史中提取适合多种子汇总的核心结果。"""

    if result.get("mode") == "centralized":
        return {
            "validation": result["best_validation"],
            "train": result.get("best_train"),
            "test": result.get("test"),
            "best_epoch": result["best_epoch"],
            "selection_metric": result.get("selection_metric", "rmse"),
        }
    if result.get("mode") == "federated":
        best_validation = result.get("best_validation", {})
        return {
            "validation": best_validation.get("validation_macro", {}),
            "test_macro": result["test_macro"],
            "test_rmse_p90": result["test_rmse_p90"],
            "privacy": result["privacy"],
            "best_round": result["best_round"],
        }
    if result.get("mode") == "baseline":
        return {
            "centralized": result["centralized"],
            "federated_macro": result["federated_macro"],
        }
    return result


def config_brief(cfg: dict, task: str, name: str | None = None) -> dict:
    """打印训练前的最终配置，避免实验名、联邦算法和 DP 参数串用。"""

    federated_used = task == "federated"
    federated_config = (
        {
            "clients": int(cfg["federated"]["clients"]),
            "algorithm": cfg["federated"]["algorithm"],
            "mu": float(cfg["federated"]["mu"]),
            "uniform_mean": bool(cfg["federated"].get("uniform_mean", True)),
            "personalized_head": bool(cfg["federated"].get("personalized_head", False)),
            "ala_sample_ratio": cfg["federated"].get("ala_sample_ratio"),
            "ala_weight_lr": cfg["federated"].get("ala_weight_lr"),
            "ala_initial_epochs": cfg["federated"].get("ala_initial_epochs"),
            "ala_adapt_epochs": cfg["federated"].get("ala_adapt_epochs"),
        }
        if federated_used
        else {"used": False}
    )
    privacy_config = (
        {
            "enabled": bool(cfg["privacy"].get("enabled", False)),
            "C": float(cfg["privacy"]["clip_norm"]),
            "sigma": float(cfg["privacy"]["noise_multiplier"]),
            "delta": float(cfg["privacy"]["delta"]),
            "max_epsilon": cfg["privacy"].get("max_epsilon"),
            "planned_epsilon": (
                gaussian_rdp_epsilon(
                    noise_multiplier=float(cfg["privacy"]["noise_multiplier"]),
                    rounds=int(cfg["federated"]["rounds"]),
                    delta=float(cfg["privacy"]["delta"]),
                )
                if bool(cfg["privacy"].get("enabled", False))
                else None
            ),
        }
        if federated_used
        else {"used": False}
    )

    return {
        "experiment": name or cfg.get("experiment_name", "manual"),
        "code_revision": cfg.get("code_revision"),
        "task": task,
        "seed": int(cfg["seed"]),
        "device": cfg.get("device", "auto"),
        "data": {
            "source": cfg["data"]["source"],
            "train_ratio": float(cfg["data"]["train_ratio"]),
            "val_ratio": float(cfg["data"]["val_ratio"]),
            "test_ratio": round(
                1.0 - cfg["data"]["train_ratio"] - cfg["data"]["val_ratio"],
                10,
            ),
            "history": int(cfg["data"]["history"]),
            "horizon": int(cfg["data"]["horizon"]),
            "graph": cfg["data"]["graph"],
            "target_knn_k": int(cfg["data"].get("target_knn_k", 6)),
            "hop_radius": int(cfg["data"].get("hop_radius", 2)),
        },
        "training": {
            "batch_size": int(cfg["training"]["batch_size"]),
            "centralized_batch_size": int(
                cfg["training"].get("centralized_batch_size", cfg["training"]["batch_size"])
            ),
            "federated_batch_size": int(
                cfg["training"].get("federated_batch_size", cfg["training"]["batch_size"])
            ),
            "eval_batch_size": int(
                cfg["training"].get("eval_batch_size", cfg["training"]["batch_size"])
            ),
            "amp": bool(cfg["training"].get("amp", True)),
            "amp_dtype": str(cfg["training"].get("amp_dtype", "bf16")),
            "num_workers": int(cfg["training"].get("num_workers", 0)),
            "eval_num_workers": int(cfg["training"].get("eval_num_workers", 0)),
            "max_train_windows": cfg["training"].get("max_train_windows"),
            "max_val_windows": cfg["training"].get("max_val_windows"),
            "max_test_windows": cfg["training"].get("max_test_windows"),
            "bootstrap_block_windows": int(
                cfg["training"].get("bootstrap_block_windows", 96)
            ),
            "epochs": int(cfg["training"]["epochs"]),
            "rounds": int(cfg["federated"]["rounds"]),
            "local_epochs": int(cfg["federated"]["local_epochs"]),
            "learning_rate": float(cfg["training"]["learning_rate"]),
            "weight_decay": float(cfg["training"]["weight_decay"]),
            "selection_metric": _selection_metric(cfg["training"]),
            "evaluate_test": bool(cfg["training"].get("evaluate_test", False)),
            "scheduler": deepcopy(cfg["training"].get("scheduler", {})),
            "early_stop_min_delta": float(
                cfg["training"].get("early_stop_min_delta", 0.0)
            ),
            "patience": int(
                cfg["federated"].get("patience", cfg["training"].get("patience", 0))
                if federated_used
                else cfg["training"].get("patience", 0)
            ),
        },
        "federated": federated_config,
        "model": {
            "architecture": str(cfg["model"].get("architecture", "pa_stfed")),
            "input_dim": int(cfg["model"].get("input_dim", 5)),
            "hidden_dim": int(cfg["model"]["hidden_dim"]),
            "functional_dim": int(cfg["model"]["functional_dim"]),
            "spatial_heads": int(cfg["model"]["spatial_heads"]),
            "transformer_layers": int(cfg["model"]["transformer_layers"]),
            "transformer_heads": int(cfg["model"]["transformer_heads"]),
            "dropout": float(cfg["model"]["dropout"]),
            "robust_kappa": float(cfg["model"]["robust_kappa"]),
            **{
                key: bool(cfg["model"].get(key, True))
                for key in (
                    "use_physical",
                    "use_functional",
                    "use_spatial_gate",
                    "use_temporal_gate",
                    "use_residual_anchor",
                )
            },
        },
        # 公开基线的专属结构参数单独记录，避免 dry-run 只显示 PA-STFed 字段
        # 而让 LSTM/iTransformer 的层数和头数无法审计。
        "baselines": {
            "lstm_layers": int(cfg.get("baselines", {}).get("lstm_layers", 2)),
            "itransformer_layers": int(
                cfg.get("baselines", {}).get("itransformer_layers", 2)
            ),
            "itransformer_heads": int(
                cfg.get("baselines", {}).get("itransformer_heads", 4)
            ),
            "agcrn_embedding_dim": int(
                cfg.get("baselines", {}).get("agcrn_embedding_dim", 10)
            ),
            "agcrn_cheb_order": int(
                cfg.get("baselines", {}).get("agcrn_cheb_order", 2)
            ),
            "gwnet_embedding_dim": int(
                cfg.get("baselines", {}).get("gwnet_embedding_dim", 10)
            ),
            "gwnet_layers": int(
                cfg.get("baselines", {}).get("gwnet_layers", 4)
            ),
        },
        "privacy": privacy_config,
        "config_signature": config_signature(cfg),
        "output_tag": cfg.get("experiment_tag", ""),
    }


def show_experiment(base_cfg: dict, name: str, seed: int) -> dict:
    """解析 experiments.yaml，但不加载数据、不建模、不训练。"""

    cfg, task = experiment_config(base_cfg, name, seed)
    preview = config_brief(cfg, task, name)
    print(json.dumps(preview, ensure_ascii=False, indent=2))
    return preview


def expected_result_path(cfg: dict, task: str) -> Path:
    """返回一个实验完成后应生成的主结果文件。"""

    filenames = {
        "baselines": "baseline_result.json",
        "centralized": "centralized_result.json",
        "federated": "federated_result.json",
    }
    if task not in filenames:
        raise ValueError(f"No result filename registered for task={task!r}")
    return output_path(cfg, filenames[task])


def result_file_is_valid(path: Path, expected_signature: str | None = None) -> bool:
    """仅把包含 mode 字段的 JSON 视为可恢复结果，避免跳过半写入文件。"""

    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict) or not isinstance(payload.get("mode"), str):
        return False
    if expected_signature is not None:
        return payload.get("config_signature") == expected_signature
    return True


def load_result_brief(path: Path) -> dict | None:
    """从结果 JSON 提取汇总所需字段；损坏或不存在时返回 None。"""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("mode"), str):
        return None
    return result_brief(payload)


def collect_existing_formal_items() -> list[dict]:
    """收集已有正式结果，按 ``(experiment, seed)`` 去重。

    ``run_all`` 可能被分批调用。旧实现每次从空列表开始写 manifest，导致后
    一批实验覆盖前一批记录。这里同时读取旧 manifest 和结果目录；结果文件
    是最终事实来源，可补回 manifest 中因中断或旧版本逻辑遗漏的完成项。
    """

    by_key: dict[tuple[str, int], dict] = {}
    manifest_path = OUTPUTS / "all_manifest.json"
    if manifest_path.exists():
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        for item in payload.get("items", []) if isinstance(payload, dict) else []:
            if not isinstance(item, dict):
                continue
            if item.get("status") not in {"completed", "skipped_existing"}:
                continue
            try:
                key = (str(item["experiment"]), int(item["seed"]))
            except (KeyError, TypeError, ValueError):
                continue
            brief = item.get("result")
            if brief is None:
                brief = load_result_brief(Path(str(item.get("result_path", ""))))
            if brief is None:
                continue
            normalized = dict(item)
            normalized["result"] = brief
            by_key[key] = normalized

    mode_to_task = {
        "centralized": "centralized",
        "federated": "federated",
        "baseline": "baselines",
    }
    suffix_by_task = {
        "centralized": "_centralized_result.json",
        "federated": "_federated_result.json",
        "baselines": "_baseline_result.json",
    }
    for path in sorted(OUTPUTS.glob("*_result.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        mode = str(payload.get("mode", ""))
        task = mode_to_task.get(mode)
        if task is None or payload.get("experiment_name") is None:
            continue
        try:
            key = (str(payload["experiment_name"]), int(payload["seed"]))
        except (KeyError, TypeError, ValueError):
            continue
        # 已有 manifest 项优先；结果目录只负责补齐缺失的正式记录。
        if key in by_key:
            continue
        brief = result_brief(payload)
        suffix = suffix_by_task[task]
        tag = path.name[:-len(suffix)] if path.name.endswith(suffix) else path.stem
        by_key[key] = {
            "experiment": key[0],
            "seed": key[1],
            "task": task,
            "tag": tag,
            "result_path": str(path),
            "status": "completed",
            "result": brief,
        }

    return [by_key[key] for key in sorted(by_key)]


def summarize_all_items(items: list[dict]) -> dict:
    """将 all_manifest 中的成功结果按实验名汇总为均值、标准差和样本数。"""

    grouped: dict[str, list[dict]] = {}
    for item in items:
        if item.get("status") not in {"completed", "skipped_existing"}:
            continue
        brief = item.get("result") or load_result_brief(Path(item["result_path"]))
        if brief is None:
            continue
        item["result"] = brief
        grouped.setdefault(str(item["experiment"]), []).append(item)

    def summarize_metrics(metric_records: list[dict]) -> dict[str, dict[str, float | int]]:
        """汇总一组同口径的验证或测试指标。"""

        metric_names = sorted({key for record in metric_records for key in record})
        metrics: dict[str, dict[str, float | int]] = {}
        for metric_name in metric_names:
            values = [
                float(record[metric_name])
                for record in metric_records
                if isinstance(record.get(metric_name), (int, float))
                and np.isfinite(float(record[metric_name]))
            ]
            if values:
                metrics[metric_name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "n": len(values),
                }
        return metrics

    summary: dict[str, dict] = {}
    for name, records in grouped.items():
        first = records[0]
        validation_records: list[dict] | None = None
        if first["task"] == "centralized":
            metric_records = [
                record["result"].get("test") or {} for record in records
            ]
            validation_records = [
                record["result"].get("validation", {}) for record in records
            ]
        elif first["task"] == "federated":
            test_records = [record["result"].get("test_macro") for record in records]
            use_validation = not all(isinstance(value, dict) for value in test_records)
            metric_records = [
                {
                    **(
                        record["result"].get("validation", {})
                        if use_validation
                        else record["result"].get("test_macro", {})
                    ),
                    "rmse_p90": (
                        None
                        if use_validation
                        else record["result"].get("test_rmse_p90")
                    ),
                }
                for record in records
            ]
        else:
            summary[name] = {
                "task": first["task"],
                "seeds": [record["seed"] for record in records],
                "n": len(records),
                "raw": [record["result"] for record in records],
            }
            continue

        summary[name] = {
            "task": first["task"],
            "seeds": [record["seed"] for record in records],
            "n": len(records),
            "metrics": summarize_metrics(metric_records),
        }
        if validation_records is not None:
            summary[name]["validation_metrics"] = summarize_metrics(validation_records)

    return summary


def run_all(
    base_cfg: dict,
    seeds: list[int],
    resume: bool = False,
    skip_dp: bool = False,
    dry_run: bool = False,
    experiment_names: list[str] | None = None,
) -> dict:
    """顺序执行完整实验矩阵，并支持 AutoDL 断点续跑。

    每个实验使用 experiments.yaml 中声明的 seed_count；当前正式矩阵注册
    2026/2027/2028 三个种子，单种子筛查任务显式声明 seed_count=1。
    """

    catalog = load_experiment_catalog()
    jobs = catalog.get("experiments", {})
    # 默认矩阵只纳入 include_in_all=true 的结项主线；额外任务须显式指定。
    names = (
        [name for name, job in jobs.items() if bool(job.get("include_in_all", True))]
        if experiment_names is None
        else list(experiment_names)
    )
    unknown = [name for name in names if name not in jobs]
    if unknown:
        raise ValueError(
            f"Unknown experiment(s): {unknown}; available={list(jobs)}"
        )
    if skip_dp:
        names = [name for name in names if not name.startswith("dp_")]
    if not names:
        raise ValueError("No experiments selected")

    if not dry_run and str(base_cfg.get("device", "auto")).lower() == "cuda":
        # 在第一个任务开始前就终止，避免服务器使用 CPU 版 PyTorch 时
        # 仍然启动全部实验并耗费数小时生成一批 CPU 结果。
        if not torch.cuda.is_available():
            raise RuntimeError(
                "device=cuda but torch.cuda.is_available() is false. "
                "Install a CUDA-enabled PyTorch build in the AutoDL environment "
                "and rerun; no experiment was started."
            )

    def selected_job_seeds(name: str) -> list[int]:
        count = int(jobs[name].get("seed_count", len(seeds)))
        if count < 1:
            raise ValueError(f"seed_count must be positive for experiment {name!r}")
        # 命令行可显式提供更少种子用于烟测；正式命令使用清单注册的三个种子。
        return seeds[: min(count, len(seeds))]

    OUTPUTS.mkdir(exist_ok=True)
    # include_in_all 只控制默认是否加入矩阵，不再代表“调参模式”。
    # 显式指定的任意实验都统一写入正式 manifest，避免旧调参逻辑串入结果汇总。
    manifest_path = OUTPUTS / "all_manifest.json"
    existing_items = collect_existing_formal_items()
    existing_experiments = {str(item["experiment"]) for item in existing_items}
    existing_seeds = {int(item["seed"]) for item in existing_items}
    all_experiment_names = sorted(existing_experiments | set(names))
    merged_seed_counts: dict[str, int] = {}
    for experiment_name in all_experiment_names:
        merged_seeds = {
            int(item["seed"])
            for item in existing_items
            if str(item.get("experiment")) == experiment_name
        }
        if experiment_name in names:
            merged_seeds.update(selected_job_seeds(experiment_name))
        merged_seed_counts[experiment_name] = len(merged_seeds)
    manifest = {
        "run_kind": "formal",
        "status": "dry_run" if dry_run else "running",
        # manifest 是增量账本：保留旧实验，并将本次请求的配置并入元数据。
        "seeds": sorted(existing_seeds | {int(seed) for seed in seeds}),
        "experiments": all_experiment_names,
        "seed_count_by_experiment": merged_seed_counts,
        "resume": resume,
        "skip_dp": skip_dp,
        "dry_run": dry_run,
        "items": existing_items,
    }

    def upsert_item(item: dict) -> None:
        """按 experiment+seed 更新一条记录，避免分批运行产生重复项。"""

        key = (str(item["experiment"]), int(item["seed"]))
        for index, previous in enumerate(manifest["items"]):
            if (str(previous.get("experiment")), int(previous.get("seed", -1))) == key:
                manifest["items"][index] = item
                return
        manifest["items"].append(item)

    def save_manifest() -> None:
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    if dry_run:
        for name in names:
            job_seeds = selected_job_seeds(name)
            for seed in job_seeds:
                cfg, task = experiment_config(base_cfg, name, int(seed))
                print(json.dumps(config_brief(cfg, task, name), ensure_ascii=False, indent=2))
                upsert_item({
                    "experiment": name,
                    "seed": int(seed),
                    "task": task,
                    "tag": cfg["experiment_tag"],
                    "status": "dry_run",
                    "result_path": str(expected_result_path(cfg, task)),
                })
        save_manifest()
        summary_path = OUTPUTS / "all_summary.json"
        summary_path.write_text(
            json.dumps(summarize_all_items(manifest["items"]), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        manifest["summary_path"] = str(summary_path)
        manifest["failed_count"] = 0
        save_manifest()
        print(f"[PA-STFed][ALL] dry-run complete; manifest={manifest_path}")
        return manifest

    print("[PA-STFed][ALL] start audit")
    audit_cfg = deepcopy(base_cfg)
    audit_cfg["seed"] = int(seeds[0])
    audit_cfg.pop("experiment_name", None)
    audit_cfg.pop("experiment_tag", None)
    audit(cfg=audit_cfg)

    for name in names:
        job_seeds = selected_job_seeds(name)
        for seed in job_seeds:
            cfg, task = experiment_config(base_cfg, name, int(seed))
            result_path = expected_result_path(cfg, task)
            item = {
                "experiment": name,
                "seed": int(seed),
                "task": task,
                "tag": cfg["experiment_tag"],
                "result_path": str(result_path),
            }

            if resume and result_file_is_valid(result_path, config_signature(cfg)):
                item["status"] = "skipped_existing"
                item["result"] = load_result_brief(result_path)
                upsert_item(item)
                save_manifest()
                print(f"[PA-STFed][ALL] skip existing {name} seed={seed}")
                continue

            print(f"\n[PA-STFed][ALL] start {name} seed={seed}")
            print(json.dumps(config_brief(cfg, task, name), ensure_ascii=False, indent=2))
            try:
                result = run_task(task, cfg)
                item["status"] = "completed"
                item["result"] = result_brief(result)
                print(f"[PA-STFed][ALL] completed {name} seed={seed}")
            except Exception as exc:  # 保留错误并继续其余实验，便于远程无人值守。
                item["status"] = "failed"
                item["error"] = repr(exc)
                item["traceback"] = traceback.format_exc()
                print(f"[PA-STFed][ALL] failed {name} seed={seed}: {exc}")
            upsert_item(item)
            save_manifest()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    failures = [item for item in manifest["items"] if item["status"] == "failed"]
    manifest["status"] = "failed" if failures else "completed"
    manifest["failed_count"] = len(failures)
    summary_path = OUTPUTS / "all_summary.json"
    summary_path.write_text(
        json.dumps(summarize_all_items(manifest["items"]), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    manifest["summary_path"] = str(summary_path)
    save_manifest()
    print(json.dumps({
        "status": manifest["status"],
        "manifest": str(manifest_path),
        "completed": sum(item["status"] in {"completed", "skipped_existing"} for item in manifest["items"]),
        "failed": len(failures),
    }, ensure_ascii=False, indent=2))
    if failures:
        raise RuntimeError(f"{len(failures)} experiments failed; inspect {manifest_path}")
    return manifest


def resolve_seeds(args: argparse.Namespace, base_cfg: dict, catalog: dict) -> list[int]:
    """解析正式矩阵使用的随机种子列表。"""

    if args.seeds:
        seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    elif args.seed is not None:
        seeds = [int(args.seed)]
    else:
        seeds = [int(value) for value in catalog.get("seeds", [base_cfg["seed"]])]
    if not seeds:
        raise ValueError("At least one seed is required")
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser(description="PA-STFed 可复现实验入口")
    parser.add_argument(
        "task",
        choices=["all", "audit", "show"],
        nargs="?",
        default="all",
        help="正式任务类型；无参数时执行完整实验矩阵",
    )
    parser.add_argument("--name", help="实验矩阵中的实验名")
    parser.add_argument("--seed", type=int, help="覆盖单次运行的随机种子")
    parser.add_argument("--seeds", help="正式矩阵的种子列表，例如 2026,2027,2028")
    parser.add_argument("--hidden", type=int, help="覆盖基础配置的 hidden_dim")
    parser.add_argument("--learning-rate", type=float, help="覆盖基础配置的 learning_rate")
    parser.add_argument("--history", type=int, help="覆盖基础配置的历史窗口")
    parser.add_argument("--dropout", type=float, help="覆盖基础配置的 dropout")
    parser.add_argument("--weight-decay", type=float, help="覆盖 AdamW weight_decay")
    parser.add_argument("--kappa", type=float, help="覆盖 Charbonnier kappa")
    parser.add_argument("--hop-radius", type=int, help="覆盖物理图跳数")
    parser.add_argument(
        "--residual-anchor",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="开启或关闭残差锚定；关闭形式为 --no-residual-anchor",
    )
    parser.add_argument(
        "--experiments",
        help="用英文逗号分隔要运行的实验名；省略时运行正式实验矩阵。",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="all 模式跳过已经存在且可解析的结果文件",
    )
    parser.add_argument(
        "--skip-dp",
        action="store_true",
        help="all 模式跳过 DP 实验，仅运行非 DP 矩阵",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="all 模式只打印最终配置，不执行审计和训练",
    )
    args = parser.parse_args()

    # 正式 all 矩阵必须由 experiments.yaml 唯一决定。保留这些 CLI 参数仅
    # 便于 show/audit 兼容旧命令，但禁止它们悄悄生成无法与主矩阵比较的结果。
    if args.task == "all":
        ad_hoc = {
            "hidden": args.hidden,
            "learning_rate": args.learning_rate,
            "history": args.history,
            "dropout": args.dropout,
            "weight_decay": args.weight_decay,
            "kappa": args.kappa,
            "hop_radius": args.hop_radius,
            "residual_anchor": args.residual_anchor,
        }
        used = [name for name, value in ad_hoc.items() if value is not None]
        if used:
            raise ValueError(
                "正式 all 模式禁止临时超参覆盖 "
                f"{used}; 请在 experiments.yaml 注册独立实验后再运行。"
            )

    base_cfg = load_project_config()
    if args.hidden is not None:
        base_cfg["model"]["hidden_dim"] = int(args.hidden)
        # 常用的 32/64/128 宽度分别对应 4/4/8 个 Transformer 头。
        base_cfg["model"]["transformer_heads"] = 8 if int(args.hidden) >= 128 else 4
        base_cfg["model"]["spatial_heads"] = 4
    if args.learning_rate is not None:
        base_cfg["training"]["learning_rate"] = float(args.learning_rate)
    if args.history is not None:
        base_cfg["data"]["history"] = int(args.history)
    if args.dropout is not None:
        base_cfg["model"]["dropout"] = float(args.dropout)
    if args.weight_decay is not None:
        base_cfg["training"]["weight_decay"] = float(args.weight_decay)
    if args.kappa is not None:
        base_cfg["model"]["robust_kappa"] = float(args.kappa)
    if args.hop_radius is not None:
        base_cfg["data"]["hop_radius"] = int(args.hop_radius)
    if args.residual_anchor is not None:
        base_cfg["model"]["use_residual_anchor"] = bool(args.residual_anchor)
    if args.task == "audit":
        if args.seed is not None:
            base_cfg["seed"] = int(args.seed)
        result = run_task("audit", base_cfg)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.task == "show":
        if not args.name:
            raise ValueError("show requires --name from experiments.yaml")
        seed = args.seed if args.seed is not None else int(
            load_experiment_catalog().get("seeds", [base_cfg["seed"]])[0]
        )
        show_experiment(base_cfg, args.name, seed)
        return

    if args.task == "all":
        catalog = load_experiment_catalog()
        seeds = resolve_seeds(args, base_cfg, catalog)
        experiment_names = (
            [value.strip() for value in args.experiments.split(",") if value.strip()]
            if args.experiments
            else None
        )
        run_all(
            base_cfg,
            seeds,
            resume=args.resume,
            skip_dp=args.skip_dp,
            dry_run=args.dry_run,
            experiment_names=experiment_names,
        )
        return


if __name__ == "__main__":
    main()
