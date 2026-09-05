"""SmartDS 空间客户端上的 FedAvg、FedProx 与客户端级 DP 聚合。"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch.utils.data import DataLoader

from config import autocast_context, make_grad_scaler
from data import GraphView, LoadWindowDataset, make_data_loader
from metrics import mae, mape, rmse, smape, wape
from model import (
    PA_STFed,
    load_shared_state,
    local_parameter_prefixes,
    shared_state_dict,
)
from privacy import aggregate_mean_updates


def charbonnier_loss(prediction: torch.Tensor, target: torch.Tensor, kappa: float) -> torch.Tensor:
    """平滑稳健损失：小残差近似二次，大残差区近似线性。"""

    return charbonnier_elementwise(prediction, target, kappa).mean()


def charbonnier_elementwise(
    prediction: torch.Tensor, target: torch.Tensor, kappa: float
) -> torch.Tensor:
    """返回逐元素 Charbonnier 值，供节点/馈线层级加权损失复用。"""

    if kappa <= 0:
        raise ValueError("kappa must be positive")
    error = prediction - target
    return kappa**2 * torch.sqrt(1.0 + (error / kappa) ** 2) - kappa**2


def scale_aware_charbonnier_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    kappa: float,
    scale: torch.Tensor,
    feeder_loss_weight: float = 0.0,
) -> torch.Tensor:
    """按训练集节点 IQR 加权的节点损失，可选加入馈线聚合损失。

    ``prediction``/``target`` 的形状为 ``[batch, horizon, nodes]``。节点项
    先对节点按训练集 IQR 加权，再对 batch 和 horizon 求平均；馈线项使用
    同一组尺度加权的归一化残差，不引入额外未来标签。
    """

    if prediction.shape != target.shape or prediction.ndim < 1:
        raise ValueError("prediction and target must have identical non-empty shapes")
    scale = torch.as_tensor(scale, dtype=prediction.dtype, device=prediction.device)
    if scale.ndim != 1 or scale.numel() != prediction.shape[-1]:
        raise ValueError(
            f"scale must have one value per node; got shape={tuple(scale.shape)}, "
            f"nodes={prediction.shape[-1]}"
        )
    if not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("training IQR scale must be finite and strictly positive")
    denominator = scale.sum().clamp_min(torch.finfo(scale.dtype).eps)
    node_weights = scale.reshape(*([1] * (prediction.ndim - 1)), -1)
    elementwise = charbonnier_elementwise(prediction, target, kappa)
    node_loss = (elementwise * node_weights).sum(dim=-1) / denominator
    total = node_loss.mean()
    if feeder_loss_weight:
        residual = prediction - target
        feeder_residual = (residual * node_weights).sum(dim=-1) / denominator
        feeder_loss = charbonnier_elementwise(
            feeder_residual, torch.zeros_like(feeder_residual), kappa
        ).mean()
        total = total + float(feeder_loss_weight) * feeder_loss
    return total


def scale_aware_l1_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """按训练集节点 IQR 加权的绝对误差 numerator 对齐损失。

    ``prediction``/``target`` 的形状为 ``[batch, horizon, nodes]``。分母
    只使用同一组训练集 IQR 节点权重之和，不使用 batch target magnitude，
    因而对应 WAPE 的原始负荷绝对误差 numerator 的训练域形式。
    """

    if prediction.shape != target.shape or prediction.ndim < 1:
        raise ValueError("prediction and target must have identical non-empty shapes")
    scale = torch.as_tensor(scale, dtype=prediction.dtype, device=prediction.device)
    if scale.ndim != 1 or scale.numel() != prediction.shape[-1]:
        raise ValueError(
            f"scale must have one value per node; got shape={tuple(scale.shape)}, "
            f"nodes={prediction.shape[-1]}"
        )
    if not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("training IQR scale must be finite and strictly positive")
    denominator = scale.sum().clamp_min(torch.finfo(scale.dtype).eps)
    node_weights = scale.reshape(*([1] * (prediction.ndim - 1)), -1)
    weighted_abs_error = (torch.abs(prediction - target) * node_weights).sum(dim=-1)
    return (weighted_abs_error / denominator).mean()


def metric_summary(
    prediction: torch.Tensor,
    target: torch.Tensor,
    dataset: LoadWindowDataset | None = None,
) -> dict[str, float]:
    """计算误差指标；传入数据集时先还原到原始负荷尺度。"""

    prediction = prediction.detach()
    target = target.detach()
    if dataset is not None:
        prediction = dataset.denormalize(prediction)
        target = dataset.denormalize(target)
    mape_floor = getattr(dataset, "mape_floor", None) if dataset is not None else None
    if mape_floor is None:
        mape_valid = target.abs() > 1e-6
    else:
        floor = torch.as_tensor(mape_floor, dtype=target.dtype, device=target.device)
        floor = floor.reshape(*([1] * (target.ndim - 1)), -1)
        mape_valid = target.abs() >= floor
    return {
        "mae": float(mae(prediction, target).cpu()),
        "rmse": float(rmse(prediction, target).cpu()),
        "wape": float(wape(prediction, target).cpu()),
        "smape": float(smape(prediction, target).cpu()),
        "mape": float(mape(prediction, target, threshold=mape_floor).cpu()),
        # MAPE 排除零目标；记录覆盖率，避免脱离统计口径解读该百分比。
        "mape_valid_ratio": float(mape_valid.float().mean().cpu()),
    }


def _shared_squared_distance(
    model: PA_STFed,
    global_state: dict[str, torch.Tensor],
    personalized_head: bool = False,
) -> torch.Tensor:
    """直接累加共享参数与服务器参数的平方距离，避免反复拼接大向量。"""

    distance = torch.zeros((), device=next(model.parameters()).device)
    local_prefixes = local_parameter_prefixes(personalized_head)
    for name, parameter in model.named_parameters():
        if name.startswith(local_prefixes):
            continue
        reference = global_state[name].to(
            device=parameter.device,
            dtype=parameter.dtype,
        )
        distance = distance + (parameter - reference).square().sum()
    return distance


def train_local(
    model: PA_STFed,
    dataset: LoadWindowDataset,
    graph: GraphView,
    config: dict,
    global_state: dict[str, torch.Tensor] | None = None,
    loader: DataLoader | None = None,
    graph_tensors_device: tuple[torch.Tensor, torch.Tensor] | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scaler=None,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """执行一个客户端的本地训练，并返回共享参数与本地指标。"""

    device = next(model.parameters()).device
    training_config = config["training"]
    if loader is None:
        loader = make_data_loader(
            dataset,
            int(training_config.get("federated_batch_size", training_config["batch_size"])),
            True,
            training_config,
        )
    if optimizer is None:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config["training"]["learning_rate"]),
            weight_decay=float(config["training"]["weight_decay"]),
        )
    if scaler is None:
        scaler = make_grad_scaler(training_config, device)
    if graph_tensors_device is None:
        adjacency = torch.from_numpy(graph.adjacency).to(device)
        edge_features = torch.from_numpy(graph.edge_features).to(device)
    else:
        adjacency, edge_features = graph_tensors_device
    algorithm = str(config["federated"].get("algorithm", "FedAvg")).lower()
    mu = float(config["federated"].get("mu", 0.0))
    if algorithm != "fedprox" and abs(mu) > 0.0:
        raise ValueError(
            f"proximal mu must be 0 for algorithm={config['federated'].get('algorithm')}; got {mu}"
        )
    personalized_head = bool(config["federated"].get("personalized_head", False))
    proximal_state = (
        {name: value.to(device=device) for name, value in global_state.items()}
        if algorithm == "fedprox" and global_state is not None and mu > 0
        else None
    )
    model.train()
    transfer_non_blocking = bool(
        training_config.get("pin_memory", False)
    ) and device.type == "cuda"
    loss_sum = torch.zeros((), device=device)
    batches = 0
    for _ in range(int(config["federated"].get("local_epochs", config["training"]["epochs"]))):
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=transfer_non_blocking)
            targets = targets.to(device, non_blocking=transfer_non_blocking)
            with autocast_context(training_config, device):
                output = model(inputs, adjacency, edge_features)["prediction"]
                loss = charbonnier_loss(
                    output, targets, float(config["model"]["robust_kappa"])
                )
            if bool(training_config.get("smoke_checks", False)):
                if output.shape != targets.shape:
                    raise RuntimeError(
                        f"smoke shape mismatch: prediction={output.shape}, "
                        f"target={targets.shape}"
                    )
                if not torch.isfinite(output).all() or not torch.isfinite(loss):
                    raise RuntimeError("smoke detected NaN/Inf in prediction or loss")
            if proximal_state is not None:
                # 标准 FedProx 近端项为 mu/2 * ||theta_k - theta_global||_2^2。
                proximal = _shared_squared_distance(model, proximal_state, personalized_head)
                loss = loss + 0.5 * mu * proximal
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config["training"]["grad_clip_norm"])
            )
            if bool(training_config.get("smoke_checks", False)) and not torch.isfinite(
                gradient_norm
            ):
                raise RuntimeError("smoke detected NaN/Inf in gradients")
            scaler.step(optimizer)
            scaler.update()
            # 每个 batch 都 .cpu() 会强制同步 CUDA；只在本地 epoch 结束时取一次标量。
            loss_sum += loss.detach()
            batches += 1
    state = shared_state_dict(model, personalized_head=personalized_head)
    return state, {
        "loss": float((loss_sum / max(batches, 1)).cpu()),
        "samples": float(len(dataset)),
        "batches": float(batches),
    }


def weighted_average(states: Iterable[dict[str, torch.Tensor]], weights: list[float] | None = None) -> dict[str, torch.Tensor]:
    """对共享模型状态做等权或样本数加权平均。"""

    states = list(states)
    if not states:
        raise ValueError("states must not be empty")
    if weights is None:
        weights = [1.0 / len(states)] * len(states)
    if len(weights) != len(states) or any(weight < 0 for weight in weights):
        raise ValueError("weights must be non-negative and match the number of states")
    total = sum(weights)
    if total <= 0:
        raise ValueError("sum of weights must be positive")
    normalized = [weight / total for weight in weights]
    return {
        name: sum((state[name] * weight for state, weight in zip(states, normalized)))
        for name in states[0]
    }


def aggregate_private_updates(
    states: list[dict[str, torch.Tensor]],
    global_state: dict[str, torch.Tensor],
    clip_norm: float,
    noise_multiplier: float,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """裁剪客户端共享更新，并在客户端均值尺度上加入高斯噪声。

    每个客户端先形成 delta_k = theta_k - theta_global，再对完整共享更新向量
    做 L2 裁剪。K 个客户端求均值后，噪声标准差为 sigma*C/K，因此作用在
    全局均值上的噪声方差为 sigma^2*C^2/K^2。
    """

    if not states:
        raise ValueError("states must not be empty")
    names = tuple(global_state)
    updates = [
        torch.cat(
            [
                (state[name] - global_state[name]).reshape(-1).float()
                for name in names
            ]
        )
        for state in states
    ]
    mean_update, audit = aggregate_mean_updates(
        updates,
        clip_norm=clip_norm,
        noise_multiplier=noise_multiplier,
    )

    aggregated: dict[str, torch.Tensor] = {}
    offset = 0
    for name in names:
        reference = global_state[name]
        count = reference.numel()
        delta = mean_update[offset : offset + count].reshape(reference.shape)
        aggregated[name] = (reference.float() + delta).to(dtype=reference.dtype)
        offset += count
    audit["privacy_enabled"] = True
    return aggregated, audit


def build_client_model(template: PA_STFed, node_count: int) -> PA_STFed:
    """复制共享架构，同时允许每个客户端具有不同节点数。"""

    return PA_STFed(
        node_count=node_count,
        history=template.history,
        horizon=template.horizon,
        input_dim=template.input_projection.in_features,
        hidden_dim=template.input_projection.out_features,
        functional_dim=template.functional.embedding_1.shape[1],
        spatial_heads=template.physical.heads,
        transformer_layers=len(template.temporal.encoder.layers),
        transformer_heads=template.temporal.encoder.layers[0].self_attn.num_heads,
        dropout=template.dropout.p,
        use_physical=template.use_physical,
        use_functional=template.use_functional,
        use_spatial_gate=template.use_spatial_gate,
        use_temporal_gate=template.use_temporal_gate,
        use_residual_anchor=template.use_residual_anchor,
    )
