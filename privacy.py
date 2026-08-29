"""客户端更新裁剪、中心高斯聚合和 q=1 的 Rényi DP 会计。"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor


def clip_update(update: Tensor, clip_norm: float) -> tuple[Tensor, float]:
    """把单个客户端的完整模型更新裁剪到 L2 范数 C。"""

    if clip_norm <= 0:
        raise ValueError("clip_norm must be positive")
    norm = torch.linalg.vector_norm(update)
    scale = min(1.0, float(clip_norm / max(float(norm), 1e-12)))
    return update * scale, scale


def aggregate_mean_updates(
    updates: Sequence[Tensor],
    clip_norm: float,
    noise_multiplier: float = 0.0,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, dict[str, float]]:
    """裁剪客户端更新并在等权均值上加入中心高斯噪声。

    噪声直接在均值尺度采样，其标准差为 sigma*C/K，方差为
    sigma^2*C^2/K^2。若服务器也属于威胁模型，调用方还必须把该机制置于
    安全聚合之后；本函数本身不实现密码学安全聚合。
    """

    if not updates:
        raise ValueError("updates must not be empty")
    if noise_multiplier < 0:
        raise ValueError("noise_multiplier must be non-negative")

    clipped: list[Tensor] = []
    scales: list[float] = []
    for update in updates:
        value, scale = clip_update(update, clip_norm)
        clipped.append(value)
        scales.append(scale)

    mean = torch.stack(clipped).mean(dim=0)
    if noise_multiplier > 0:
        std = noise_multiplier * clip_norm / len(clipped)
        noise = torch.randn(
            mean.shape,
            generator=generator,
            device=mean.device,
            dtype=mean.dtype,
        ) * std
        mean = mean + noise
    else:
        std = 0.0

    return mean, {
        "clients": float(len(clipped)),
        "clip_rate": float(sum(scale < 1.0 for scale in scales) / len(scales)),
        "mean_scale": float(sum(scales) / len(scales)),
        "noise_std_on_mean": float(std),
    }


def gaussian_rdp_epsilon(
    noise_multiplier: float,
    rounds: int,
    delta: float,
    orders: Sequence[float] | None = None,
) -> float:
    """计算全客户端参与（q=1）高斯机制组合后的保守 epsilon。

    单轮高斯机制在 Rényi 阶 alpha 下满足
    epsilon_alpha = alpha/(2*sigma^2)。R 轮组合后线性累加，再转换为
    (epsilon, delta)-DP。该公式不能直接用于 q<1 的客户端抽样。
    """

    if noise_multiplier <= 0:
        return float("inf")
    if rounds < 1:
        raise ValueError("rounds must be positive")
    if not 0 < delta < 1:
        raise ValueError("delta must be in (0, 1)")
    if orders is None:
        orders = tuple(float(order) for order in range(2, 129))

    candidates: list[float] = []
    for order in orders:
        if order <= 1:
            continue
        rdp = rounds * order / (2.0 * noise_multiplier**2)
        epsilon = rdp + math.log(1.0 / delta) / (order - 1.0)
        candidates.append(epsilon)
    if not candidates:
        raise ValueError("orders must contain at least one value greater than 1")
    return float(min(candidates))
