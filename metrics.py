"""负荷预测指标、门控饱和度诊断与表征相似度。"""

from __future__ import annotations

import torch
from torch import Tensor


def mae(prediction: Tensor, target: Tensor) -> Tensor:
    return (prediction - target).abs().mean()


def rmse(prediction: Tensor, target: Tensor) -> Tensor:
    return torch.sqrt((prediction - target).square().mean())


def wape(prediction: Tensor, target: Tensor, eps: float = 1e-6) -> Tensor:
    return 100.0 * (prediction - target).abs().sum() / (target.abs().sum() + eps)


def smape(prediction: Tensor, target: Tensor, eps: float = 1e-6) -> Tensor:
    numerator = 2.0 * (prediction - target).abs()
    denominator = prediction.abs() + target.abs() + eps
    return 100.0 * (numerator / denominator).mean()


def mape(
    prediction: Tensor,
    target: Tensor,
    eps: float = 1e-6,
    threshold: Tensor | float | None = None,
) -> Tensor:
    """计算带零值保护的 MAPE（百分数）。

    负荷序列可能在部分时刻出现零目标值；这些位置不具有有限的相对误差，
    因此从 MAPE 的分母统计中排除，而不是用一个很小的数制造虚高误差。
    WAPE 和 SMAPE 仍作为论文主指标，用于避免零值和低负荷区间对结论的支配。
    """

    if threshold is None:
        valid = target.abs() > eps
    else:
        floor = torch.as_tensor(threshold, dtype=target.dtype, device=target.device)
        floor = floor.reshape(*([1] * (target.ndim - 1)), -1)
        valid = target.abs() >= floor
    if not torch.any(valid):
        return torch.full((), float("nan"), dtype=prediction.dtype, device=prediction.device)
    relative_error = (prediction - target).abs() / target.abs().clamp_min(eps)
    return 100.0 * relative_error[valid].mean()


def gate_diagnostics(values: Tensor, threshold: float = 0.05, eps: float = 1e-6) -> dict[str, float]:
    """统计门控分布与两端饱和率。

    ``sat0``/``sat1`` 是论文诊断协议采用的近边界率（阈值默认为 0.05）；
    ``exact0``/``exact1`` 用于区分真正的数值边界常数。两类量同时记录，
    避免把 sigmoid 的近饱和误写成严格等于 0 或 1。
    """

    flat = values.detach().float().reshape(-1).cpu()
    if flat.numel() == 0:
        raise ValueError("gate diagnostics require at least one value")
    # float32 中 1-1e-8 会舍入回 1，随后出现 0*log(0)=NaN。
    # 使用可表示的边界并通过 xlogy 定义 0*log(0)=0，保证诊断稳定。
    probability = flat.clamp(min=eps, max=1.0 - eps)
    entropy = -(
        torch.special.xlogy(probability, probability)
        + torch.special.xlogy(1.0 - probability, 1.0 - probability)
    ).mean()
    quantiles = torch.quantile(flat, torch.tensor([0.01, 0.05, 0.50, 0.95, 0.99]))
    return {
        "mean": float(flat.mean()),
        "std": float(flat.std(unbiased=False)),
        "q01": float(quantiles[0]),
        "q05": float(quantiles[1]),
        "q50": float(quantiles[2]),
        "q95": float(quantiles[3]),
        "q99": float(quantiles[4]),
        "sat0": float((flat <= threshold).float().mean()),
        "sat1": float((flat >= 1.0 - threshold).float().mean()),
        "exact0": float((flat == 0.0).float().mean()),
        "exact1": float((flat == 1.0).float().mean()),
        "entropy": float(entropy),
    }


def linear_cka(left: Tensor, right: Tensor, eps: float = 1e-12) -> float:
    """用线性 CKA 衡量物理图与功能图表征是否出现冗余退化。"""

    x = left.detach().reshape(-1, left.shape[-1]).float()
    y = right.detach().reshape(-1, right.shape[-1]).float()
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    numerator = torch.linalg.matrix_norm(x.T @ y, ord="fro").square()
    denominator = torch.linalg.matrix_norm(x.T @ x, ord="fro") * torch.linalg.matrix_norm(y.T @ y, ord="fro")
    return float(numerator / (denominator + eps))
