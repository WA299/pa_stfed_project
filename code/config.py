"""配置读取、随机种子固定和训练设备选择。"""

from __future__ import annotations

import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def configure_torch_runtime(runtime_config: dict[str, Any] | None = None) -> None:
    """在创建模型前一次性配置 CUDA 数学内核。

    TF32 只用于 Ampere 及更新架构 GPU 的矩阵乘法，不改变张量形状、数据
    划分或实验统计定义；它可以减少 Tensor Core 空转。若要求确定性复现，
    则关闭 cuDNN 自动基准搜索，避免不同运行选择不同内核。
    """

    runtime_config = runtime_config or {}
    if not torch.cuda.is_available():
        return

    allow_tf32 = bool(runtime_config.get("allow_tf32", True))
    deterministic = bool(runtime_config.get("deterministic", True))
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = bool(
        runtime_config.get("cudnn_benchmark", not deterministic)
    ) and not deterministic


def amp_dtype(training_config: dict[str, Any], device: torch.device) -> torch.dtype | None:
    """根据配置返回 CUDA 自动混合精度类型；未启用时返回 ``None``。"""

    if device.type != "cuda" or not bool(training_config.get("amp", True)):
        return None
    requested = str(training_config.get("amp_dtype", "bf16")).lower()
    if requested in {"bf16", "bfloat16"} and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if requested in {"fp16", "float16", "half"} or not torch.cuda.is_bf16_supported():
        return torch.float16
    raise ValueError(f"Unsupported training.amp_dtype={requested!r}")


def autocast_context(training_config: dict[str, Any], device: torch.device):
    """创建同时兼容 CUDA 和 CPU 的自动混合精度上下文。"""

    dtype = amp_dtype(training_config, device)
    if dtype is None:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def make_grad_scaler(training_config: dict[str, Any], device: torch.device):
    """仅在 FP16 下创建梯度缩放器；BF16 的数值范围通常无需缩放。"""

    dtype = amp_dtype(training_config, device)
    enabled = dtype == torch.float16
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # 兼容较旧版本的 PyTorch 接口
        return torch.cuda.amp.GradScaler(enabled=enabled)


def load_config(path: str | Path) -> dict:
    """读取 UTF-8 YAML 配置。"""

    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping")
    return config


def set_seed(seed: int) -> None:
    """固定 Python、NumPy 与 PyTorch 随机性，便于复现实验。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(value: str) -> torch.device:
    """auto 优先选择 CUDA；显式指定不可用 CUDA 时立即报错。"""

    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "当前 PyTorch 不提供可用的 CUDA："
            f"torch={torch.__version__}, torch.version.cuda={torch.version.cuda!r}。"
            "请在当前环境安装 CUDA 版 PyTorch。"
        )
    return device
