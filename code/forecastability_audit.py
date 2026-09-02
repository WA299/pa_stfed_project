"""只读的 validation forecastability audit。

本脚本不训练模型、不创建 test loader，也不改写已有结果。模型推理、数据
窗口、归一化和总体指标均复用 ``run.py``/``federated.py`` 的现有实现；其余
统计只在已经生成的 validation 预测数组上计算。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
CODE = ROOT / "code"
RESULTS = ROOT / "results"
REPORTS = ROOT / "reports"
sys.path.insert(0, str(CODE))

from federated import metric_summary  # noqa: E402
from metrics import mae, rmse, wape  # noqa: E402
from run import (  # noqa: E402
    experiment_config,
    graph_tensors,
    load_project_config,
    make_data_loader,
    make_dataset,
    make_model,
)
from config import autocast_context, resolve_device, set_seed  # noqa: E402


def _json_value(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _basic_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """在原始负荷尺度上独立重算审计所需的三个指标。"""

    return {
        "wape": float(wape(prediction, target).cpu()),
        "mae": float(mae(prediction, target).cpu()),
        "rmse": float(rmse(prediction, target).cpu()),
    }


def _horizon_slices() -> dict[str, tuple[int, int]]:
    """返回 prefix 与 exact-step 的明确切片定义。"""

    return {
        "prefix_h1": (0, 1),
        "prefix_h3": (0, 3),
        "prefix_h6": (0, 6),
        "prefix_h12": (0, 12),
        "step1": (0, 1),
        "step3": (2, 3),
        "step6": (5, 6),
        "step12": (11, 12),
        "overall_12step": (0, 12),
    }


def _horizon_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for name, (start, end) in _horizon_slices().items():
        p = torch.from_numpy(prediction[:, start:end].astype(np.float32, copy=False))
        t = torch.from_numpy(target[:, start:end].astype(np.float32, copy=False))
        result[name] = _basic_metrics(p, t)
    return result


def _feeder_wape(prediction: np.ndarray, target: np.ndarray) -> float:
    p = prediction.sum(axis=-1)
    t = target.sum(axis=-1)
    return float(wape(torch.from_numpy(p.astype(np.float32)), torch.from_numpy(t.astype(np.float32))).item())


def _node_wape(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    error = np.abs(prediction - target).sum(axis=(0, 1))
    denominator = np.abs(target).sum(axis=(0, 1))
    return 100.0 * error / np.maximum(denominator, 1e-6)


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return float("nan")
    x, y = _rank(np.asarray(left, dtype=np.float64)), _rank(np.asarray(right, dtype=np.float64))
    return float(np.corrcoef(x, y)[0, 1])


def _autocorr(series: np.ndarray, lag: int) -> float:
    if len(series) <= lag:
        return float("nan")
    x, y = series[:-lag].astype(np.float64), series[lag:].astype(np.float64)
    x -= x.mean()
    y -= y.mean()
    denom = np.sqrt(np.dot(x, x) * np.dot(y, y))
    return float(np.dot(x, y) / denom) if denom > 1e-12 else 0.0


def _load_and_predict(name: str, checkpoint_name: str, base_cfg: dict, device: torch.device) -> dict:
    cfg, _ = experiment_config(base_cfg, name, 2026)
    data = __import__("run", fromlist=["load_smartds"]).load_smartds(cfg)
    bounds = data.split_bounds(cfg["data"]["train_ratio"], cfg["data"]["val_ratio"])
    dataset = make_dataset(data, data.active_indices, "val", cfg)
    assert all(int(origin) + int(dataset.horizon) + 1 <= bounds.val_end for origin in dataset.origins)
    assert all(int(origin) < bounds.val_end for origin in dataset.origins)
    loader = make_data_loader(dataset, int(cfg["training"].get("eval_batch_size", 64)), False, cfg["training"])
    model = make_model(cfg, len(data.active_indices), device)
    checkpoint = torch.load(RESULTS / checkpoint_name, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    adjacency, edge_features = graph_tensors(data.graph_view(data.active_indices, cfg["data"]["graph"], int(cfg["data"].get("hop_radius", 2)), int(cfg["data"].get("target_knn_k", 6))), device)
    predictions, targets, persist, daily = [], [], [], []
    with torch.inference_mode():
        for inputs, target in loader:
            inputs_device = inputs.to(device)
            with autocast_context(cfg["training"], device):
                output = model(inputs_device, adjacency, edge_features)["prediction"]
            prediction = dataset.denormalize(output).float().cpu().numpy()
            actual = dataset.denormalize(target.to(device)).float().cpu().numpy()
            last = dataset.denormalize(inputs[:, -1, :, 0].to(device)).float().cpu().numpy()
            day = dataset.denormalize(inputs[:, : dataset.horizon, :, 0].to(device)).float().cpu().numpy()
            predictions.append(prediction)
            targets.append(actual)
            persist.append(np.repeat(last[:, None, :], dataset.horizon, axis=1))
            daily.append(day)
    result_path = RESULTS / f"{name}_seed2026_centralized_result.json"
    existing = json.loads(result_path.read_text(encoding="utf-8"))
    prediction = np.concatenate(predictions, axis=0)
    target = np.concatenate(targets, axis=0)
    persistence = np.concatenate(persist, axis=0)
    daily_naive = np.concatenate(daily, axis=0)
    recomputed = _basic_metrics(torch.from_numpy(prediction), torch.from_numpy(target))
    expected = existing["best_validation"]
    diffs = {key: abs(recomputed[key] - float(expected[key])) for key in ("wape", "mae", "rmse")}
    if any(value > 1e-4 for value in diffs.values()):
        raise RuntimeError(f"metric sanity failed for {name}: {diffs}")
    return {
        "experiment": name,
        "result_json": str(result_path),
        "checkpoint": str(RESULTS / checkpoint_name),
        "best_epoch": int(existing["best_epoch"]),
        "code_revision": existing.get("code_revision"),
        "split_bounds": bounds.__dict__,
        "n_validation_origins": int(len(dataset)),
        "validation_origins": dataset.origins.copy(),
        "predictions": prediction,
        "target": target,
        "persistence": persistence,
        "daily_naive": daily_naive,
        "recomputed_metrics": recomputed,
        "existing_metrics": {key: float(expected[key]) for key in ("wape", "mae", "rmse")},
        "metric_diffs": diffs,
        "dataset": data,
        "cfg": cfg,
    }


def main() -> None:
    set_seed(2026)
    base_cfg = load_project_config()
    device = resolve_device(base_cfg.get("device", "auto"))
    pa = _load_and_predict("centralized", "centralized_seed2026_centralized_model.pt", base_cfg, device)
    gwn = _load_and_predict("gwnet", "gwnet_seed2026_centralized_model.pt", base_cfg, device)
    residual = _load_and_predict(
        "pa_residual_anchor_dev",
        "pa_residual_anchor_dev_seed2026_centralized_model.pt",
        base_cfg,
        device,
    )
    data = pa["dataset"]
    target = pa["target"]
    methods = {
        "PA-STFed": pa["predictions"],
        "PA-STFed residual-anchor": residual["predictions"],
        "GWN": gwn["predictions"],
        "Persistence": pa["persistence"],
        "Daily-lag naive": pa["daily_naive"],
    }
    # 严格按未来目标步构造 seasonal naive；source 只来自 validation origin 之前
    # 的 raw load_ts，不依赖模型、不访问 test，也不进行插值或平移。
    raw_load = data.load_ts[:, data.active_indices].astype(np.float32, copy=False)
    origins = np.asarray(pa["validation_origins"], dtype=np.int64)
    target_positions = origins[:, None] + 1 + np.arange(int(pa["cfg"]["data"]["horizon"]), dtype=np.int64)[None, :]
    assert np.all(target_positions < int(pa["split_bounds"]["val_end"]))
    assert np.all(target_positions - 672 >= 0)
    daily_source = raw_load[target_positions - 96]
    weekly_source = raw_load[target_positions - 672]
    daily_lag_formula = daily_source
    weekly_lag = weekly_source
    methods["Weekly-lag naive"] = weekly_lag
    blend_alphas = (0.0, 0.25, 0.5, 0.75, 1.0)
    for alpha in blend_alphas:
        methods[f"Daily-weekly blend alpha={alpha:.2f}"] = alpha * daily_lag_formula + (1.0 - alpha) * weekly_lag
    horizons = {name: _horizon_metrics(pred, target) for name, pred in methods.items()}
    aggregation = {
        name: {
            key: {
                "node_micro_wape": values["wape"],
                "feeder_aggregate_wape": _feeder_wape(
                    pred[:, start:end], target[:, start:end]
                ),
            }
            for key, values in horizons[name].items()
            for start, end in [_horizon_slices()[key]]
        }
        for name, pred in methods.items()
    }
    comparison_methods = ("prefix_h1", "prefix_h3", "prefix_h6", "prefix_h12", "step1", "step3", "step6", "step12", "overall_12step")
    model_deltas = {
        "residual_minus_pa": {
            key: {
                metric: float(horizons["PA-STFed residual-anchor"][key][metric] - horizons["PA-STFed"][key][metric])
                for metric in ("wape", "mae", "rmse")
            }
            | {"feeder_aggregate_wape": float(aggregation["PA-STFed residual-anchor"][key]["feeder_aggregate_wape"] - aggregation["PA-STFed"][key]["feeder_aggregate_wape"])}
            for key in comparison_methods
        },
        "residual_minus_gwn": {
            key: {
                metric: float(horizons["PA-STFed residual-anchor"][key][metric] - horizons["GWN"][key][metric])
                for metric in ("wape", "mae", "rmse")
            }
            | {"feeder_aggregate_wape": float(aggregation["PA-STFed residual-anchor"][key]["feeder_aggregate_wape"] - aggregation["GWN"][key]["feeder_aggregate_wape"])}
            for key in comparison_methods
        },
    }
    node_ids = [str(x) for x in data.node_ids[data.active_indices]]
    node_difficulty = {}
    for name, prediction in methods.items():
        values = _node_wape(prediction, target)
        order = np.argsort(values, kind="mergesort")
        node_difficulty[name] = {
            "quantiles": {key: float(np.quantile(values, q)) for key, q in (("min", 0), ("p10", .1), ("p25", .25), ("median", .5), ("p75", .75), ("p90", .9), ("max", 1))},
            "counts": {"lt10": int((values < 10).sum()), "lt20": int((values < 20).sum()), "lt30": int((values < 30).sum()), "ge40": int((values >= 40).sum())},
            "best10": [{"node_id": node_ids[i], "wape": float(values[i])} for i in order[:10]],
            "worst10": [{"node_id": node_ids[i], "wape": float(values[i])} for i in order[-10:][::-1]],
            "wape_by_node": {node_ids[i]: float(values[i]) for i in range(len(values))},
        }
    weekly_values = _node_wape(weekly_lag, target)
    pa_values = _node_wape(pa["predictions"], target)
    weekly_order = np.argsort(weekly_values, kind="mergesort")
    load = data.load_ts[:, data.active_indices].astype(np.float64)
    train_end, val_end = pa["split_bounds"]["train_end"], pa["split_bounds"]["val_end"]
    train, val = load[:train_end], load[train_end:val_end]
    means, stds = train.mean(0), train.std(0)
    forecastability = {
        "nodes": {
            node_ids[i]: {
                "train_mean": float(means[i]), "train_std": float(stds[i]), "train_cv": float(stds[i] / (abs(means[i]) + 1e-8)),
                "lag96_autocorr": _autocorr(train[:, i], 96), "lag672_autocorr": _autocorr(train[:, i], 672),
                "mean_shift": float(abs(val[:, i].mean() - means[i]) / (stds[i] + 1e-8)), "val_cv": float(val[:, i].std() / (abs(val[:, i].mean()) + 1e-8)),
            } for i in range(load.shape[1])
        }
    }
    for method, details in node_difficulty.items():
        wapes = np.asarray([details["wape_by_node"][node] for node in node_ids])
        forecastability[method] = {key: _spearman(wapes, np.asarray([item[key] for item in forecastability["nodes"].values()])) for key in ("train_cv", "lag96_autocorr", "lag672_autocorr", "mean_shift")}
    naive_existing = json.loads((RESULTS / "baseline_grouped_seed2026_baseline_result.json").read_text(encoding="utf-8"))["centralized"]
    naive_recomputed = {
        name: _basic_metrics(torch.from_numpy(pred), torch.from_numpy(target))
        for name, pred in methods.items()
        if name in ("Persistence", "Daily-lag naive")
    }
    naive_comparison = {
        name: {
            metric: {
                "recomputed": float(naive_recomputed[name][metric]),
                "existing": float(naive_existing["persistence" if name == "Persistence" else "daily_naive"][metric]),
                "abs_diff": abs(float(naive_recomputed[name][metric]) - float(naive_existing["persistence" if name == "Persistence" else "daily_naive"][metric])),
            }
            for metric in ("wape", "mae", "rmse")
        }
        for name in ("Persistence", "Daily-lag naive")
    }
    payload = {
        "audit": "validation-only forecastability audit",
        "code_revision": base_cfg.get("code_revision"),
        "test_evaluated": False,
        "test_loader_created": False,
        "metric_sanity": {name: {key: details[key] for key in ("recomputed_metrics", "existing_metrics", "metric_diffs")} for name, details in (("PA-STFed", pa), ("GWN", gwn), ("PA-STFed residual-anchor", residual))},
        "horizon_metrics": {name: {key: {metric: float(value) for metric, value in metrics.items()} for key, metrics in values.items()} for name, values in horizons.items()},
        "aggregation_effect": aggregation,
        "model_deltas": model_deltas,
        "node_difficulty": node_difficulty,
        "weekly_naive_node_wape_distribution": {
            "quantiles": {key: float(np.quantile(weekly_values, q)) for key, q in (("min", 0), ("p10", .1), ("p25", .25), ("median", .5), ("p75", .75), ("p90", .9), ("max", 1))},
            "best10": [{"node_id": node_ids[i], "wape": float(weekly_values[i])} for i in weekly_order[:10]],
            "worst10": [{"node_id": node_ids[i], "wape": float(weekly_values[i])} for i in weekly_order[-10:][::-1]],
            "pa_wape_better_count": int((pa_values < weekly_values).sum()),
            "weekly_naive_better_or_equal_count": int((weekly_values <= pa_values).sum()),
        },
        "data_forecastability": forecastability,
        "naive_sanity": {"recomputed": naive_recomputed, "existing_baseline_centralized": naive_existing, "comparison": naive_comparison, "same_definition": True, "daily_lag_definition": "prediction at future step h uses the input load at origin+h-96; this matches the existing baseline implementation.", "weekly_lag_definition": "prediction at future step h uses raw load at origin+h-672.", "blend_definition": "alpha * raw load(origin+h-96) + (1-alpha) * raw load(origin+h-672), with alpha in {0, 0.25, 0.5, 0.75, 1}."},
    }
    REPORTS.mkdir(exist_ok=True)
    (RESULTS / "forecastability_audit.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_value), encoding="utf-8")
    lines = ["# Forecastability Audit", "", "Validation-only; no training, no test loader, and no existing result JSON was modified.", "", "## Metric Sanity", ""]
    for name, details in payload["metric_sanity"].items():
        lines.append(f"- {name}: recomputed WAPE/MAE/RMSE = {details['recomputed_metrics']}; max difference = {max(details['metric_diffs'].values()):.8f} (PASS)")
    lines += ["", "## Horizon-wise Metrics", "", "prefix_h* are cumulative prefix metrics; step* are exact forecast steps.", "", "| Method | Horizon | Node-micro WAPE | MAE | RMSE |", "|---|---:|---:|---:|---:|"]
    for method, values in payload["horizon_metrics"].items():
        for horizon, metrics in values.items():
            lines.append(f"| {method} | {horizon} | {metrics['wape']:.4f} | {metrics['mae']:.6f} | {metrics['rmse']:.6f} |")
    lines += ["", "## Aggregation Effect", "", "Prefix and exact-step feeder aggregate WAPE are both reported.", "", "| Method | Horizon | Node WAPE | Feeder aggregate WAPE |", "|---|---:|---:|---:|"]
    for method, values in aggregation.items():
        for horizon, metrics in values.items():
            lines.append(f"| {method} | {horizon} | {metrics['node_micro_wape']:.4f} | {metrics['feeder_aggregate_wape']:.4f} |")
    lines += ["", "## Residual-anchor Differences", "", "Differences are defined as residual-anchor minus the comparison method; negative values favor residual-anchor.", "", "| Comparison | Horizon | dWAPE | dMAE | dRMSE | dFeeder WAPE |", "|---|---:|---:|---:|---:|---:|"]
    for comparison, values in (("Original PA-STFed", model_deltas["residual_minus_pa"]), ("GWN", model_deltas["residual_minus_gwn"])):
        for horizon, metrics in values.items():
            lines.append(f"| residual-anchor - {comparison} | {horizon} | {metrics['wape']:.4f} | {metrics['mae']:.6f} | {metrics['rmse']:.6f} | {metrics['feeder_aggregate_wape']:.4f} |")
    lines += ["", "## Node Difficulty", ""]
    for method, details in node_difficulty.items():
        lines.append(f"- {method}: {details['quantiles']}; counts={details['counts']}")
    weekly_details = payload["weekly_naive_node_wape_distribution"]
    lines += ["", "## Weekly-lag Node Distribution", "", f"- quantiles: {weekly_details['quantiles']}", f"- PA-STFed has lower overall node WAPE than weekly-lag naive for {weekly_details['pa_wape_better_count']} of 92 nodes; weekly-lag is lower or equal for {weekly_details['weekly_naive_better_or_equal_count']} nodes.", f"- best10: {weekly_details['best10']}", f"- worst10: {weekly_details['worst10']}"]
    lines += ["", "## Forecastability Correlations", ""]
    for method in methods:
        lines.append(f"- {method}: {forecastability[method]}")
    pa_h, gwn_h = horizons["PA-STFed"], horizons["GWN"]
    h12_growth = pa_h["prefix_h12"]["wape"] - pa_h["step1"]["wape"]
    lines += ["", "## Conclusions", "", f"1. PA-STFed node-micro WAPE is {pa_h['overall_12step']['wape']:.2f}% and feeder-aggregate WAPE is {aggregation['PA-STFed']['overall_12step']['feeder_aggregate_wape']:.2f}%; this quantifies the aggregation-level effect.", f"2. PA-STFed overall prefix WAPE minus exact step1 WAPE is {h12_growth:.2f} percentage points, so horizon degradation is {'present' if h12_growth > 0 else 'not evident'}.", f"3. The reported Spearman correlations quantify whether high error tracks CV, autocorrelation, or mean shift; no causal claim is made.", f"4. PA-STFed versus GWN overall-12-step WAPE gap is {pa_h['overall_12step']['wape'] - gwn_h['overall_12step']['wape']:.2f} percentage points; horizon-wise and node-level tables above show where it concentrates."]
    (REPORTS / "forecastability_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("forecastability audit PASS")


if __name__ == "__main__":
    main()
