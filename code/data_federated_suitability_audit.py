"""Train-only suitability audit for federated load forecasting."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from data import archive_sha256
from experiment_runtime import _hash_partitions, load_project_config, load_smartds


def _summary(values: np.ndarray) -> dict[str, float]:
    return {"mean": float(values.mean()), "median": float(np.median(values)), "p10": float(np.percentile(values, 10)), "p90": float(np.percentile(values, 90))}


def _hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _autocorr(x: np.ndarray, lag: int) -> np.ndarray:
    if lag >= len(x): return np.full(x.shape[1], np.nan)
    left, right = x[:-lag], x[lag:]
    left = left - left.mean(axis=0); right = right - right.mean(axis=0)
    denom = np.sqrt((left * left).sum(axis=0) * (right * right).sum(axis=0))
    return np.divide((left * right).sum(axis=0), denom, out=np.zeros_like(denom), where=denom > 0)


def _wape(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(100.0 * np.abs(prediction - target).sum() / np.maximum(np.abs(target).sum(), 1e-12))


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


def _wasserstein(a: np.ndarray, b: np.ndarray) -> float:
    # Equal-size empirical distributions: mean absolute quantile separation.
    q = np.linspace(0.0, 1.0, 1024)
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


def _profiles(values: np.ndarray, period: int) -> np.ndarray:
    usable = values[: len(values) // period * period]
    profile = usable.reshape(-1, period, values.shape[1]).mean(axis=0).T
    scale = np.maximum(profile.mean(axis=1, keepdims=True), 1e-12)
    return profile / scale


def run() -> dict:
    cfg = load_project_config(); data = load_smartds(cfg)
    bounds = data.split_bounds(cfg["data"]["train_ratio"], cfg["data"]["val_ratio"])
    active = data.active_indices; train = data.load_ts[:bounds.train_end, active].astype(np.float64)
    nodes, clients = len(active), data.client_partitions(8)
    median = np.median(train, axis=0); q25, q75 = np.percentile(train, [25, 75], axis=0); iqr = np.maximum(q75-q25, 1e-12)
    mean, std = train.mean(axis=0), train.std(axis=0); cv = std / np.maximum(np.abs(mean), 1e-12)
    corr = np.corrcoef(train, rowvar=False); off = np.abs(corr[~np.eye(nodes, dtype=bool)])
    _, singular, _ = np.linalg.svd(train - train.mean(axis=0), full_matrices=False); explained = singular**2 / np.maximum((singular**2).sum(), 1e-12)
    daily_nodes, weekly_nodes = _profiles(train, 96), _profiles(train, 672)
    partition_local = [np.array([np.flatnonzero(active == node)[0] for node in part]) for part in clients]
    client_stats, daily_profiles, weekly_profiles, distributions = [], [], [], []
    for client_id, idx in enumerate(partition_local):
        block = train[:, idx]; scale = float(block.mean()); distributions.append(block.reshape(-1))
        daily = _profiles(block, 96).mean(axis=0); weekly = _profiles(block, 672).mean(axis=0)
        daily_profiles.append(daily); weekly_profiles.append(weekly)
        client_stats.append({"client_id": client_id, "node_count": int(len(idx)), "mean": scale, "std": float(block.std()), "cv": float(block.std()/max(abs(scale),1e-12)), "average_load_scale": scale, "daily_profile": daily.tolist(), "weekly_profile": weekly.tolist()})
    daily_cos, weekly_cos, wasserstein = np.eye(8), np.eye(8), np.zeros((8,8))
    for i in range(8):
        for j in range(8):
            if i != j:
                daily_cos[i,j] = _cosine(daily_profiles[i], daily_profiles[j]); weekly_cos[i,j] = _cosine(weekly_profiles[i], weekly_profiles[j]); wasserstein[i,j] = _wasserstein(distributions[i], distributions[j])
    def profile_pairs(profiles, same: bool) -> float:
        values=[]
        for i in range(nodes):
            for j in range(i+1,nodes):
                same_client=any(i in idx and j in idx for idx in partition_local)
                if same_client == same: values.append(_cosine(profiles[i],profiles[j]))
        return float(np.mean(values))
    client_scales=np.asarray([item["average_load_scale"] for item in client_stats]); pair_mask=~np.eye(8,dtype=bool)
    report={
        "analysis":"data_federated_suitability_audit_trainonly", "train_only":True, "validation_accessed":False, "test_accessed":False,
        "data_source_sha256":archive_sha256(data.source), "node_indices_sha256":_hash(active.astype(np.int64)), "client_partition_sha256":_hash_partitions(clients),
        "split_bounds":{"train_end":int(bounds.train_end),"val_end":int(bounds.val_end),"total":int(bounds.total)}, "target_node_count":nodes, "client_count":8,
        "basic_quality":{"per_node":{"min":train.min(axis=0).tolist(),"max":train.max(axis=0).tolist(),"mean":mean.tolist(),"std":std.tolist(),"cv":cv.tolist(),"zero_ratio":(train==0).mean(axis=0).tolist()}, "constant_node_count":int(np.count_nonzero(std==0)), "near_constant_node_count":int(np.count_nonzero(std<=1e-8)), "nan_inf_count":int(np.count_nonzero(~np.isfinite(train))), "negative_value_count":int(np.count_nonzero(train<0)), "extreme_value_ratio":float(np.mean((train < median-8*iqr)|(train > median+8*iqr)))} ,
        "temporal_predictability":{"lag1":_summary(_autocorr(train,1)),"lag96":_summary(_autocorr(train,96)),"lag672":_summary(_autocorr(train,672)),"persistence_raw_wape":_wape(train[:-1],train[1:]),"daily_persistence_raw_wape":_wape(train[:-96],train[96:])},
        "node_redundancy":{"pearson_correlation":corr.tolist(),"off_diagonal_abs_correlation":_summary(off),"pair_ratio_abs_gt_08":float(np.mean(off>.8)),"pair_ratio_abs_gt_09":float(np.mean(off>.9)),"pair_ratio_abs_gt_095":float(np.mean(off>.95)),"pca_explained_variance":{"1":float(explained[:1].sum()),"5":float(explained[:5].sum()),"10":float(explained[:10].sum()),"20":float(explained[:20].sum())}},
        "client_heterogeneity":{"clients":client_stats,"daily_profile_cosine":daily_cos.tolist(),"weekly_profile_cosine":weekly_cos.tolist(),"load_distribution_wasserstein":wasserstein.tolist(),"daily_cosine_pairs":_summary(daily_cos[pair_mask]),"weekly_cosine_pairs":_summary(weekly_cos[pair_mask]),"wasserstein_pairs":_summary(wasserstein[pair_mask])},
        "within_between_profile_similarity":{"daily_within":profile_pairs(daily_nodes,True),"daily_between":profile_pairs(daily_nodes,False),"daily_between_minus_within":profile_pairs(daily_nodes,False)-profile_pairs(daily_nodes,True),"weekly_within":profile_pairs(weekly_nodes,True),"weekly_between":profile_pairs(weekly_nodes,False),"weekly_between_minus_within":profile_pairs(weekly_nodes,False)-profile_pairs(weekly_nodes,True)},
        "scale_vs_shape":{"scale_cv_across_clients":float(client_scales.std()/max(abs(client_scales.mean()),1e-12)),"daily_shape_distance_mean":float(np.mean(1-daily_cos[pair_mask])),"weekly_shape_distance_mean":float(np.mean(1-weekly_cos[pair_mask]))},
    }
    path=Path(__file__).resolve().parents[1]/"results"/"data_federated_suitability_audit_trainonly.json"; path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"); print(json.dumps({"output":str(path),"targets":nodes,"train_only":True},ensure_ascii=False)); return report


if __name__ == "__main__":
    argparse.ArgumentParser(description="Train-only federated suitability audit").parse_args(); run()
