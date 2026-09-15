"""Validation-free cross-client transfer utility diagnostic."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from config import configure_torch_runtime, resolve_device, set_seed
from data import archive_sha256, make_data_loader
from experiment_runtime import graph_tensors, load_project_config, load_smartds, make_dataset, make_model, experiment_config
from federated import build_client_model, metric_summary, scale_aware_l1_loss, train_local
from models import local_parameter_prefixes, shared_state_dict


def _cosine(a, b):
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / den) if den else 0.0


def _select_indices(n, count, seed):
    if count >= n:
        return np.arange(n, dtype=np.int64)
    return np.linspace(0, n - 1, count, dtype=np.int64)


def _eval(model, dataset, graph, loader, device, cfg):
    model.eval(); preds, targets = [], []
    adjacency, edge_features = graph_tensors(graph, device)
    with torch.no_grad():
        for x, y in loader:
            out = model(x.to(device), adjacency, edge_features)["prediction"]
            preds.append(out.cpu()); targets.append(y)
    prediction, target = torch.cat(preds), torch.cat(targets)
    raw = metric_summary(prediction, target, dataset)
    raw["wl1"] = float(scale_aware_l1_loss(prediction, target, dataset.scale).cpu())
    return raw


def _spearman(x, y):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if len(x) < 2:
        return None
    def rank(values):
        order = np.argsort(values, kind="mergesort"); out = np.empty(len(values), dtype=np.float64)
        sorted_values = values[order]; start = 0
        while start < len(values):
            end = start + 1
            while end < len(values) and sorted_values[end] == sorted_values[start]: end += 1
            out[order[start:end]] = 0.5 * (start + end - 1) + 1.0; start = end
        return out
    xr, yr = rank(x), rank(y); xr -= xr.mean(); yr -= yr.mean()
    den = np.linalg.norm(xr) * np.linalg.norm(yr)
    return None if den == 0 else float(np.dot(xr, yr) / den)


def run(seed=2026, device_name="auto"):
    set_seed(seed); base = load_project_config()
    cfg, _ = experiment_config(base, "stattn_moduleala_wl1_screen", seed)
    cfg["seed"] = seed; cfg["device"] = device_name
    cfg["federated"]["algorithm"] = "FedAvg"; cfg["federated"]["rounds"] = 1; cfg["federated"]["local_epochs"] = 1; cfg["federated"]["mu"] = 0.0
    cfg["training"]["max_train_windows"] = None; cfg["training"]["evaluate_test"] = False; cfg["training"]["federated_batch_size"] = 512
    configure_torch_runtime(cfg.get("runtime", {})); device = resolve_device(device_name)
    data = load_smartds(cfg); bounds = data.split_bounds(cfg["data"]["train_ratio"], cfg["data"]["val_ratio"])
    parts = data.client_partitions(int(cfg["federated"]["clients"]))
    template = make_model(cfg, len(parts[0]), device)
    initial = {n: v.detach().cpu().clone() for n, v in template.named_parameters()}
    global_initial = shared_state_dict(template)
    datasets = [make_dataset(data, nodes, "train", cfg) for nodes in parts]
    graphs = [data.graph_view(nodes, cfg["data"]["graph"], int(cfg["data"].get("hop_radius", 2)), int(cfg["data"].get("target_knn_k", 6))) for nodes in parts]
    models, states, calib_sets, base_sets, train_metrics = [], [], [], [], []
    calibration_count = 512
    for idx, (nodes, ds, graph) in enumerate(zip(parts, datasets, graphs)):
        cal_idx = _select_indices(len(ds), min(calibration_count, len(ds)), seed + idx)
        cal_set = torch.utils.data.Subset(ds, cal_idx.tolist())
        mask = np.ones(len(ds), dtype=bool); mask[cal_idx] = False
        remaining = np.flatnonzero(mask)
        train_idx = remaining[_select_indices(len(remaining), min(2048, len(remaining)), seed + 17 * idx)]
        train_set = torch.utils.data.Subset(ds, train_idx.tolist())
        model = build_client_model(template, len(nodes)).to(device)
        current = dict(model.named_parameters())
        for name, value in initial.items():
            if name in current and current[name].shape == value.shape: current[name].data.copy_(value.to(device))
        loader = make_data_loader(train_set, cfg["training"]["federated_batch_size"], True, cfg["training"])
        state, metrics = train_local(model, ds, graph, cfg, global_initial, loader=loader, graph_tensors_device=graph_tensors(graph, device))
        models.append(model); states.append({n: v.detach().cpu().clone() for n, v in model.state_dict().items()}); calib_sets.append(cal_set); base_sets.append(ds); train_metrics.append(metrics)
    utility = {"0.25": np.zeros((8, 8)), "0.50": np.zeros((8, 8))}; utility_wl1 = {a: np.zeros((8, 8)) for a in utility}; baselines = []
    calib_loaders = [make_data_loader(s, 1024, False, cfg["training"]) for s in calib_sets]
    for i in range(8):
        base_metric = _eval(models[i], base_sets[i], graphs[i], calib_loaders[i], device, cfg)
        baselines.append(base_metric)
        recipient = {n: v.detach().cpu().clone() for n, v in models[i].state_dict().items()}
        for j in range(8):
            for alpha in (0.25, 0.50):
                mixed = {n: v.clone() for n, v in recipient.items()}
                donor = states[j]
                for name in mixed:
                    if name.startswith(local_parameter_prefixes(False)): continue
                    if name in donor and donor[name].shape == mixed[name].shape:
                        mixed[name] = ((1-alpha)*recipient[name] + alpha*donor[name]).to(mixed[name].dtype).to(device)
                models[i].load_state_dict(mixed, strict=True)
                score = _eval(models[i], base_sets[i], graphs[i], calib_loaders[i], device, cfg)
                alpha_key = f"{alpha:.2f}"
                utility[alpha_key][i,j] = baselines[i]["wape"] - score["wape"]
                utility_wl1[alpha_key][i,j] = baselines[i]["wl1"] - score["wl1"]
        models[i].load_state_dict(recipient, strict=True)
    # Existing heterogeneity artifact supplies update-cosine and topology similarity.
    hetero_path = Path(__file__).resolve().parents[1] / "results" / f"client_heterogeneity_diagnostic_seed{seed}.json"
    hetero = json.loads(hetero_path.read_text(encoding="utf-8")) if hetero_path.exists() else {}
    update_proxy = hetero.get("update_cosine", {})
    topology_proxy = hetero.get("spatial_relation", {}).get("normalized_similarity")
    mean_update = np.mean(np.asarray([update_proxy[m] for m in ("physical", "temporal", "horizon_decoder", "head")], dtype=np.float64), axis=0) if update_proxy else None
    out = {"analysis":"client_transfer_utility_diagnostic","seed":seed,"device":str(device),"test_evaluated":False,"data_source_sha256":archive_sha256(data.source),"train_only":True,"calibration_windows_per_client":512,"training_windows_per_client":2048,"calibration_training_index_overlap":0,"validation_accessed":False,"test_accessed":False,"split_bounds":{"train_end":int(bounds.train_end),"val_end":int(bounds.val_end)},"self_baseline_wape":[m["wape"] for m in baselines],"self_baseline_wl1":[m["wl1"] for m in baselines],"utility_wape":{a: utility[a].tolist() for a in utility},"utility_wl1":{a: utility_wl1[a].tolist() for a in utility},"positive_donor_count":{a:[int(np.count_nonzero(utility[a][i, np.arange(8)!=i]>0)) for i in range(8)] for a in utility},"positive_donor_count_wl1":{a:[int(np.count_nonzero(utility_wl1[a][i, np.arange(8)!=i]>0)) for i in range(8)] for a in utility},"off_diagonal_positive_pairs":{a:int(np.count_nonzero(utility[a][~np.eye(8,dtype=bool)]>0)) for a in utility},"off_diagonal_positive_pairs_wl1":{a:int(np.count_nonzero(utility_wl1[a][~np.eye(8,dtype=bool)]>0)) for a in utility},"proxy_relations":{"update_cosine":update_proxy,"mean_update_cosine":mean_update.tolist() if mean_update is not None else None,"topology_similarity":topology_proxy},"local_training":train_metrics}
    for alpha, mat in utility.items():
        off = ~np.eye(8,dtype=bool); gains = mat[off]; wl1_gains = utility_wl1[alpha][off]
        best = [int(np.argmax(np.where(np.arange(8)==i, -np.inf, mat[i]))) for i in range(8)]
        best_gain = [float(mat[i,best[i]]) for i in range(8)]
        best_wl1 = [int(np.argmax(np.where(np.arange(8)==i, -np.inf, utility_wl1[alpha][i]))) for i in range(8)]
        out["summary_"+alpha]={"wape":{"mean":float(gains.mean()),"median":float(np.median(gains)),"min":float(gains.min()),"max":float(gains.max())},"wl1":{"mean":float(wl1_gains.mean()),"median":float(np.median(wl1_gains)),"min":float(wl1_gains.min()),"max":float(wl1_gains.max())},"best_donor_id":best,"best_gain_wape":best_gain,"best_donor_id_wl1":best_wl1,"pair_count":56,"spearman":{"utility_wape_vs_mean_update_cosine":_spearman(mat[off],mean_update[off]) if mean_update is not None else None,"utility_wape_vs_topology_similarity":_spearman(mat[off],np.asarray(topology_proxy)[off]) if topology_proxy is not None else None,"pair_count":56}}
    path = Path(__file__).resolve().parents[1] / "results" / f"client_transfer_utility_diagnostic_seed{seed}.json"; path.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8"); print(json.dumps({"output":str(path),"test_evaluated":False},ensure_ascii=False)); return out


if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--seed",type=int,default=2026); p.add_argument("--device",default="auto"); a=p.parse_args(); run(a.seed,a.device)
