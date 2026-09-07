"""Build the Innovation 1 ledger from existing result and audit artifacts.

This is a read-only reporting utility. It does not train, load checkpoints, or
create test loaders. Numeric model metrics come from best_validation in the
existing result JSON files; horizon/feeder diagnostics come from the existing
forecastability audit JSON.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
REPORTS = ROOT / "reports"
SUPPORTING = REPORTS / "supporting"


def result_for(experiment: str, seed: int = 2026) -> tuple[dict, str]:
    files = sorted(RESULTS.glob(f"{experiment}_seed{seed}_*_result.json"))
    if not files:
        raise FileNotFoundError(f"No result JSON for {experiment} seed {seed}")
    with files[0].open(encoding="utf-8") as handle:
        return json.load(handle), files[0].name


def metrics_for(experiment: str) -> tuple[dict, str, dict]:
    result, filename = result_for(experiment)
    return result["best_validation"], filename, result


def fmt(value: object) -> str:
    return "" if value in (None, "") else f"{float(value):.6f}"


def audit_metric(audit: dict, label: str, horizon: str = "overall_12step") -> dict:
    return audit["horizon_metrics"][label][horizon]


def feeder_metric(audit: dict, label: str, horizon: str = "overall_12step") -> float:
    return audit["aggregation_effect"][label][horizon]["feeder_aggregate_wape"]


def main() -> None:
    audit = json.loads((SUPPORTING / "forecastability_audit.json").read_text(encoding="utf-8"))
    yaml_text = (ROOT / "code" / "experiments.yaml").read_text(encoding="utf-8")

    labels = {
        "centralized": "PA-STFed",
        "gwnet": "GWN",
        "pa_residual_anchor_dev": "PA-STFed residual-anchor",
        "pa_residual_scale_loss_dev": "PA-STFed residual-scale-loss",
        "pa_residual_multilevel_loss_dev": "PA-STFed residual-multilevel-loss",
        "pa_residual_multilevel_l002_dev": "PA-STFed residual-multilevel-lambda0.02",
        "pa_residual_multilevel_l005_dev": "PA-STFed residual-multilevel-lambda0.05",
        "pa_horizon_decoder_scale_dev": "PA-STFed horizon-decoder",
        "pa_horizon_decoder_wl1_dev": "PA-STFed horizon-decoder-wl1",
        "pa_horizon_specific_head_scale_dev": "PA-STFed horizon-specific-head",
    }

    spatial_ablation = [
        ("spatial_physical_only", "KEEP", "spatial ablation", "", "Physical topology branch only; target relation graph is not a physical Line graph.", "Physical-only", "Parallel spatial ablation; not an incremental predecessor."),
        ("spatial_functional_only", "KEEP", "spatial ablation", "", "Functional relation branch only; no physical branch.", "Functional-only", "Parallel spatial ablation; not an incremental predecessor."),
        ("centralized", "KEEP", "spatial ablation base", "", "Physical + static functional PA-STFed dual base.", "Dual physical+functional", "Parallel dual-graph reference and base for the incremental chain."),
    ]
    main_chain = [
        ("pa_residual_anchor_dev", "KEEP", "incremental", "centralized", "Adds persistence residual anchor: last observed load plus learned correction.", "Residual Anchor", "Residual anchor lowers node WAPE in this development screen."),
        ("pa_residual_scale_loss_dev", "KEEP", "incremental", "pa_residual_anchor_dev", "Uses training-IQR scale-aware node loss.", "Scale-aware loss", "Scale-aware loss gives a small node-level change and remains a controlled objective variant."),
        ("pa_horizon_decoder_scale_dev", "KEEP", "incremental", "pa_residual_scale_loss_dev", "Adds one-layer horizon cross-attention decoder with zero-initialized correction head.", "Horizon Decoder", "Horizon decoder improves node WAPE over residual-scale in this screen."),
        ("pa_horizon_decoder_wl1_dev", "CURRENT BEST", "candidate", "pa_horizon_decoder_scale_dev", "Uses WAPE-aligned scale-aware L1 objective.", "WAPE-aligned WL1", "Lowest node-level WAPE among current development candidates; not final evidence."),
    ]

    required_dev = [
        "pa_residual_anchor_dev", "pa_residual_scale_loss_dev",
        "pa_residual_multilevel_loss_dev", "pa_residual_multilevel_l002_dev",
        "pa_residual_multilevel_l005_dev", "pa_multilevel_tcn_transformer_dev",
        "pa_dynamic_functional_scale_dev", "pa_multiscale_patch_scale_dev",
        "pa_horizon_decoder_scale_dev", "pa_horizon_timequery_scale_dev",
        "pa_horizon_specific_head_scale_dev", "pa_horizon_decoder_wl1_dev",
        "calendar_pa_screen", "calendar_gwnet_screen", "calendar_lstm_screen",
    ]
    for experiment in required_dev:
        if experiment not in yaml_text:
            raise AssertionError(f"{experiment} is absent from experiments.yaml")
        result_for(experiment)

    checks = []
    for experiment in required_dev:
        result, filename = result_for(experiment)
        passed = result.get("test_evaluated") is False and result.get("test") is None
        if not passed:
            raise AssertionError(f"test contract failed for {experiment}")
        checks.append(f"- `{experiment}`: `{filename}`, `test_evaluated=false`, `test=null` (PASS)")

    rows: list[dict] = []

    def add_row(experiment: str, status: str, role: str, reference: str,
                paper_use: str, notes: str, feeder: float | None = None) -> None:
        metrics, filename, _ = metrics_for(experiment)
        reference_wape = metrics_for(reference)[0]["wape"] if reference else None
        feeder_note = "" if feeder is not None else " Feeder WAPE unavailable: no matching aggregation entry in the existing audit artifact."
        rows.append({
            "experiment": experiment,
            "status": status,
            "role": role,
            "wape": metrics.get("wape"),
            "mae": metrics.get("mae"),
            "rmse": metrics.get("rmse"),
            "smape": metrics.get("smape"),
            "mape": metrics.get("mape"),
            "feeder_wape": feeder,
            "reference_experiment": reference,
            "delta_wape_pp": None if reference_wape is None else metrics["wape"] - reference_wape,
            "paper_use": paper_use,
            "notes": f"{notes}{feeder_note} Source={filename}; seed=2026; test_evaluated=false.",
        })

    for experiment, status, role, reference, change, paper_use, _ in spatial_ablation:
        add_row(experiment, status, role, reference, paper_use, change,
                feeder_metric(audit, labels[experiment]) if experiment in labels else None)
    for experiment, status, role, reference, change, paper_use, _ in main_chain:
        add_row(experiment, status, role, reference, paper_use, change,
                feeder_metric(audit, labels[experiment]) if experiment in labels else None)
    add_row("gwnet", "REFERENCE", "external baseline", "", "baseline comparison",
            "GWN validation reference.", feeder_metric(audit, "GWN"))

    reject_specs = [
        ("pa_residual_multilevel_loss_dev", "loss exploration", "Tests lambda=0.1 feeder-level term.", "Multilevel loss lambda=0.1", "Overall node WAPE 29.182411; higher than scale-aware 28.945082."),
        ("pa_residual_multilevel_l002_dev", "loss exploration", "Tests lambda=0.02 feeder-level term.", "Multilevel loss lambda=0.02", "Overall node WAPE 29.102186; no node-level gain over scale-aware."),
        ("pa_residual_multilevel_l005_dev", "loss exploration", "Tests lambda=0.05 feeder-level term.", "Multilevel loss lambda=0.05", "Overall node WAPE 29.041759; no node-level gain over scale-aware."),
        ("pa_multilevel_tcn_transformer_dev", "temporal architecture", "Adds a causal two-layer TCN branch parallel to Transformer.", "TCN screen", "Matched against `pa_residual_multilevel_loss_dev`: delta WAPE approximately -0.005934 percentage points; no meaningful isolated gain."),
        ("pa_dynamic_functional_scale_dev", "functional graph", "Adds input-conditioned residual functional relation.", "Dynamic functional graph", "Overall node WAPE 29.360912; no gain over static dual reference."),
        ("pa_multiscale_patch_scale_dev", "temporal architecture", "Adds multi-scale causal patch temporal branch.", "Multi-scale patch", "Overall node WAPE 28.943995, effectively unchanged from scale-aware 28.945082; no isolated evidence of benefit."),
        ("pa_horizon_timequery_scale_dev", "decoder variation", "Conditions horizon queries on deterministic future phase features.", "Future-phase query", "Overall node WAPE 28.899345, worse than plain horizon decoder 28.769896."),
        ("pa_horizon_specific_head_scale_dev", "decoder variation", "Adds zero-initialized horizon-specific correction heads.", "Horizon-specific head", "Overall node WAPE 28.812002, slightly worse than plain horizon decoder 28.769896."),
    ]
    for experiment, role, change, paper_use, reject_reason in reject_specs:
        reference = "pa_residual_multilevel_loss_dev" if experiment == "pa_multilevel_tcn_transformer_dev" else ("pa_residual_scale_loss_dev" if ("multilevel" in experiment or experiment in {"pa_dynamic_functional_scale_dev", "pa_multiscale_patch_scale_dev"}) else "pa_horizon_decoder_scale_dev")
        add_row(experiment, "REJECT", role, reference, paper_use, change,
                feeder_metric(audit, labels[experiment]) if experiment in labels else None)

    for experiment in ("calendar_pa_screen", "calendar_gwnet_screen", "calendar_lstm_screen"):
        add_row(experiment, "OUT_OF_SCOPE", "auxiliary input screen", "",
                "not for innovation-1 claim",
                "8D calendar auxiliary validation; final main input remains 5D periodic input.")

    weekly = audit_metric(audit, "Weekly-lag naive")
    rows.append({
        "experiment": "weekly_lag_naive", "status": "REJECT", "role": "naive sanity",
        "wape": weekly["wape"], "mae": weekly["mae"], "rmse": weekly["rmse"],
        "smape": "", "mape": "", "feeder_wape": feeder_metric(audit, "Weekly-lag naive"),
        "reference_experiment": "", "delta_wape_pp": "", "paper_use": "not a learned experiment",
        "notes": "Audit-only validation naive; no result JSON, so sMAPE/MAPE are unavailable.",
    })

    fields = ["experiment", "status", "role", "wape", "mae", "rmse", "smape", "mape",
              "feeder_wape", "reference_experiment", "delta_wape_pp", "paper_use", "notes"]
    csv_buffer = io.StringIO()
    writer = csv.DictWriter(csv_buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: (fmt(row[key]) if key not in {"experiment", "status", "role", "reference_experiment", "paper_use", "notes"} else row[key]) for key in fields})

    main_table = [
        "| Stage | Experiment | Change | WAPE | MAE | RMSE | sMAPE | MAPE | Delta WAPE (pp) | Decision |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for experiment, status, _, reference, change, paper_use, conclusion in spatial_ablation + main_chain:
        metrics, _, _ = metrics_for(experiment)
        delta = "" if not reference else fmt(metrics["wape"] - metrics_for(reference)[0]["wape"])
        main_table.append(f"| {paper_use} | `{experiment}` | {change} | {fmt(metrics['wape'])} | {fmt(metrics['mae'])} | {fmt(metrics['rmse'])} | {fmt(metrics['smape'])} | {fmt(metrics['mape'])} | {delta} | {status} |")

    comparison_specs = [
        ("Horizon Decoder vs residual-scale", "PA-STFed horizon-decoder", "PA-STFed residual-scale-loss"),
        ("Horizon Decoder vs GWN", "PA-STFed horizon-decoder", "GWN"),
        ("WL1 vs Horizon Decoder", "PA-STFed horizon-decoder-wl1", "PA-STFed horizon-decoder"),
        ("WL1 vs GWN", "PA-STFed horizon-decoder-wl1", "GWN"),
    ]
    horizons = ["step1", "step3", "step6", "step12", "overall_12step"]
    comparison_table = [
        "| Comparison | Horizon | A WAPE | A MAE | A RMSE | A feeder WAPE | B WAPE | B MAE | B RMSE | B feeder WAPE | A-B WAPE pp |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, left, right in comparison_specs:
        for horizon in horizons:
            a = audit_metric(audit, left, horizon)
            b = audit_metric(audit, right, horizon)
            comparison_table.append(f"| {name} | {horizon} | {fmt(a['wape'])} | {fmt(a['mae'])} | {fmt(a['rmse'])} | {fmt(feeder_metric(audit, left, horizon))} | {fmt(b['wape'])} | {fmt(b['mae'])} | {fmt(b['rmse'])} | {fmt(feeder_metric(audit, right, horizon))} | {fmt(a['wape'] - b['wape'])} |")

    reject_table = [
        "| Exploration | Motivation | Core change | Key validation result | Reject reason |",
        "|---|---|---|---|---|",
        "| Weekly context | Test seasonal naive context | Weekly-lag naive (`t+h-672`); audit-only | WAPE 57.424595; feeder WAPE 26.339432 | Much worse than learned models; not a proposed model component. |",
    ]
    for experiment, role, change, paper_use, reject_reason in reject_specs:
        metrics, _, _ = metrics_for(experiment)
        reject_table.append(f"| `{experiment}` | {role} | {change} | WAPE {fmt(metrics['wape'])}; MAE {fmt(metrics['mae'])}; RMSE {fmt(metrics['rmse'])} | {reject_reason} |")

    best_wape = metrics_for("pa_horizon_decoder_wl1_dev")[0]["wape"]
    markdown = f"""# Innovation 1 Experiment Ledger

This ledger consolidates existing centralized development and validation artifacts only. No training, test evaluation, checkpoint rewrite, or existing result JSON modification was performed. Directly comparable model rows use seed 2026. The main table reads `best_validation` from each result JSON; the horizon comparison table reads the already-generated validation audit.

## 1. Task and Evaluation Protocol

- Dataset/task: SMART-DS v0.9, 2018 Full_Texas / P10R feeder.
- Target set: 92 target load nodes.
- Forecast window: history 96 -> horizon 12 (15-minute sampling).
- Scope: validation-only development screening; `test_evaluated=false`, `test=null`.
- Primary metric: node-micro WAPE.
- Auxiliary operational metric: feeder aggregate WAPE.
- Error metrics are reported after inverse normalization where specified by the source artifacts.
- Topology note: topology-kNN has 352 target-relation edges; these are target-graph relations, not 352 physical Line elements.

## 2. Innovation 1 Mainline Evolution

Values in this table are direct fields from each seed-2026 result JSON. Physical-only, Functional-only, and Dual are parallel spatial ablations; deltas begin with the incremental chain from Dual and are current WAPE minus the reference row in percentage points.

{chr(10).join(main_table)}

Interpretation by step:

- Physical-only, Functional-only, and Dual Physical+Functional are parallel spatial ablations.
- The incremental chain starts at Dual: Residual Anchor -> Scale-aware Charbonnier -> Horizon Decoder -> WAPE-numerator-aligned WL1.
- Residual Anchor changes the output to last observed load plus learned correction and is a development-only increment.
- Scale-aware loss weights normalized residuals by the training-split IQR; it is an objective change, not a graph change.
- Horizon Decoder adds horizon cross-attention while retaining the existing temporal representation and head path.
- WAPE-aligned WL1 changes the training loss to a scale-weighted absolute-error numerator; it is the current node-level development candidate.

## 3. Rejected Explorations

{chr(10).join(reject_table)}

The calendar screens are recorded in the CSV as `OUT_OF_SCOPE`: they used auxiliary 8D calendar input and do not enter the final 5D main-model claim. None of the rejected screens is presented as a contribution.

## 4. Current Key Comparisons

The following values are absolute validation metrics from `reports/supporting/forecastability_audit.json`; they include exact steps 1/3/6/12 and overall 12-step. A-B WAPE is in percentage points; negative means A has lower node WAPE. Feeder WAPE is an auxiliary aggregation-level metric.

{chr(10).join(comparison_table)}

## 5. Current Conclusions

1. The current node-level best development candidate is `pa_horizon_decoder_wl1_dev`, with validation WAPE approximately **{best_wape:.4f}**.
2. This candidate has a trade-off: RMSE and feeder aggregate WAPE are not uniformly better than the plain Horizon Decoder or residual-scale variants.
3. The relative gain versus GWN cannot all be attributed to model structure because residual anchoring, scale-aware loss, and WL1 also change the objective/output path.
4. The necessary next fairness control is **GWN + the same WL1 objective**.
5. The project has not entered final multi-seed confirmation or final test lock.

## 6. Candidate Paper Ablation Chain

`Physical-only || Functional-only || Dual -> +Residual Anchor -> +Scale-aware Charbonnier -> +Horizon Decoder -> +WAPE-numerator-aligned WL1`

This is a candidate ablation chain, not a claim that every intermediate component is independently publishable.

## 7. Consistency Checks

{chr(10).join(checks)}
- No table treats the 352 topology-kNN target relation edges as physical Line elements.
- Mainline and exploration metric columns in the CSV are generated from the corresponding result JSON `best_validation` objects; unavailable feeder/sMAPE/MAPE values remain blank with a note.
- Horizon and feeder comparison values are copied from the existing forecastability audit artifact, not recomputed or inferred here.
- No training or test loader was run by this consolidation task.

Source files used: `results/*.json`, `reports/supporting/forecastability_audit.json`, `reports/forecastability_audit.md`, and `code/experiments.yaml`.
"""

    (REPORTS / "innovation1_experiment_ledger.md").write_text(markdown.rstrip() + "\n", encoding="utf-8")
    (REPORTS / "innovation1_experiment_ledger.csv").write_text(csv_buffer.getvalue(), encoding="utf-8")


if __name__ == "__main__":
    main()
