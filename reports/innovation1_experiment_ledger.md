# Innovation 1 Experiment Ledger

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

Values in this table are direct fields from each seed-2026 result JSON. Delta is current WAPE minus the immediately preceding row in percentage points.

| Stage | Experiment | Change | WAPE | MAE | RMSE | sMAPE | MAPE | Delta WAPE (pp) | Decision |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| Physical-only | `spatial_physical_only` | Physical topology branch only; target relation graph is not a physical Line graph. | 30.474928 | 0.357362 | 0.739897 | 30.014791 | 34.577473 |  | KEEP |
| Functional-only | `spatial_functional_only` | Functional relation branch only; no physical branch. | 29.992157 | 0.351701 | 0.732202 | 29.633430 | 33.734718 | -0.482771 | KEEP |
| Dual physical+functional | `centralized` | Physical + static functional PA-STFed dual base. | 29.373686 | 0.344448 | 0.714534 | 28.947639 | 33.148235 | -0.618471 | KEEP |
| Residual Anchor | `pa_residual_anchor_dev` | Adds persistence residual anchor: last observed load plus learned correction. | 28.996521 | 0.340025 | 0.720468 | 28.597759 | 31.308863 | -0.377165 | KEEP |
| Scale-aware loss | `pa_residual_scale_loss_dev` | Uses training-IQR scale-aware node loss. | 28.945082 | 0.339422 | 0.712346 | 28.842655 | 32.687969 | -0.051439 | KEEP |
| Horizon Decoder | `pa_horizon_decoder_scale_dev` | Adds one-layer horizon cross-attention decoder with zero-initialized correction head. | 28.769896 | 0.337368 | 0.718085 | 28.528690 | 30.896641 | -0.175186 | KEEP |
| WAPE-aligned WL1 | `pa_horizon_decoder_wl1_dev` | Uses WAPE-aligned scale-aware L1 objective. | 27.998379 | 0.328321 | 0.729947 | 26.768181 | 27.881538 | -0.771517 | CURRENT BEST |

Interpretation by step:

- Physical-only and functional-only are retained as controlled ablations.
- Dual Physical+Functional is the base configuration for incremental screens.
- Residual Anchor changes the output to last observed load plus learned correction and is a development-only increment.
- Scale-aware loss weights normalized residuals by the training-split IQR; it is an objective change, not a graph change.
- Horizon Decoder adds horizon cross-attention while retaining the existing temporal representation and head path.
- WAPE-aligned WL1 changes the training loss to a scale-weighted absolute-error numerator; it is the current node-level development candidate.

## 3. Rejected Explorations

| Exploration | Motivation | Core change | Key validation result | Reject reason |
|---|---|---|---|---|
| Weekly context | Test seasonal naive context | Weekly-lag naive (`t+h-672`); audit-only | WAPE 57.424595; feeder WAPE 26.339432 | Much worse than learned models; not a proposed model component. |
| `pa_residual_multilevel_loss_dev` | loss exploration | Tests lambda=0.1 feeder-level term. | WAPE 29.182411; MAE 0.342205; RMSE 0.710936 | Overall node WAPE 29.182411; higher than scale-aware 28.945082. |
| `pa_residual_multilevel_l002_dev` | loss exploration | Tests lambda=0.02 feeder-level term. | WAPE 29.102186; MAE 0.341265; RMSE 0.711638 | Overall node WAPE 29.102186; no node-level gain over scale-aware. |
| `pa_residual_multilevel_l005_dev` | loss exploration | Tests lambda=0.05 feeder-level term. | WAPE 29.041759; MAE 0.340556; RMSE 0.711752 | Overall node WAPE 29.041759; no node-level gain over scale-aware. |
| `pa_multilevel_tcn_transformer_dev` | temporal architecture | Adds a causal two-layer TCN branch parallel to Transformer. | WAPE 29.176477; MAE 0.342136; RMSE 0.712230 | Overall node WAPE 29.176477; no isolated gain and the screen also changes loss to multilevel. |
| `pa_dynamic_functional_scale_dev` | functional graph | Adds input-conditioned residual functional relation. | WAPE 29.360912; MAE 0.344298; RMSE 0.719380 | Overall node WAPE 29.360912; no gain over static dual reference. |
| `pa_multiscale_patch_scale_dev` | temporal architecture | Adds multi-scale causal patch temporal branch. | WAPE 28.943995; MAE 0.339409; RMSE 0.717039 | Overall node WAPE 28.943995, effectively unchanged from scale-aware 28.945082; no isolated evidence of benefit. |
| `pa_horizon_timequery_scale_dev` | decoder variation | Conditions horizon queries on deterministic future phase features. | WAPE 28.899345; MAE 0.338886; RMSE 0.716711 | Overall node WAPE 28.899345, worse than plain horizon decoder 28.769896. |
| `pa_horizon_specific_head_scale_dev` | decoder variation | Adds zero-initialized horizon-specific correction heads. | WAPE 28.812002; MAE 0.337862; RMSE 0.718431 | Overall node WAPE 28.812002, slightly worse than plain horizon decoder 28.769896. |

The calendar screens are recorded in the CSV as `OUT_OF_SCOPE`: they used auxiliary 8D calendar input and do not enter the final 5D main-model claim. None of the rejected screens is presented as a contribution.

## 4. Current Key Comparisons

The following values are absolute validation metrics from `results/forecastability_audit.json`; they include exact steps 1/3/6/12 and overall 12-step. A-B WAPE is in percentage points; negative means A has lower node WAPE. Feeder WAPE is an auxiliary aggregation-level metric.

| Comparison | Horizon | A WAPE | A MAE | A RMSE | A feeder WAPE | B WAPE | B MAE | B RMSE | B feeder WAPE | A-B WAPE pp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Horizon Decoder vs residual-scale | step1 | 17.750845 | 0.207983 | 0.459357 | 5.389872 | 17.781958 | 0.208347 | 0.463068 | 4.989458 | -0.031113 |
| Horizon Decoder vs residual-scale | step3 | 27.135302 | 0.318028 | 0.692940 | 9.250701 | 27.326166 | 0.320265 | 0.689506 | 8.630059 | -0.190865 |
| Horizon Decoder vs residual-scale | step6 | 29.674500 | 0.347955 | 0.742792 | 10.591512 | 29.848616 | 0.349996 | 0.736674 | 9.469649 | -0.174116 |
| Horizon Decoder vs residual-scale | step12 | 32.536701 | 0.381859 | 0.774510 | 11.749749 | 32.683887 | 0.383587 | 0.767869 | 10.554098 | -0.147186 |
| Horizon Decoder vs residual-scale | overall_12step | 28.769896 | 0.337368 | 0.718085 | 10.118258 | 28.945082 | 0.339422 | 0.712346 | 9.034604 | -0.175186 |
| Horizon Decoder vs GWN | step1 | 17.750845 | 0.207983 | 0.459357 | 5.389872 | 17.389387 | 0.203748 | 0.457321 | 5.008239 | 0.361458 |
| Horizon Decoder vs GWN | step3 | 27.135302 | 0.318028 | 0.692940 | 9.250701 | 27.355606 | 0.320610 | 0.692143 | 8.315418 | -0.220304 |
| Horizon Decoder vs GWN | step6 | 29.674500 | 0.347955 | 0.742792 | 10.591512 | 30.064434 | 0.352527 | 0.740252 | 9.366537 | -0.389935 |
| Horizon Decoder vs GWN | step12 | 32.536701 | 0.381859 | 0.774510 | 11.749749 | 32.898613 | 0.386107 | 0.765072 | 9.715222 | -0.361912 |
| Horizon Decoder vs GWN | overall_12step | 28.769896 | 0.337368 | 0.718085 | 10.118258 | 29.005434 | 0.340130 | 0.713006 | 8.767666 | -0.235538 |
| WL1 vs Horizon Decoder | step1 | 16.817087 | 0.197042 | 0.469155 | 5.729385 | 17.750845 | 0.207983 | 0.459357 | 5.389872 | -0.933758 |
| WL1 vs Horizon Decoder | step3 | 26.266542 | 0.307846 | 0.708994 | 12.433867 | 27.135302 | 0.318028 | 0.692940 | 9.250701 | -0.868759 |
| WL1 vs Horizon Decoder | step6 | 29.005371 | 0.340109 | 0.755823 | 12.970934 | 29.674500 | 0.347955 | 0.742792 | 10.591512 | -0.669128 |
| WL1 vs Horizon Decoder | step12 | 31.835772 | 0.373633 | 0.782380 | 13.370399 | 32.536701 | 0.381859 | 0.774510 | 11.749749 | -0.700930 |
| WL1 vs Horizon Decoder | overall_12step | 27.998384 | 0.328321 | 0.729947 | 12.311967 | 28.769896 | 0.337368 | 0.718085 | 10.118258 | -0.771511 |
| WL1 vs GWN | step1 | 16.817087 | 0.197042 | 0.469155 | 5.729385 | 17.389387 | 0.203748 | 0.457321 | 5.008239 | -0.572300 |
| WL1 vs GWN | step3 | 26.266542 | 0.307846 | 0.708994 | 12.433867 | 27.355606 | 0.320610 | 0.692143 | 8.315418 | -1.089064 |
| WL1 vs GWN | step6 | 29.005371 | 0.340109 | 0.755823 | 12.970934 | 30.064434 | 0.352527 | 0.740252 | 9.366537 | -1.059063 |
| WL1 vs GWN | step12 | 31.835772 | 0.373633 | 0.782380 | 13.370399 | 32.898613 | 0.386107 | 0.765072 | 9.715222 | -1.062841 |
| WL1 vs GWN | overall_12step | 27.998384 | 0.328321 | 0.729947 | 12.311967 | 29.005434 | 0.340130 | 0.713006 | 8.767666 | -1.007050 |

## 5. Current Conclusions

1. The current node-level best development candidate is `pa_horizon_decoder_wl1_dev`, with validation WAPE approximately **27.9984**.
2. This candidate has a trade-off: RMSE and feeder aggregate WAPE are not uniformly better than the plain Horizon Decoder or residual-scale variants.
3. The relative gain versus GWN cannot all be attributed to model structure because residual anchoring, scale-aware loss, and WL1 also change the objective/output path.
4. The necessary next fairness control is **GWN + the same WL1 objective**.
5. The project has not entered final multi-seed confirmation or final test lock.

## 6. Candidate Paper Ablation Chain

`Physical-only -> Functional-only -> Dual -> +Residual Anchor -> +Horizon Decoder -> +WAPE-aligned WL1`

This is a candidate ablation chain, not a claim that every intermediate component is independently publishable.

## 7. Consistency Checks

- `pa_residual_anchor_dev`: `pa_residual_anchor_dev_seed2026_centralized_result.json`, `test_evaluated=false`, `test=null` (PASS)
- `pa_residual_scale_loss_dev`: `pa_residual_scale_loss_dev_seed2026_centralized_result.json`, `test_evaluated=false`, `test=null` (PASS)
- `pa_residual_multilevel_loss_dev`: `pa_residual_multilevel_loss_dev_seed2026_centralized_result.json`, `test_evaluated=false`, `test=null` (PASS)
- `pa_residual_multilevel_l002_dev`: `pa_residual_multilevel_l002_dev_seed2026_centralized_result.json`, `test_evaluated=false`, `test=null` (PASS)
- `pa_residual_multilevel_l005_dev`: `pa_residual_multilevel_l005_dev_seed2026_centralized_result.json`, `test_evaluated=false`, `test=null` (PASS)
- `pa_multilevel_tcn_transformer_dev`: `pa_multilevel_tcn_transformer_dev_seed2026_centralized_result.json`, `test_evaluated=false`, `test=null` (PASS)
- `pa_dynamic_functional_scale_dev`: `pa_dynamic_functional_scale_dev_seed2026_centralized_result.json`, `test_evaluated=false`, `test=null` (PASS)
- `pa_multiscale_patch_scale_dev`: `pa_multiscale_patch_scale_dev_seed2026_centralized_result.json`, `test_evaluated=false`, `test=null` (PASS)
- `pa_horizon_decoder_scale_dev`: `pa_horizon_decoder_scale_dev_seed2026_centralized_result.json`, `test_evaluated=false`, `test=null` (PASS)
- `pa_horizon_timequery_scale_dev`: `pa_horizon_timequery_scale_dev_seed2026_centralized_result.json`, `test_evaluated=false`, `test=null` (PASS)
- `pa_horizon_specific_head_scale_dev`: `pa_horizon_specific_head_scale_dev_seed2026_centralized_result.json`, `test_evaluated=false`, `test=null` (PASS)
- `pa_horizon_decoder_wl1_dev`: `pa_horizon_decoder_wl1_dev_seed2026_centralized_result.json`, `test_evaluated=false`, `test=null` (PASS)
- `calendar_pa_screen`: `calendar_pa_screen_seed2026_centralized_result.json`, `test_evaluated=false`, `test=null` (PASS)
- `calendar_gwnet_screen`: `calendar_gwnet_screen_seed2026_centralized_result.json`, `test_evaluated=false`, `test=null` (PASS)
- `calendar_lstm_screen`: `calendar_lstm_screen_seed2026_centralized_result.json`, `test_evaluated=false`, `test=null` (PASS)
- No table treats the 352 topology-kNN target relation edges as physical Line elements.
- Mainline and exploration metric columns in the CSV are generated from the corresponding result JSON `best_validation` objects; unavailable feeder/sMAPE/MAPE values remain blank with a note.
- Horizon and feeder comparison values are copied from the existing forecastability audit artifact, not recomputed or inferred here.
- No training or test loader was run by this consolidation task.

Source files used: `results/*.json`, `results/forecastability_audit.json`, `reports/forecastability_audit.md`, and `code/experiments.yaml`.
