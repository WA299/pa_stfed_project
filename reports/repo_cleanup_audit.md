# Repository Cleanup Audit

审计基线：`main` at `6b474436d2b4cb9f8928daf55d5bb86bbaa917db`。本报告仅执行静态文件、配置和引用审计；未修改、删除或移动既有文件，未训练，未创建 test loader，未访问 test 数据。审计时工作区已有一项与本任务无关的本地修改：`results/audit_report.json`，本报告不将其视为 cleanup 产物，也不建议在未核对来源前提交。

## 1. Repository inventory

### 1.1 Scope summary

| Scope | Git-tracked files | Tracked bytes | Local files excluding `__pycache__` | Local bytes | Interpretation |
|---|---:|---:|---:|---:|---|
| `code/` | 13 | 386,132 | 13 | 386,132 | 训练、模型、数据、联邦、分析与一次性恢复/报告脚本 |
| `preprocess/` | 10 | 176,669 | 10 | 176,669 | 官方 SMART-DS 重建、验证与 provenance |
| `reports/` | 14 | 258,231 | 14 | 258,231 | 人读审计报告与 supporting evidence |
| `results/` | 61 | 3,038,967 | 111 | 93,963,876 | Git 只跟踪 JSON/CSV；50 个本地 `.pt` checkpoint 被 `.gitignore` 排除 |
| **Total** | **98** | **3,859,999** | **148** | **94,784,908** | 不含 ignored `__pycache__`；整个仓库另有 data/literature/root 文件 |

`results/` 的本地 50 个 `.pt` 共 90,924,909 bytes，均被 `*.pt` 规则忽略，Git history **不能**恢复这些 checkpoint。JSON 有 56 个、约 3.03 MB；CSV 4 个；`.gitkeep` 1 个。Python caches 被 `__pycache__/` 和 `*.py[cod]` 正确忽略。

### 1.2 `code/` file inventory

| File | Size | Main responsibility | Current reference state |
|---|---:|---|---|
| `code/run.py` | 145,928 B / 3,397 lines | CLI、审计、centralized/federated 训练、ALA、评估、result ledger | **ACTIVE**：正式入口；README 调用；其他分析脚本复用其函数 |
| `code/model.py` | 54,344 B / 1,212 lines | PA-STFed、四类 baseline、图模块、TCN/patch/horizon decoder、FL 参数分组 | **ACTIVE**：被 `run.py`、`federated.py`、恢复脚本导入 |
| `code/data.py` | 53,745 B / 1,117 lines | canonical 数据加载、图模式、客户端划分、窗口与 5D 输入 | **ACTIVE**：训练、审计和 topology preprocess 共用 |
| `code/forecastability_audit.py` | 37,361 B / 724 lines | validation-only checkpoint 推理、horizon/feeder/node/naive audit | **STANDALONE ACTIVE AUDIT**：不被主入口调用；硬编码多个 checkpoint 名 |
| `code/experiments.yaml` | 25,697 B | 正式、开发、smoke、历史 rejected 与 final-test 配置矩阵 | **ACTIVE**：`run.py` 运行时加载 |
| `code/build_innovation1_ledger.py` | 16,976 B / 269 lines | 从现有 JSON 生成创新点 1 ledger | **ONE-SHOT/REPRODUCIBLE REPORT**：不被主入口导入 |
| `code/federated.py` | 14,001 B / 335 lines | local train、loss、metric summary、聚合辅助 | **ACTIVE**：被 `run.py` 和 forecastability audit 使用 |
| `code/analyze_results.py` | 10,385 B / 252 lines | 配对 block bootstrap 与结果比较 | **STANDALONE ANALYSIS**：无 import/config 引用，仅 CLI 使用 |
| `code/config.yaml` | 9,436 B | canonical 数据、5D 输入、模型和训练默认值 | **ACTIVE**：`run.py` 加载；所有 experiment 继承 |
| `code/recover_multiscale_result.py` | 7,285 B / 144 lines | 从已有 multiscale checkpoint 恢复 validation result | **ONE-TIME RECOVERY**：仅 rejected patch 实验服务，不被其他代码调用 |
| `code/config.py` | 3,819 B / 103 lines | YAML、seed、device、AMP runtime | **ACTIVE**：训练、联邦和 audit 共用 |
| `code/metrics.py` | 3,931 B / 96 lines | MAE/RMSE/WAPE/sMAPE/MAPE、gate/CKA diagnostics | **ACTIVE**：训练与分析共用 |
| `code/privacy.py` | 3,224 B / 99 lines | client update clipping、DP aggregation、RDP epsilon | **ACTIVE CONDITIONAL**：被 federated/run 导入；只在 DP 配置启用 |

### 1.3 `preprocess/` file inventory

| File | Size | Main responsibility | Current reference state |
|---|---:|---|---|
| `audit_smartds_data.py` | 54,527 B | 官方 OpenDSS scope、拓扑、legacy/load provenance 总审计 | **FOUNDATIONAL**：被 build/load audit/verify 导入 |
| `build_official_load_series.py` | 25,731 B | 官方 LoadShapes 重建 canonical load series 与 target projection | **CANONICAL REBUILD**：导入 `audit_smartds_data` |
| `audit_exogenous_alignment.py` | 19,701 B | 61 parquet timestamp、一致性 caveat、calendar sidecar | **PROVENANCE/AUDIT**：独立 CLI；产物仍被 data loader引用 |
| `build_smartds_graph_v2.py` | 18,130 B | 从官方 raw 重建 full/target graph v2 | **CANONICAL REBUILD**：调用总审计和 raw parser |
| `audit_target_topology.py` | 17,661 B | target hop/kNN/client partition 审计 | **STAGE-0 AUDIT**：直接复用 `code/data.py` |
| `audit_official_load_data.py` | 11,920 B | official parquet 与 OpenDSS load mapping 核验 | **PROVENANCE/AUDIT**：复用官方 parser |
| `validate_smartds_v2.py` | 10,190 B | processed/raw/manifest/determinism assertions | **ACTIVE VALIDATOR**：README 给出运行命令 |
| `fetch_smartds_profiles.py` | 8,674 B | 下载最小官方 profile/load data scope | **REBUILD INPUT ACQUISITION**：独立 CLI |
| `write_manifests.py` | 5,530 B | raw/profile manifest 与 SHA/source metadata | **REBUILD PROVENANCE**：独立 CLI |
| `verify_official_raw.py` | 4,605 B | 官方远端 raw 文件 SHA 核对 | **REBUILD PROVENANCE**：复用 `audit_smartds_data` 私有 `_scope_files` |

注意：preprocess 之间通过将同目录加入 `sys.path` 的脚本式 import 复用，而非稳定 package API；`verify_official_raw.py` 依赖下划线私有函数，是 cleanup 时需要修正的边界。

### 1.4 `reports/` file inventory

| File | Size | Main responsibility | Current reference state |
|---|---:|---|---|
| `reports/forecastability_audit.md` | 61,836 B | validation-only horizon、aggregation、node 与 seasonal predictability audit | **ACTIVE EVIDENCE**：创新点 ledger 的人读来源；内容很长 |
| `reports/supporting/official_load_mapping.csv` | 53,355 B | 92 target 的官方负荷映射证据 | **CANONICAL EVIDENCE**：不被 runtime import |
| `reports/supporting/node_role_audit.csv` | 39,804 B | 273 节点角色与结构/target 解释 | **CANONICAL EVIDENCE** |
| `reports/supporting/rdt_rdtlv_verification.csv` | 30,718 B | 56 对 Transformer 官方验证 | **CANONICAL EVIDENCE** |
| `reports/supporting/exogenous_alignment.json` | 30,406 B | timestamp/calendar alignment machine-readable evidence | **CANONICAL EVIDENCE** |
| `reports/innovation1_experiment_ledger.md` | 12,832 B | 创新点 1 主线、reject、fairness gap | **ACTIVE CURRENT STATUS**：由 ledger script 生成 |
| `reports/supporting/official_raw_remote_verification.json` | 8,664 B | official raw local/remote SHA 比对 | **CANONICAL EVIDENCE** |
| `reports/innovation1_experiment_ledger.csv` | 6,165 B | 每实验结构化 ledger | **ACTIVE CURRENT STATUS**：由 ledger script 生成 |
| `reports/target_topology_audit.md` | 5,370 B | topology-kNN 与 8-client tree partition freeze | **ACTIVE DATA/GRAPH EVIDENCE** |
| `reports/exogenous_data_audit.md` | 2,951 B | timestamp PASS with caveat、temperature unavailable | **ACTIVE DATA EVIDENCE** |
| `reports/data_audit.md` | 2,638 B | SMART-DS source、full topology、target/load caveats | **ACTIVE DATA EVIDENCE** |
| `reports/supporting/client_topology_stats.csv` | 1,622 B | client induced topology statistics | **SUPPORTING EVIDENCE** |
| `reports/supporting/validation_report.json` | 1,225 B | canonical processed validation assertions | **README-TARGETED EVIDENCE** |
| `reports/supporting/target_knn_stats.csv` | 645 B | k=2/4/6/8 Stage-0 audit | **HISTORICAL SELECTION EVIDENCE** |

There is no `reports/forecastability_audit.json`; the machine-readable audit is currently `results/forecastability_audit.json`. This path mismatch is a documentation/organization issue, not missing data.

### 1.5 `results/` inventory by complete filename groups

All Git-tracked result JSONs are validation/development evidence unless explicitly named as manifest/summary. Local `.pt` checkpoints mirror many rows but are ignored.

| Files/pattern (complete group coverage) | Count / tracked size | Responsibility | Referenced now |
|---|---:|---|---|
| `centralized_seed2026/2027/2028_centralized_result.json` | 3 / 40,348 B | PA-STFed formal validation | all ledger/summary and audit |
| `spatial_physical_only_seed2026/2027/2028_*`, `spatial_functional_only_seed2026/2027/2028_*` | 6 / 87,951 B | innovation-1 graph ablations | innovation ledger |
| `gwnet_seed2026/2027/2028_*`, `agcrn_seed2026/2027/2028_*`, `load_lstm_validation_seed2026/2027/2028_*`, `itransformer_validation_seed2026/2027/2028_*` | 12 / 172,156 B | centralized baselines | formal summary; GWN audit/ledger |
| `calendar_{pa,gwnet,lstm}_screen_seed2026_*` | 3 / 37,944 B | auxiliary 8D screening | innovation ledger only |
| `pa_residual_anchor_dev_*`, `pa_residual_scale_loss_dev_*`, `pa_horizon_decoder_scale_dev_*`, `pa_horizon_decoder_wl1_dev_*` | 4 / 55,472 B | active innovation-1 development chain | forecastability audit and ledger |
| `pa_residual_multilevel_{loss,l002,l005}_dev_*`, `pa_multilevel_tcn_transformer_dev_*`, `pa_dynamic_functional_scale_dev_*`, `pa_multiscale_patch_scale_dev_*`, `pa_horizon_timequery_scale_dev_*`, `pa_horizon_specific_head_scale_dev_*` | 8 / 105,007 B | rejected development evidence | forecastability/innovation ledger; some lack feeder audit |
| `fedavg_grouped_seed2026_*`, `fedprox005_grouped_seed2026_*`, `personalized_head_seed2026_*`, `local_only_seed2026_*` | 4 / 609,387 B | Stage-3 federated validation | formal manifests/summaries |
| `{fedavg,fedprox005,personalized_head,local_only}_r40_seed2026_*` | 4 / 819,066 B | convergence audit | config/history; not formal main summary |
| `modulelocal_dev_*`, `layerala_dev_*`, `moduleala_dev_*` | 3 / 671,285 B | personalized-FL development | development ledger |
| `stage_ala_smoke_*`, `stage_fedala_smoke_*` | 2 / 94,801 B | ALA smoke audit | config/history only |
| `baseline_grouped_seed2026_baseline_result.json` | 1 / 5,435 B | naive baseline | forecastability audit |
| `all_manifest.json`, `all_summary.json` | 2 / 37,104 B | formal result ledger | `run_all` merge path |
| `development_manifest.json`, `development_summary.json` | 2 / 28,252 B | development result ledger | `run_all` development path |
| `forecastability_audit.json` | 1 / 246,992 B | machine-readable validation audit | audit Markdown and innovation ledger |
| `audit_report.json` | 1 / 23,556 B | runtime/data/graph contract audit | generated by `run.py`; locally modified |
| `federated_parameter_groups.csv` | 3,594 B | FL parameter grouping evidence | no runtime reader |
| `bridge_edges.csv`, `legacy_bridge_edges.csv`, `topology_comparison.csv` | 3 / 616 B | legacy MST/topology comparison evidence | no active training; historical audit only |
| `.gitkeep` | 1 B | preserve results directory | no runtime need once tracked files exist |

Local-only checkpoints: 50 `.pt` totaling 90.9 MB. The most obvious duplicate/transient candidate is `pa_horizon_decoder_scale_dev_smoke_seed2026_centralized_model.pt` (789,889 B). Rejected-experiment checkpoints are still needed if future read-only inference must be reproduced without retraining; they cannot be recovered from Git.

## 2. `code/` redundancy and maintainability findings

### 2.1 Oversized functions/classes

| Location | Span | Finding | Suggested boundary |
|---|---:|---|---|
| `run.py:federated` | 614 lines | algorithm dispatch, client lifecycle, ALA, aggregation, validation, checkpoint/result serialization in one function | split orchestration, client adaptation, validation, serialization |
| `forecastability_audit.py:main` | 479 lines | checkpoint registry, inference, naive generation, statistics, JSON and Markdown rendering coupled | data-driven method registry + metric engine + renderer |
| `run.py:centralized` | 339 lines | train, early stopping, diagnostics, checkpoint/result output coupled | trainer/evaluator/result writer |
| `run.py:audit` | 332 lines | canonical graph checks mixed with legacy forest/MST diagnostics | move legacy diagnostics to preprocess/audit-only code |
| `model.py:PA_STFed` | 230 lines | active mainline plus rejected optional branches in one constructor/forward | active PA core plus optional experimental adapters |
| `code/build_innovation1_ledger.py:main` | 219 lines | extraction, policy classification and rendering coupled | small reporting helpers if script is retained |
| `run.py:config_brief` | 228 lines | duplicates many model/training fields also serialized in centralized/federated results | one canonical resolved-config serializer |
| `run.py:run_all` | 205 lines | job selection, resume, ledger merge, output paths coupled | experiment registry + result ledger service |
| `data.py:topology_client_partition` | 163 lines | deterministic tree partition solver nested inside data container | dedicated partition helper module/function |
| `model.py:HorizonCrossAttentionDecoder` | 145 lines | active decoder and two rejected variants share conditional logic | retain base decoder; isolate/remove rejected query/head adapters |

### 2.2 Repeated logic

- Result serialization metadata is assembled in both `run.py:centralized`, `run.py:federated`, `run.py:config_brief`, and partially in `recover_multiscale_result.py`. This caused prior metadata recovery issues and should become one schema builder.
- Graph/model construction is repeated between `run.py`, `forecastability_audit.py`, and `recover_multiscale_result.py`; audit already imports helpers from `run.py`, but still hardcodes checkpoint names and method calls.
- WAPE/MAE/RMSE are available in `metrics.py`, while `forecastability_audit.py` defines `_basic_metrics`, `_node_wape` and `_feeder_wape`. Aggregation-specific functions are legitimate, but base metric formulas should have one source.
- Result ledger paths and experiment classification appear in `run.py`, `experiments.yaml`, and `build_innovation1_ledger.py`. The report generator currently embeds KEEP/REJECT decisions and exact prose, so every experiment rename needs multiple edits.
- Legacy topology reconstruction exists in `data.py` and is invoked again by `run.py:audit`; current training uses `topology_knn` only.

### 2.3 Deprecated/conditional feature flags

- Legacy graph modes `raw`, `inf`, `legacy_inf`, `forest`, `mst_no_tag`, `mst_tag`, `projected`, `projected_inf` remain in `GraphMode` and `graph_view`. They are absent from the formal experiment matrix but still referenced by `run.py:audit`; therefore not safe to remove independently.
- `use_multiscale_patch_branch`, `temporal_architecture=tcn_transformer`, `functional_graph_mode=dynamic_residual`, `horizon_query_time_features`, and `horizon_specific_residual_head` only serve rejected screens.
- `spatial_dual_graph` duplicates the effective dual settings of `centralized`; it has no result JSON in the current repository. Keeping both names risks accidental duplicate runs.
- `config.yaml` defaults to `federated.algorithm=FedProx` and `mu=0.005`. Experiments override this correctly, but the comment saying `mu=0` is FedAvg no longer captures the runtime assertion that proximal loss is enabled only for `algorithm == FedProx`. The default is safe but semantically surprising.
- Final-test configurations (`centralized_test`, `fedavg_test`, `personalized_head_test`) are intentionally dormant and must not be removed merely because unused; they are test-lock entry points and explicitly set `evaluate_test=true`.

### 2.4 Uncalled and one-time code

- Static text-reference scanning found `check_horizon_decoder_initialization_consistency()` defined but not called. It appears to be a completed development assertion, not runtime behavior. Confirm with a real call-graph/test pass before deletion.
- `recover_multiscale_result.py` is a one-time repair utility for a rejected experiment. Its output now exists; it is not imported elsewhere. **SAFE_DELETE candidate after confirming the JSON is complete.**
- `build_innovation1_ledger.py` and `analyze_results.py` are standalone reporting tools. They are not dead code, but should not sit beside the training entry point if a cleanup proceeds.
- `forecastability_audit.py` is a standalone audit program, not training code; it should remain reproducible but should not continue accumulating hardcoded per-checkpoint branches.
- No reliable automatic conclusion of “unused” should be based only on imports: CLI entry points and PyTorch dunder methods are intentionally invoked indirectly.

## 3. Rejected-path reference trace

| Rejected path | Implementation/config/results/report references | Decision | Why / prerequisite |
|---|---|---|---|
| TCN | `CausalTCNBranch`, `TemporalEncoder`, `PA_STFed`, `run.py` config/result metadata, `pa_multilevel_tcn_transformer_dev`, result JSON, innovation ledger | **STILL_REFERENCED** | Cannot delete class alone. First freeze ledger, remove experiment config, stop serializing TCN metadata, then remove branch and run checkpoint compatibility checks. |
| Dynamic functional graph | `StaticFunctionalGraph(dynamic_residual)`, PA constructor, `run.py`, experiment/result, ledger | **STILL_REFERENCED** | Static and dynamic paths share one class. Refactor static path or remove conditional code atomically. |
| Multiscale patch | `MultiScalePatchTemporalBranch`, PA constructor/forward, metadata helper, `run.py`, experiment/result, `recover_multiscale_result.py`, ledger | **NEED_REFACTOR** | Widest rejected dependency fan-out; remove recovery script/config/report runtime references before deleting module code. |
| Future-phase query | conditional fields and `time_query` inside active `HorizonCrossAttentionDecoder`, PA/run metadata, experiment/result, ledger | **NEED_REFACTOR** | Active plain Horizon Decoder must remain bitwise/initialization compatible. Extract or remove only the query-conditioning extension, retaining base parameter initialization order. |
| Horizon-specific head | conditional parameters inside active decoder, metadata and initialization check, experiment/result/audit/ledger | **NEED_REFACTOR** | Shares active decoder; remove extension only after preserving base checkpoint load and metadata schema. |
| Weekly experiment/context | no `experiments.yaml` learned experiment; weekly-lag and daily-weekly blend are hardcoded in `forecastability_audit.py`, audit JSON/MD, innovation ledger generator | **STILL_REFERENCED** | It is audit evidence rather than model code. If frozen outputs remain, generation logic could later be removed or generalized, but doing so breaks audit reproducibility. |

None of these paths is presently an unconditional `SAFE_DELETE` implementation. `recover_multiscale_result.py` is the only direct `SAFE_DELETE` code candidate related to them; generated rejected result JSONs remain historical evidence and are Git-recoverable.

## 4. `experiments.yaml` classification

Every configured experiment is classified exactly once below. Classification describes present research use, not whether a result exists.

### ACTIVE_MAINLINE

- `centralized`
- `spatial_physical_only`
- `spatial_functional_only`
- `spatial_dual_graph` — configuration duplicate of dual base; consolidate name before future runs
- `pa_residual_anchor_dev`
- `pa_residual_scale_loss_dev`
- `pa_horizon_decoder_scale_dev`
- `pa_horizon_decoder_wl1_dev` — current node-WAPE development candidate

### ACTIVE_FAIRNESS

- `baseline`
- `load_lstm_validation`
- `itransformer_validation`
- `agcrn`
- `gwnet`

Missing fairness control: **GWN + the same WL1 objective** is required by the innovation ledger but is not yet configured. This audit does not add it.

### ACTIVE_FEDERATED

- `fedavg`
- `fedprox_005`
- `personalized_head`
- `local_only`
- `modulelocal_dev`
- `layerala_dev`
- `moduleala_dev`
- `dp_fedavg` — active conditional privacy baseline, no current result JSON

### HISTORICAL_REJECTED

- `pa_residual_multilevel_loss_dev`
- `pa_residual_multilevel_l002_dev`
- `pa_residual_multilevel_l005_dev`
- `pa_multilevel_tcn_transformer_dev`
- `pa_dynamic_functional_scale_dev`
- `pa_multiscale_patch_scale_dev`
- `pa_horizon_timequery_scale_dev`
- `pa_horizon_specific_head_scale_dev`

### AUXILIARY_AUDIT

- `stage1_smoke_centralized`, `stage1_smoke_fedavg`
- `stage_ala_smoke`, `stage_modulelocal_smoke`, `stage_fedala_smoke`
- `local_only_r40`, `fedavg_r40`, `fedprox_005_r40`, `personalized_head_r40`
- `calendar_pa_screen`, `calendar_gwnet_screen`, `calendar_lstm_screen`
- `centralized_test`, `fedavg_test`, `personalized_head_test` — dormant final-test lock configurations; must remain disabled until explicit authorization

## 5. Documentation consistency

### README.md

Accurate: official SMART-DS scope, 273 nodes, 216 Lines + 56 Transformers, 272 physical edges, one component, 92 targets, 15-minute/35,040 points, 96-to-12 task, no MST physical topology, basic validation/dry-run commands.

Outdated:

- “正式实验尚未基于新数据重新开始” is false. The repository contains centralized three-seed validation, Stage-3 federated validation/development, forecastability audits and an innovation-1 development ledger.
- “canonical SmartDS 数据验证、负荷来源追溯和图设计确认阶段” understates that topology-kNN k=6 and the 8-client partition were already frozen and development screening has progressed to a current candidate.
- Model section does not mention current innovation-1 candidate components (residual anchor, Horizon Decoder, WL1) or the fact that these remain validation-only development decisions.
- Run section says formal training must wait for data/graph freeze, although that freeze already happened. It should instead state that final multi-seed confirmation and test remain locked.

### PROJECT_STATUS.md

Accurate: canonical timestamp caveat, temperature unavailable, 5D final input semantics, ModuleALA implementation/runtime audit, topology/partition stability, no external weather substitution.

Outdated:

- Current Stage still says ModuleALA 2-round performance smoke; it omits the completed centralized development screens, forecastability audit and innovation-1 ledger.
- Model Status says only 2-round ModuleALA/VanillaFedALA runtime smoke, while `modulelocal_dev`, `layerala_dev`, and `moduleala_dev` result files also exist. Their research status should be distinguished from smoke.
- Next Steps focus on calendar/personalization exclusions and omit the current required fairness control (GWN + same WL1), subsequent multi-seed confirmation, and final test lock.
- Latest Commit is not a commit hash and still emphasizes the pre-cleanup backup. It should report the current commit/hash when status is next updated.
- It does not identify `pa_horizon_decoder_wl1_dev` as current node-level development candidate or state its RMSE/feeder-WAPE trade-off.

## 6. Proposed cleanup plan (not executed)

The ordering matters. Git history can restore tracked code/JSON, but it cannot restore ignored `.pt` files. Do not start with checkpoint deletion.

### DELETE

1. **Ignored caches:** `code/__pycache__/`, `preprocess/__pycache__/` and `.pyc`. Reason: reproducible runtime cache, already ignored. Impact: none beyond recompilation.
2. **One-time recovery utility:** `code/recover_multiscale_result.py`, after verifying its result JSON schema/checksum. Reason: rejected experiment recovery is complete. Impact: future recovery would require checking out the script from Git history.
3. **Duplicate/transient local checkpoint:** `results/pa_horizon_decoder_scale_dev_smoke_seed2026_centralized_model.pt`, after confirming it is not the formal best checkpoint. Reason: smoke duplicate. Impact: cannot be restored from Git; only delete after explicit approval.
4. **Legacy MST CSVs from active results surface:** `bridge_edges.csv`, `legacy_bridge_edges.csv`, `topology_comparison.csv` may be deleted after canonical evidence is confirmed in reports/Git history. Reason: they are not active physical topology inputs. Impact: direct historical comparison files disappear from HEAD but remain in Git history.
5. **Rejected experiment configs/results/code only as an atomic later change:** do not delete pieces independently. Tracked rejected JSONs are small historical evidence; keeping them is cheaper and safer than keeping rejected runtime branches.
6. **Do not bulk-delete `.pt`:** 50 ignored checkpoints (90.9 MB) are not Git-recoverable. Define a retention policy first: active mainline/fairness/federated checkpoints versus rejected/smoke checkpoints.

### MOVE

1. Move `results/forecastability_audit.json` to `reports/supporting/forecastability_audit.json`, then update both audit/ledger scripts and docs. Reason: it is analysis evidence, not a training result; user-facing references already expect it under reports. Impact: path migration must be atomic.
2. Move `results/federated_parameter_groups.csv` to `reports/supporting/`. Reason: parameter-group audit evidence. Impact: update any manual documentation links.
3. Move legacy topology CSVs to `reports/supporting/legacy/` **only if** history-in-HEAD remains desired; otherwise prefer deletion using Git history. Do not create backup/archive copies solely for cleanup.
4. Move standalone reporting scripts (`forecastability_audit.py`, `analyze_results.py`, `build_innovation1_ledger.py`) out of the training-core surface only if a simple non-nested location is agreed. Reason: distinguish training runtime from reporting. Impact: imports/README commands and path constants must change. Avoid introducing a complex package tree.

### REFACTOR

1. Split `run.py` along existing responsibilities while keeping `python code/run.py ...` as the stable entry point: training/evaluation, federated orchestration, and result-ledger helpers. Highest payoff and highest regression risk.
2. Create one resolved-config/result metadata builder shared by centralized, federated, dry-run, recovery and analysis. This prevents divergent result schemas.
3. Refactor `forecastability_audit.py` to a declarative method/checkpoint registry. Adding a method should not require repeated load, metric, delta, rendering and hardcoded filename blocks.
4. Reduce `model.py` to active components: preserve the base Horizon Decoder and main PA path; remove TCN, dynamic functional, patch, future-phase and specific-head extensions only after config/report dependencies are detached and checkpoint loading is tested.
5. Separate legacy graph diagnostics from `data.py`/`run.py:audit`. Canonical training should expose only `topology_knn`; historical MST/projected modes can remain recoverable through Git history or a dedicated audit script.
6. Consolidate `centralized` and `spatial_dual_graph` naming to prevent duplicated equivalent runs.
7. Add explicit experiment status metadata to `experiments.yaml` (`ACTIVE_MAINLINE`, `ACTIVE_FAIRNESS`, etc.) and make `run_all` select by status rather than name heuristics. This removes duplicated classification from report scripts.
8. Update README and PROJECT_STATUS in a later, separate commit to reflect Stage-1 development reality, current WL1 candidate, fairness gap, and locked test status.
9. Stabilize preprocess imports: expose official raw parser/scope helpers as public functions rather than importing `_scope_files`; keep one-way dependency from build/validate scripts to parser utilities.
10. Add lightweight import/config/checkpoint-compatibility tests before and after cleanup. Required checks: active PA and GWN checkpoint load, active Horizon Decoder parameter names, topology contract, development `test_evaluated=false`, dry-run experiment resolution, and no creation of test loaders.

## 7. Recommended execution order

1. Update README/PROJECT_STATUS and add experiment status metadata.
2. Freeze the innovation ledger and forecastability artifact paths.
3. Refactor reporting registry and result metadata builder.
4. Move audit artifacts/scripts.
5. Remove one-time recovery and ignored caches.
6. Detach rejected configs/reports, then delete rejected model branches atomically.
7. Split `run.py`/legacy graph audit with checkpoint and dry-run regression checks.
8. Decide local checkpoint retention last; never assume Git can restore ignored `.pt` files.

This audit authorizes no cleanup action. Every listed DELETE/MOVE/REFACTOR item remains a proposal for a later explicit task.
