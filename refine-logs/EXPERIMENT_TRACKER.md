# PA-STFed 实验追踪表

状态仅使用 `TODO`、`RUNNING`、`DONE`、`BLOCKED`、`ARCHIVED`。旧版
`code_revision=20260826_literature_baselines_v4` 的结果统一标记为 `ARCHIVED`，
不得和 `20260829_topology_projection_v2` 混合汇总。

| Run ID | 里程碑 | 目的 | 系统/变体 | Split | 指标 | 优先级 | 状态 | 备注 |
|---|---|---|---|---|---|---|---|---|
| R000 | M0 | 数据与拓扑审计 | `audit` | n/a | 形状、组件、边数、跳数、Non-IID | MUST | DONE | `outputs/audit_report.json` 已更新 |
| R001 | M0 | 参数共享审计 | audit 输出 | n/a | state 分组、源码位置 | MUST | DONE | `outputs/federated_parameter_groups.csv` |
| R002 | M0 | 拓扑烟测 | `topology_forest` | val | WAPE/RMSE/MAE/sMAPE/MAPE | MUST | TODO | seed 2026 |
| R003 | M0 | 拓扑烟测 | `topology_mst_no_tag` | val | 同上 | MUST | TODO | seed 2026 |
| R004 | M0 | 拓扑烟测 | `topology_mst_tag` | val | 同上 | MUST | TODO | seed 2026 |
| R005 | M0 | 双图烟测 | `spatial_physical_only` | val | 同上 | MUST | TODO | seed 2026 |
| R006 | M0 | 双图烟测 | `spatial_functional_only` | val | 同上 | MUST | TODO | seed 2026 |
| R007 | M0 | 双图烟测 | `spatial_dual_graph` | val | 同上、门控诊断 | MUST | TODO | seed 2026 |
| R008 | M0 | 联邦烟测 | `local_only` | val | 客户端宏平均、std、P90/P95 | MUST | TODO | seed 2026 |
| R009 | M0 | 联邦烟测 | `fedavg` | val | 同上 | MUST | TODO | seed 2026 |
| R010 | M0 | 联邦烟测 | `fedprox_005` | val | 同上 | MUST | TODO | seed 2026 |
| R011 | M0 | 联邦烟测 | `personalized_head` | val | 同上 | MUST | TODO | seed 2026 |
| R012 | M1 | 时间序列基线 | `load_lstm_validation` | val | 集中式五指标 | MUST | TODO | seeds 2026--2028 |
| R013 | M1 | 时间序列基线 | `itransformer_validation` | val | 集中式五指标 | MUST | TODO | seeds 2026--2028 |
| R014 | M1 | 图基线 | `agcrn` | val | 集中式五指标 | MUST | TODO | adapted implementation |
| R015 | M1 | 图基线 | `gwnet` | val | 集中式五指标 | MUST | TODO | adapted implementation |
| R016 | M2 | 拓扑生死实验 | `topology_forest` | val | 块 Bootstrap + 客户端/节点 | MUST | TODO | seeds 2026--2028 |
| R017 | M2 | 拓扑生死实验 | `topology_mst_no_tag` | val | 同上 | MUST | TODO | seeds 2026--2028 |
| R018 | M2 | 拓扑生死实验 | `topology_mst_tag` | val | 同上 | MUST | TODO | seeds 2026--2028 |
| R019 | M3 | 双图生死实验 | `spatial_physical_only` | val | 块 Bootstrap + CKA/门控 | MUST | TODO | seeds 2026--2028 |
| R020 | M3 | 双图生死实验 | `spatial_functional_only` | val | 同上 | MUST | TODO | seeds 2026--2028 |
| R021 | M3 | 双图生死实验 | `spatial_dual_graph` | val | 同上 | MUST | TODO | seeds 2026--2028 |
| R022 | M4 | 联邦协同 | `local_only` | val | 宏平均、std、P90/P95、最差客户端 | MUST | TODO | seeds 2026--2028 |
| R023 | M4 | 联邦协同 | `fedavg` | val | 同上 | MUST | TODO | seeds 2026--2028 |
| R024 | M4 | Non-IID 稳定性 | `fedprox_005` | val | 同上、聚合审计 | MUST | TODO | seeds 2026--2028 |
| R025 | M4 | 个性化适配 | `personalized_head` | val | 同上、共享/本地参数 | MUST | TODO | seeds 2026--2028 |
| R026 | M5 | 统计检验 | `analyze_results.py` | val/test | WAPE 配对区块 Bootstrap | MUST | TODO | 仅比较同一窗口块 |
| R027 | M6 | 最终测试 | `centralized_test` | test | 五指标、节点级 | MUST | TODO | 锁参后一次 |
| R028 | M6 | 最终测试 | `fedavg_test` | test | 客户端尾部指标 | MUST | TODO | 锁参后一次 |
| R029 | M6 | 最终测试 | `personalized_head_test` | test | 客户端尾部指标 | NICE | TODO | 仅当验证阶段有必要 |
| R030 | B6 | 隐私附录 | `dp_fedavg` | val | 六元组、epsilon、精度折损 | NICE | TODO | 不宣称严格隐私 |

## 记录规则

- 每次运行保存完整命令、终端日志、配置指纹和退出码。
- `--resume` 只能跳过同一 `config_signature` 且 JSON 可解析的结果。
- 任何失败任务先记录错误并修复，再从相同种子重跑；不要删除失败痕迹。
- 只有 R027--R029 完成后才允许在论文摘要中写最终测试数值。
