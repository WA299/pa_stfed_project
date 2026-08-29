# 本阶段文件清单与关键变更

## 新建文件

| 文件 | 用途 |
|---|---|
| `data/legacy/smartds_graph_legacy.npz` | 旧 NPZ 只读封存副本，SHA256 固定 |
| `data/raw/SMARTDS/...` | OEDI SMART-DS v1.0 官方探针文件；仅用于来源/解析验证，未认定为当前数据来源 |
| `data/processed/smartds_metadata_v2.json` | v2 构建状态；当前为 blocker，不含 canonical 图 |
| `preprocess/audit_smartds_data.py` | 旧数据数组、官方 OpenDSS 证据、节点角色、时间序列和拓扑审计 |
| `preprocess/build_smartds_graph_v2.py` | 证据满足条件时构建 full physical graph 与 target projection；否则拒绝构建 |
| `reports/audit_report.json` | 机器可读审计结果 |
| `reports/data_provenance.md` | 来源、版本、时间范围和不确定性记录 |
| `reports/rdt_rdtlv_verification.csv` | 56 对 `rdt-rdtlv` 关系逐对验证 |
| `reports/node_role_audit.csv` | 273 个节点角色初筛 |
| `reports/topology_comparison.csv` | legacy 图变体诊断统计，均标记为非 canonical |
| `reports/v2_compatibility.md` | v2 与 PA-STFed 数据/联邦接口兼容性、后续实验闸门 |
| `MANIFEST.json` | legacy 与官方探针文件 SHA256、URL、状态 |

## 关键差异

1. 旧工程文件、`outputs/`、checkpoint、日志和配置没有覆盖或删除。
2. 旧 `adj` 只作为 legacy 统计；Euclidean MST 不再具有 physical graph 资格。
3. OpenDSS 解析保留 `line`/`transformer` 的设备类型、来源文件和行号；重复 `bus=` winding 也会被解析。
4. `rdt-rdtlv` 只有在官方 Transformer 记录中找到端点证据才会标记 `verified=true`；当前 56 对均为 false。
5. 181 个零负荷节点不插值；角色报告明确区分“未验证”与“结构母线候选”。
6. v2 构建脚本对节点唯一性、边界、边证据、邻接一致性、映射可逆性、目标负荷形状和 NaN/Inf 执行断言。
7. 当前官方探针是 `2018/SFO/P10U`，与 legacy `p10rdt/p10rlv` 节点 0 匹配，故处理状态为 `BLOCKED_NO_VERIFIED_CANONICAL_SOURCE`。

## 2026-08-29 数据源核验更新

1. 按官方 OEDI `SMART-DS/v0.9/2018/Full_Texas/P10R` 目录锁定 `p10rhs0_1247--p10rdt7719` feeder，并将父级 substation 与目标 feeder 设为显式 scope。
2. 273/273 legacy 节点、92/92 Load 母线和 56/56 Transformer 关系均获得官方 OpenDSS 文件/行号证据。
3. 下载 `Loads.dss`/`LoadShapes.dss` 实际引用的 122 个官方 profile，并生成 `PROFILE_MANIFEST.json`。
4. canonical 状态更新为 `RAW_SOURCE_VERIFIED_CANONICAL`；full graph 为 273 节点、272 条官方设备边、1 个连通分量，不含 MST。
5. 新增 `validate_smartds_v2.py`，对节点、边界、来源证据、映射、数值完整性和投影统计执行独立一致性测试。
