# SmartDS 数据来源与拓扑审计

审计时间（UTC）：`2026-08-29T11:16:02.760195+00:00`  
状态：**BLOCKED_NO_VERIFIED_CANONICAL_SOURCE**

## 当前可确认事实

- legacy 文件：`E:\pa_stfed_project\data-rebuild\data\legacy\smartds_graph_legacy.npz`；SHA256：`06a8c1d583159c1e3e972cdf12250e665e93f7f425fc83db57c772aa95cdb82f`。
- 字段形状：`node_coords=(273, 2)`、`adj=(273, 273)`、`edge_index=(2, 432)`、`load_ts=(35040, 273)`、`node_ids=(273,)`。
- 节点数 `273`；全年存在非零序列的节点 `92`；全年为零的节点 `181`。
- 旧 `adj` 是 `216` 条无向边、`57` 个连通分量；该统计只描述二次处理 NPZ，不等同于官方 OpenDSS 拓扑。
- 序列长度 `35040`。按 15 分钟采样相当于 `365.00` 天，但文件不含可验证绝对 timestamp，不能写成具体年份或日期。

## 官方来源检索结果

官方候选来源为 OEDI 公共数据湖的 `s3://oedi-data-lake/SMART-DS/v1.0/`，其目录包含 2016--2018 年、AUS/GSO/SFO 区域、OpenDSS `Master.dss`、`Lines.dss`、`Transformers.dss`、`Loads.dss` 和 `Buscoords.dss`。本次审计只把实际放入 `data/raw/SMARTDS` 的文件计入证据。

对 OEDI 2016--2018 目录索引的核查显示：SFO 目录有 `P10U` 但没有 `P10R`，AUS 仅列出 `P1R` 等子区，GSO 使用 `rural/urban-suburban/industrial` 命名；这只能排除当前候选目录，不能替代对未知来源的完整检索。
- 当前 raw 探针发现 DSS 文件 `11` 个、设备连接记录 `1128` 条。
- 已下载探针属于 OEDI `2018/SFO/P10U/base_timeseries`，其中 feeder 母线前缀为 `p10udt`；当前 legacy 节点前缀为 `p10rdt`/`p10rlv`，按严格字符串映射匹配数为 0。该探针用于验证解析链，不是当前 NPZ 的来源认定。
- legacy 坐标范围为 x=[-99.15498,-99.13215]、y=[29.92752,29.99944]；候选探针坐标范围见 `audit_report.json`，坐标差异只能作为排除线索，不能用于猜测来源。
- 与 273 个 legacy 节点按官方母线名（大小写不敏感）匹配 `0` 个（比例 `0.000`）。
- canonical 构建要求 273/273 节点全部匹配且设备文件能提供来源证据；部分匹配一律 blocker。不得把 `p10rdtXXXXX ↔ p10rdtXXXXXlv` 写成 Transformer。

## 当前 blocker 与边界

1. 项目目录没有找到能够生成当前 NPZ 的原始 `.dss`、下载脚本或 metadata；当前 NPZ 的 `p10rdt` 命名不能单独证明 SMART-DS 子集。
2. 若 raw 目录为空或节点匹配不足，`smartds_graph_legacy.npz` 只能作为历史基线；本阶段不生成 canonical v2 图，也不启动正式训练。
3. 181 个零负荷节点暂不插值、不删除；只有官方 Load/Bus/Transformer/Line 证据齐全后，才区分结构母线、Transformer side bus、缺失映射和真实负荷。
4. 任何欧氏 MST、节点名规则或旧邻接矩阵都不能替代官方物理设备关系。

## 生成文件

- `audit_report.json`：机器可读审计结果。
- `node_role_audit.csv`：273 个节点角色初筛。
- `rdt_rdtlv_verification.csv`：56 对命名线索的逐对证据状态。
- `topology_comparison.csv`：旧图与中继投影的诊断统计（均非 canonical）。
- `data_provenance.md`：本报告。
