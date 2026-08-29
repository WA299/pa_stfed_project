# SmartDS 数据审计决策

legacy 审计状态：`BLOCKED_LOAD_SERIES_MAPPING_UNVERIFIED`  
本文件只记录数据来源和一致性事实，不包含模型训练或论文性能结论。

当前已另外生成一份不依赖 legacy 数值的官方序列候选：
`data/processed/official_v1_2018_P10R_p10rdt7719/`。该候选直接按
`Loads.dss kW * LoadShapes.dss mult` 重建，独立一致性验证为 `PASS`；
它与 legacy NPZ 必须作为两条不同的数据血缘处理。

## 结论清单

1. **官方来源**：Open Energy Data Initiative（OEDI）SMART-DS，官方对象键为 `SMART-DS/v0.9/2018/Full_Texas/P10R/scenarios/base_timeseries/`，scope 为 substation `p10rhs0_1247` 下的 feeder `p10rhs0_1247--p10rdt7719`。版本信息来自官方对象键和下载清单；当前 OpenDSS 文件本身没有独立版本字段。旧 NPZ 的生成脚本和原始下载记录仍未找到，故只能确认其节点/拓扑与该 scope 相匹配，不能宣称其负荷数值直接来自官方 timeseries。
2. **时间长度**：legacy `load_ts` 有 35,040 行、无 NaN/Inf、无重复整行，按 15 min 仅能说明索引长度等于 365 天，不能从文件本身确认绝对日期。官方 61 个有功 profile 与 61 个 parquet 均为 35,040 行，parquet 时间列通过连续 15 min 审计。
3. **92 个目标节点**：92/92 个非零 legacy 节点可与官方 `Loads.dss` 母线匹配，且每个母线均有官方 Load 元件证据；但逐点数值映射为 0/92 精确通过，不能把列值称为官方重建值。
4. **181 个零负荷节点**：它们在 legacy 序列中全年为零，且官方 scope 内未发现对应 Load 元件；结合 degree 和设备端点记录，当前应作为结构母线候选保留，不插值、不作为预测目标。若后续需要更细角色，依据 `node_role_audit.csv` 的设备证据逐节点复核。
5. **57 个 legacy 分量**：旧邻接有 216 条无向边和 57 个分量；官方完整 Line+Transformer 图为 1 个分量。因此 57 个分量主要由旧预处理漏读 56 条 Transformer connectivity 造成，而非官方网络真实断连，也不是删除零负荷节点产生的 canonical 事实。
6. **56 对 rdt/rdtlv**：`p10rdtXXXXX` 与 `p10rdtXXXXXlv` 的 56/56 对均在官方 `Transformers.dss` 中逐对找到设备名、文件和行号证据，`verified=true`。
7. **Transformer 遗漏**：是。旧 NPZ 的 216 条边与官方 Line 边数量相符，但没有把 56 条官方 Transformer 边写入旧邻接。
8. **官方完整拓扑**：273 个节点、216 条 Line 无向边、56 条 Transformer 无向边，共 272 条无向边，1 个连通分量；每条 canonical 边均保留 `edge_type` 和 `edge_source=file:line`。坐标距离和 hop 只能作位置/拓扑先验，不能解释为导线长度或电气阻抗。
9. **MST 是否需要**：不需要。官方 Line+Transformer 已连通；欧氏 MST、名称补边和距离推断边不得进入 canonical physical graph，只能作为历史敏感性对照。
10. **后续图选择**：优先 full official physical graph，并保留零负荷结构节点作为消息传播中继，只在 92 个 target 节点计算 loss。按“路径内部不含第三个 target”得到的 target projection 有 4,186 条无向边、密度 1.0、平均度 91，过密，不宜直接作为稀疏空间图。
11. **PA-STFed 适配**：loader 需读取 full graph 的 `edge_index/edge_type/edge_source`、`target_mask`、`load_mask` 及 full-target 映射；物理注意力接收官方边属性；loss 只索引 target 节点；旧 `hop_radius`/MST 逻辑降级为 legacy 诊断。FedAvg/FedProx/Personalized 代码与数据来源解耦，但在数据闸门解除前不得运行。
12. **未决事项**：legacy `load_ts` 的生成脚本、确切单位/缩放和列值映射尚未确认；官方 OpenDSS-native 与 parquet-native 口径内部总体高度一致，但少数商业 profile 存在最大约 0.86 kW 的 profile/parquet 差异，需在正式使用前记录解释。当前 provisional NPZ 已隔离，验证脚本会拒绝其进入训练。

13. **官方序列候选**：节点顺序由 feeder `Buscoords.dss` 独立确定；官方重建产物为 273 节点、272 条无向边（Line=216、Transformer=56）、1 个连通分量、35040 个时间点、92 个目标节点。另有 1 条指向相邻 feeder 的禁用边界开关记录，仅保留在 metadata 证据中。`Intermediates.txt` 的 149 条记录只作为线段几何证据。
14. **远程来源核验**：当前 scope 的 15 个 OpenDSS 文件与 OEDI `opendss/` 官方对象逐项比较 SHA256 和字节数，15/15 通过；机器可读证据为 `reports/official_raw_remote_verification.json`。

## 闸门

旧 NPZ 只有在官方数值序列逐点映射通过（或获得可信的 legacy 生成证据并解释差异）后才能解除 legacy blocker。官方独立候选已通过自身数据闸门，但在本任务验收前仍禁止 FedAvg、FedProx、Personalized、MST 消融和任何用于论文结论的训练。
