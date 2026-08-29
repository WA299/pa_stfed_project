# SmartDS Data Audit

## Canonical Scope

当前正式候选数据来自 Open Energy Data Initiative 发布的 SMART-DS v0.9：

- 年份与数据集：2018 Full_Texas
- 区域与场景：P10R / `base_timeseries`
- 变电站：`p10rhs0_1247`
- feeder：`p10rhs0_1247--p10rdt7719`
- 采样间隔与长度：15 分钟，共 35,040 个时间步

来源和文件哈希记录见 `data/raw/SMARTDS/` 下的 manifest；逐项审计证据见 `reports/supporting/`。

## Physical Topology

官方 feeder 范围内的 `Lines.dss` 与 `Transformers.dss` 重建结果为：273 个节点、216 条 Line 边、56 条 Transformer 边、272 条无向边和 1 个连通分量。每条 canonical 边均携带官方设备类型和源文件行号；物理图中不包含欧氏 MST 或其他启发式补边。

旧版 `smartds_graph.npz` 的 57 个连通分量不是 SMART-DS feeder 的真实结构。旧预处理遗漏了 56 条官方 Transformer 边，因而将一棵完整连通树拆成 57 个分量。历史 MST 文件和旧实验结果已退出当前工程，仍可从 `backup/pre-cleanup` 分支追溯。

## Node Roles

273 个节点中有 92 个节点拥有全年非零负荷序列并作为预测目标。其余 181 个节点是无预测负荷的结构或中继母线；它们保留在物理拓扑中，不插补负荷，也不进入预测损失。节点级证据见 `reports/supporting/node_role_audit.csv`。

## Load Reconstruction

canonical 负荷按官方 OpenDSS 定义，由每个母线上的 `Loads.dss` 额定 kW 与其引用的 `LoadShapes.dss` multiplier 逐时相乘并汇总。完整序列保存在 `smartds_full_graph_v2.npz` 的 `load_ts` 字段，形状为 `[35040, 273]`；`target_mask` 标记 92 个预测节点。

旧 NPZ 的 `load_ts` 与该官方重建序列存在数值不一致，审计状态为 `BLOCKED_VALUES_MISMATCH`。因此旧 NPZ 及基于它产生的训练结果不能作为当前 ground truth 或论文证据。

## Canonical Artifacts

- `data/processed/smartds_full_graph_v2.npz`
- `data/processed/smartds_metadata_v2.json`
- `reports/supporting/validation_report.json`

full NPZ 的 SHA-256 为 `7997e0882f6f35b2c8af4bcdeeb43e011bcf9d45adfefd25f1a840c3151b940a`。

## Remaining Issues

原始 OpenDSS 和 profile 大文件按要求不保留在 Git 仓库，当前验证器默认执行 canonical artifact 的结构、哈希和元数据一致性检查；若重新下载 raw 数据，可再执行逐边源文件复核。另一个尚未冻结的问题是：模型如何在保留 181 个结构节点拓扑作用的同时，仅对 92 个负荷节点进行编码与预测。该问题解决前不启动正式训练。
