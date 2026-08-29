# 数据重建变更记录

## 2026-08-29

1. 将旧 `smartds_graph.npz` 封存为 `data/legacy/smartds_graph_legacy.npz`，保留原始 SHA256 和所有旧输出。
2. 按官方 SMART-DS 用户指南及 OEDI 对象存储追溯到 `v0.9/2018/Full_Texas/P10R/base_timeseries` 的 `p10rhs0_1247--p10rdt7719` feeder。
3. 审计范围固定为目标 feeder 与父级 `p10rhs0_1247`，不再递归混入其它 feeder。
4. 273/273 节点、92/92 Load 母线和 56/56 `rdt-rdtlv` Transformer 关系获得 OpenDSS 文件与行号证据。
5. 下载 `Loads.dss`/`LoadShapes.dss` 实际引用的 122 个官方 profile，写入 `PROFILE_MANIFEST.json`。
6. 生成 canonical full graph：273 节点、216 条 Line 边、56 条 Transformer 边、1 个连通分量；不含欧氏 MST、kNN 或名称补边。
7. 生成 target graph，并将 92 节点拓扑投影过密诊断（4186 条无向边、密度 1.0）写入 metadata。
8. 新增 `validate_smartds_v2.py`，独立验证节点唯一、边界、邻接一致、官方来源、映射可逆、数值完整和无 MST 等条件。
9. 按官方文档纳入 `Intermediates.txt`：149 条线段几何记录作为证据保存，不创建额外物理边或节点；修正 provisional 文件检查字段的语义。
10. 新增 `build_official_load_series.py`：不读取 legacy 负荷数值，独立生成官方 OpenDSS-native canonical 候选，并完成 deterministic rerun（full/target NPZ SHA256 一致）。
11. 新增 `verify_official_raw.py`：对当前 scope 的 15 个 OpenDSS 文件逐一请求官方 OEDI 对象并比较本地/远程 SHA256 与文件大小，结果全部通过。
12. 加强 `validate_smartds_v2.py`：新增 `targets_have_nonzero_series` 断言，明确拒绝把全年零序列结构节点误纳入 target；官方候选复跑结果仍为 `PASS`。

旧版 blocker 记录保存在 `change_log_legacy.md`，仅供审计历史查阅。
