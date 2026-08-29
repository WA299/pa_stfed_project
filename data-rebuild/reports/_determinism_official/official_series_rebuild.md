# 官方 SMART-DS 负荷序列重建报告

状态：`RAW_SOURCE_VERIFIED_CANONICAL`（独立官方序列候选）

本报告不把 legacy `smartds_graph.npz` 的 `load_ts` 当作输入；旧序列与官方序列的映射审计仍为 `BLOCKED_VALUES_MISMATCH`，两者不得混用。

## 来源与公式

- 官方来源：OEDI SMART-DS v0.9 / 2018 / Full_Texas / P10R / base_timeseries。
- feeder：`p10rhs0_1247/p10rhs0_1247--p10rdt7719`；节点顺序来自 feeder `Buscoords.dss`，共 `273` 个。
- 物理边仅来自该 feeder 的 `Lines.dss` 与 `Transformers.dss`；`Intermediates.txt` 只保存线段中间坐标，不产生额外节点或边。
- 每个目标母线的序列为 `sum(Loads.dss kW * LoadShapes.dss mult)`；中心抽头 `_1/_2` 的 `kW` 已按官方说明拆分，不重复乘 `0.5`。
- profile 长度统一为 `35040`，采样间隔为 `15` 分钟；未引入天气、节假日或真实日期特征。

## 重建统计

- full graph：`273` 节点，`272` 条无向边，`1` 个连通分量。
- 边类型：Line `216`，Transformer `56`。
- 预测目标：`92` 个；零负荷结构节点：`181` 个，保留在 full graph 但不进入 loss。
- feeder 边界外记录：`1` 条，未写入当前 feeder 图，完整证据见 metadata。
- target projection：`4186` 条无向边，密度 `1.0000`；若过密，后续模型应优先使用 full graph。

## 审计边界

- SmartDS 是合成配电网；该产物可支持合成网络实验，不能外推为真实台区实测结果。
- 旧 NPZ 负荷列与官方序列无法逐点复现，不能通过自由缩放、平移或列重排解除 blocker。
- 本脚本只完成数据重建和一致性检查，不启动 PA-STFed、FedAvg、FedProx 或任何论文结论实验。
