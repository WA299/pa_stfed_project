# SMART-DS timestamp 与 temperature 对齐审计

**总体结论：PASS_WITH_TEMPERATURE_UNAVAILABLE**

本审计只核对官方外生数据来源及逐点时序，不修改模型、不训练，也不通过插值、平移或时区猜测强行对齐。

## Timestamp

- 官方来源：当前 feeder 引用的 `61` 条 kW profile 及其一一对应的 SMART-DS `load_data` parquet。
- 时间范围：`2018-01-01T00:15:00` 至 `2019-01-01T00:00:00`，语义为 15 分钟区间末时间，原文件未携带时区。
- 点数：35040；重复：0；非 15 分钟间隔：0；缺口：0。
- 61 个 parquet 的 timestamp 完全一致：`True`。
- profile/parquet 数值归一化逐点一致：58/61；不一致文件：`com_kw_37592-South-Central_pu.csv, com_kw_37875-South-Central_pu.csv, com_kw_38868-South-Central_pu.csv`。
- canonical `load_ts` 可由官方 profile 与 Loads.dss 映射逐元素重建：`True`。
- Timestamp 结论：`PASS for common calendar indexing, with profile-value caveat`。

已从该 timestamp 确定性生成 `hour_of_day`、`day_of_week`（Monday=0）、`weekend` 和 `month`；哈希记录在 supporting JSON 中。

## Temperature

SMART-DS 用户指南说明通用 `solar_data` 文件可含 NSRDB Temperature，但当前发布范围必须单独核实。审计官方公开 S3 后发现：

- P10R 前缀对象数：3861；温度/solar/weather/NSRDB/PVSystems 匹配对象数：0。
- `Full_Texas/solar_data/`、`P10R/solar_data/` 与 `P10R/scenarios/base_timeseries/solar_data/` 的对象数均为 0。
- 现有官方 `load_data` parquet 列中不含 temperature。
- 无法从当前 v0.9 / 2018 / Full_Texas / P10R 数据确定 feeder 对应的 NSRDB 地点或温度序列。
- Temperature 结论：`UNAVAILABLE`；min/mean/max/NaN 均不可计算，记为 `null`，不再继续获取，也不得用其他地区或自行插值序列替代。

## Canonical 输出

已生成不含 temperature 的官方公共 calendar sidecar：`data/processed/smartds_calendar_v1.npz`（SHA256：`8ad5465b6139eefc003cfb1266d33b1b173402ab29a2e82a082fced35dc749b7`）。原 `smartds_full_graph_v2.npz` 保持不变，未生成扩展 canonical NPZ。

## 证据边界

- 61 个 parquet 自身均给出同一组连续 timestamp；3 条商业 profile 与同名 parquet 总负荷不满足逐点归一化相等，作为 profile-value caveat 保留在 supporting JSON，不改变共同 calendar indexing 结论。
- canonical `load_ts` 与 61 条官方 profile 及 Loads.dss 映射的 float32 逐元素重建完全一致。
- 用户指南对 `solar_data` 的一般结构描述不能证明 Full_Texas/P10R 实际发布了该文件；本审计以指定版本官方对象清单为准。
- 当前正式输入仅使用 historical load + calendar；temperature 不进入数据加载器。
