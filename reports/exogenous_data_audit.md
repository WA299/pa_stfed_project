# SMART-DS timestamp 与 temperature 对齐审计

**总体结论：FAIL**

本审计只核对官方外生数据来源及逐点时序，不修改模型、不训练，也不通过插值、平移或时区猜测强行对齐。

## Timestamp

- 官方来源：当前 feeder 引用的 `61` 条 kW profile 及其一一对应的 SMART-DS `load_data` parquet。
- 时间范围：`2018-01-01T00:15:00` 至 `2019-01-01T00:00:00`，语义为 15 分钟区间末时间，原文件未携带时区。
- 点数：35040；重复：0；非 15 分钟间隔：0；缺口：0。
- 61 个 parquet 的 timestamp 完全一致：`True`。
- profile/parquet 数值归一化逐点一致：58/61；不一致文件：`com_kw_37592-South-Central_pu.csv, com_kw_37875-South-Central_pu.csv, com_kw_38868-South-Central_pu.csv`。
- canonical `load_ts` 可由官方 profile 与 Loads.dss 映射逐元素重建：`True`。
- Timestamp 结论：`FAIL`。

已从该 timestamp 确定性生成 `hour_of_day`、`day_of_week`（Monday=0）、`weekend` 和 `month`；哈希记录在 supporting JSON 中。

## Temperature

SMART-DS 用户指南说明通用 `solar_data` 文件可含 NSRDB Temperature，但当前发布范围必须单独核实。审计官方公开 S3 后发现：

- P10R 前缀对象数：3861；温度/solar/weather/NSRDB/PVSystems 匹配对象数：0。
- `Full_Texas/solar_data/`、`P10R/solar_data/` 与 `P10R/scenarios/base_timeseries/solar_data/` 的对象数均为 0。
- 现有官方 `load_data` parquet 列中不含 temperature。
- 无法从当前 v0.9 / 2018 / Full_Texas / P10R 数据确定 feeder 对应的 NSRDB 地点或温度序列。
- Temperature 结论：`FAIL`；min/mean/max/NaN 均不可计算，记为 `null`，不得用其他地区或自行插值序列替代。

## Canonical 输出

由于 temperature 未通过来源与对齐审计，未生成扩展 canonical NPZ。原 `smartds_full_graph_v2.npz` 保持不变。

## 证据边界

- 61 个 parquet 自身均给出同一组连续 timestamp，但 3 条商业 profile 与同名 parquet 总负荷不满足逐点归一化相等；因此不能证明 canonical 中这 3 条 profile 与 timestamp 的逐点关系，本阶段严格判 FAIL。
- canonical `load_ts` 与 61 条官方 profile 及 Loads.dss 映射的 float32 逐元素重建完全一致；这证明 canonical 构造正确，但不能替代缺失的 profile-to-timestamp 证据。
- 用户指南对 `solar_data` 的一般结构描述不能证明 Full_Texas/P10R 实际发布了该文件；本审计以指定版本官方对象清单为准。
- 若以后获得带 P10R 地点映射的官方 temperature，必须重新运行本脚本，不能沿用本次 FAIL 结果推定对齐。
