# SmartDS 数据来源与拓扑审计

审计时间（UTC）：`2026-08-29T12:59:49.436471+00:00`  
状态：**BLOCKED_LOAD_SERIES_MAPPING_UNVERIFIED**

## 1. 数据来源（证据等级 A）

- 提供方：Open Energy Data Initiative（OEDI）SMART-DS。
- 版本/年份/数据集：`SMART-DS v0.9 / 2018 / Full_Texas`。
- 区域：`P10R`；场景：`base_timeseries`。
- substation：`p10rhs0_1247`；feeder：`p10rhs0_1247--p10rdt7719`。
- 官方对象存储前缀：`s3://oedi-data-lake/SMART-DS/v0.9/2018/Full_Texas/P10R/scenarios/base_timeseries/`。
- 本次严格 scope：`p10rhs0_1247/p10rhs0_1247--p10rdt7719` 及父级 `p10rhs0_1247`，解析文件 `14` 个；未把其它 feeder 递归纳入。
- 节点命名仅用于与官方母线 ID 做大小写不敏感的逐项匹配；设备关系均来自 OpenDSS `Lines.dss`/`Transformers.dss`，不由名称或距离推断。

## 2. Legacy NPZ 审计（证据等级 A）

- 文件：`E:\pa_stfed_project\data-rebuild\data\legacy\smartds_graph_legacy.npz`；SHA256：`06a8c1d583159c1e3e972cdf12250e665e93f7f425fc83db57c772aa95cdb82f`。
- 字段：`node_coords=(273, 2)`、`adj=(273, 273)`、`edge_index=(2, 432)`、`load_ts=(35040, 273)`、`node_ids=(273,)`。
- `273` 个节点，其中 `92` 个全年非零、`181` 个全年为零；旧邻接为 `216` 条无向边、`57` 个分量。
- 序列长度 `35040`；按 15 分钟采样为 `365.00` 天。文件无可验证绝对 timestamp，因此不得外推日期、星期、节假日或天气。
- NaN/Inf=`0/0`；非零节点中间零值节点数 `0`。单位和缩放未在 NPZ 元数据中标注。

## 3. 官方拓扑核查（证据等级 A）

- 官方 OpenDSS 端点记录 `317` 条；legacy 节点匹配 `273/273`。
- 完整官方图：`273` 节点、`272` 条无向边、`1` 个连通分量；其中 line=`216`、transformer=`56`。
- 56 对 `p10rdtXXXXX ↔ p10rdtXXXXXlv`：官方 Transformer 逐对验证 `56/56`；因此旧 NPZ 确实遗漏了这 56 条 Transformer connectivity。
- 官方图最大度 `5`、平均度 `1.993`；旧 57 个分量主要由缺失 Transformer 边造成，而非删除零负荷节点后的真实断连。

## 4. 节点角色与负荷曲线（证据等级 A/B）

- `92` 个非零目标节点均有官方 Load 设备证据；`181` 个全年零序列节点被识别为结构母线候选。结构节点的 0 不做插值。
- 官方 LoadShape 数 `61`，profile 文件 `122` 个；本地 profile 完整存在：`True`。每个 shape 的 `npts`、`interval` 与来源行记录在 `audit_report.json`。
- 官方 OpenDSS-native 重建定义为按母线汇总 `Loads.dss` 全部 Load 元件的 `kW × mult`；中心抽头 `_1/_2` 的 `kW` 已经是拆分后的半客户峰值，不能再次乘 0.5。若直接读取整户 parquet，则每个 `_1/_2` 元件才乘 0.5。
- 本次逐点核对状态为 `BLOCKED_VALUES_MISMATCH`，匹配 `92/92` 个母线，精确通过 `0` 个。
- 最大绝对误差 `21.914859605922985` kW、平均相对误差 `0.6629356733995491`；因此当前 NPZ 的负荷数值映射未通过，不能把现有序列宣称为官方 profile 的直接重建结果。
- parquet 独立审计状态 `NOT_RUN`；报告 `reports/official_load_data_audit.json`。官方文件内部一致性与 legacy 对比结果均以该机器可读报告为准。

- 在映射状态为 `PASS_EXACT` 前，legacy `load_ts` 只能作为待解释的历史输入；禁止用自由缩放、时间平移或列重排把它强行拟合到官方 profile/parquet。

## 5. 约束与下一步

1. canonical physical graph 只允许官方 Line/Transformer 等明确设备关系；Euclidean MST、节点名补边和旧 adj 仅作 legacy 敏感性对照。
2. 推荐后续模型保留 181 个零负荷结构节点作为 message-passing relay，仅在 92 个 target 节点计算损失；target projection 图若过密，不直接作为稀疏图。
3. 本报告不包含任何正式模型训练、FedAvg/FedProx 或消融结论；联邦客户端仍是合成网络空间划分模拟。
4. 若 profile 单位、原始生成脚本或设备元件映射出现冲突，应暂停模型实验并在本报告中记录 blocker。

## 6. 审计产物

- `audit_report.json`：机器可读字段、scope、拓扑和时间序列统计。
- `node_role_audit.csv`：节点角色及官方/legacy 分量。
- `rdt_rdtlv_verification.csv`：56 对 Transformer 逐对证据。
- `topology_comparison.csv`：旧图诊断对照。
- `load_series_mapping_audit.csv`：逐目标母线的官方重建误差审计。
- `PROFILE_MANIFEST.json`：官方 profile SHA256 与下载清单。
- `data_provenance.md`：本报告。
