# PA-STFed v2 数据层兼容性与实验闸门报告

审计日期：2026-08-29  
legacy 状态：`BLOCKED_LOAD_SERIES_MAPPING_UNVERIFIED`；官方独立候选状态：`RAW_SOURCE_VERIFIED_CANONICAL`  
本报告不包含任何正式模型训练结果。

## 1. 数据来源与数据流

官方依据为 [SMART-DS 用户指南](https://github.com/openEDI/documentation/blob/main/SMART-DS/Readme.md)，采用提交 `9cdf598733f94d72de09ce0015f4dda671982f9f` 的本地快照（与当前 main 内容仅尾部换行不同）。版本/年份/区域信息来自官方 OEDI 对象键和下载清单，OpenDSS 文件本身未嵌入独立版本字段。当前 legacy NPZ 已通过官方 OEDI 对象存储中的以下 scope 逐项追溯：

```text
SMART-DS v0.9 / 2018 / Full_Texas / P10R / base_timeseries
  substation: p10rhs0_1247
  feeder:     p10rhs0_1247--p10rdt7719
```

当前处理链为：

```text
官方 OpenDSS + profile 文件
  -> feeder-scoped 解析（父级 substation + 目标 feeder）
  -> 官方拓扑证据（273 节点，含结构母线；当前仅作审计）
  -> target_mask 选取 92 个有 Load 证据且序列非零的节点
  -> 仅 target 节点参与预测损失，结构节点保留作消息传播中继
  -> target shortest-hop / topology projection 作为诊断候选
  -> metadata、来源清单、CSV 审计和独立一致性测试
官方负荷序列独立路径：`Loads.dss kW * LoadShapes.dss mult`
  -> data/processed/official_v1_2018_P10R_p10rdt7719/
  -> validate_smartds_v2.py = PASS
  -> 下一阶段才允许适配 PA-STFed/FedAvg/FedProx
```

旧 `E:/pa_stfed_project/smartds_graph.npz` 只保存在
`data/legacy/smartds_graph_legacy.npz`，SHA256 为
`06a8c1d583159c1e3e972cdf12250e665e93f7f425fc83db57c772aa95cdb82f`；旧
`outputs/`、checkpoint、日志和配置未修改。

## 2. 拓扑构建核查（证据等级 A）

| 方案 | 节点 | 无向边 | 连通分量 | 推断边 | canonical |
|---|---:|---:|---:|---:|:---:|
| legacy raw full | 273 | 216 | 57 | 0 | 否 |
| legacy target induced | 92 | 0 | 92 | 0 | 否 |
| legacy target relay projection | 92 | 54 | 56 | 0 | 否 |
| legacy full Euclidean MST | 273 | 272 | 1 | 56 | 否 |
| official full physical | 273 | 272 | 1 | 0 | 是 |

官方 scope 内解析到 216 条唯一 Line 边和 56 条唯一 Transformer 边。旧 NPZ 的 216 条边与官方 Line 数一致，但遗漏了 56 条 Transformer connectivity；因此旧 57 个分量主要由预处理漏读 Transformer 文件造成，而不是 SmartDS 原始网络真实断连，也不是删除零负荷节点造成。该拓扑结论已获得官方文件证据；legacy 数值映射仍未通过，但官方独立序列候选已单独生成并验证。

`rdt_rdtlv_verification.csv` 显示 56/56 对
`p10rdtXXXXX <-> p10rdtXXXXXlv` 均在官方
`p10rhs0_1247/p10rhs0_1247--p10rdt7719/Transformers.dss` 中找到设备名、文件和行号证据。canonical 图不包含 Euclidean MST、kNN 或根据名称生成的边。

`Intermediates.txt` 解析到 149 条线段几何记录，仅作为可视化证据附着到
对应 Line，不创建额外物理节点或边。

完整官方图中的 `edge_type` 仅为 `line` 或 `transformer`；每条有向边均带
`edge_source=file:line`，并保存不虚构的 `length/units/phases/enabled/switch`
等可见属性。坐标距离和 hop 只表示图关系/位置先验，不能解释为导线物理长度或电气阻抗。

## 3. 节点与时间序列审计

- 273 个节点 ID 唯一；92 个节点全年存在非零负荷，均有官方 `Loads.dss` 证据。
- 181 个全年零序列节点均为当前官方图中的结构母线候选；它们保留在 full graph 中，不插值、不作为预测目标。
- `load_ts` 形状为 `[35040, 273]`，无 NaN/Inf、无重复整行，按官方 15 min 配置对应 365 天。
- `LoadShapes.dss` 审计得到 61 个 shape、122 个 `com_kw/com_kvar` profile 文件；实际引用 profile 已下载至 `data/raw/SMARTDS/.../profiles/`，并在 `PROFILE_MANIFEST.json` 记录 SHA256。另按官方文档补齐了当前 61 个 `res/com` 客户 profile 对应的 parquet（约 110 MB），其 `Time` 和 `total_site_electricity_kw` 字段用于独立交叉核验。
- legacy NPZ 不含绝对 timestamp；因此模型只能使用序列索引与 15 min 周期特征，不能加入未经对齐的真实日期、星期、节假日或天气变量。
- 官方 OpenDSS-native 重建应按每个母线在 `Loads.dss` 中的全部 Load 元件计算 `sum(kW * mult_profile)`；中心抽头 `_1/_2` 的 `kW` 已经是拆分后的半客户峰值。若直接读取整户 parquet，则对每个 `_1/_2` 元件使用 0.5。当前 `92/92` 个母线虽有 profile 文件，但 `0/92` 个节点逐点通过，状态为 `BLOCKED_VALUES_MISMATCH`，最大绝对误差约 `21.9` kW、平均相对误差约 `66.3\%`。不允许用自由缩放、时间平移或列重排解除该 blocker；详细 parquet 交叉核验见 `official_load_data_audit.json/csv`。
- 因此 legacy `load_ts` 的单位、缩放、生成脚本和列映射仍不能确认；现阶段只能作为历史输入审计，不能声称已完全复现官方 timeseries。

## 4. v2 图数据与投影诊断

旧流程此前生成的 NPZ 仅是负荷映射未验证前的 provisional 产物；重新运行构建脚本后已移入显式隔离目录，不应将其送入模型：

```text
data/processed/candidate_v09_2018_P10R_p10rdt7719/
  blocked_provisional/*.npz.blocked
  smartds_metadata_v2.json
```

其拓扑部分曾解析出 273 节点、272 条官方设备边、1 个分量；但因 legacy 负荷映射未验证，文件已降级为隔离的 provisional 证据，故不具备 canonical 训练资格。
官方独立候选位于：

```text
data/processed/official_v1_2018_P10R_p10rdt7719/
  smartds_full_graph_v2.npz
  smartds_target_graph_v2.npz
  smartds_metadata_v2.json
```

该候选的负荷序列直接来源于官方 OpenDSS-native 公式，未读取 legacy
`load_ts`；其独立验证报告为 `PASS`。
依据“最短路径内部不含第三个 target 节点”的投影规则，92 个 target 产生
4186 条无向候选边，即密度 1.0、平均度 91、最大度 91、1 个分量；hop
分布为 min=2、median=26、P90=51、max=66。该投影图明确标记为
`target_projection_overdense=true`，不应直接作为稀疏空间注意力图。

后续 PA-STFed 优先方案是 full official graph + zero-load structural relay，
仅在 92 个 target 节点计算 loss。target projection 仅用于敏感性/诊断，不能
把投影边称为原始直接线路。

## 5. 联邦接口约束

当前 8 个客户端是合成网络上的空间聚合模拟，不代表真实独立法人或物理隔离。
v2 loader 需要传入 full graph 的 `edge_index/edge_type/edge_source`、
`target_mask` 和 full-target 映射；FedAvg/FedProx 聚合逻辑与数据来源解耦。
客户端功能嵌入等节点数相关参数必须本地保留，不能跨 `N_k` 不同的客户端聚合；
共享层才进入服务器聚合；个性化预测头按实验协议保留在客户端。

## 6. 实验闸门（当前不运行）

在当前数据修复任务验收前禁止运行正式模型训练。若下一阶段明确选择官方独立候选，必须使用其 `validation_report.json=PASS` 和
`official_series_verified=true`；legacy 路径仍为 `FAIL/BLOCKED_VALUES_MISMATCH`。
下面矩阵仅作下一阶段预注册，不产生论文结论：

1. 拓扑敏感性：official full physical（主方案）与 official target projection（仅诊断；当前过密）；legacy MST-no-tag/tag 如保留，只能作为历史敏感性对照，不作 canonical 物理拓扑。
2. 双图消融：Physical-only、Functional-only、Physical+Functional。
3. 联邦机制：LocalOnly、FedAvg、FedProx、Personalized。

集中式 PA-STFed、LSTM、iTransformer、AGCRN-adapted、Graph-WaveNet-adapted
只能在相同 target、窗口、时间切分和指标协议下比较。所有结论必须报告
WAPE、sMAPE、RMSE、MAE、低负荷阈值 MAPE 及客户端均值/标准差/最差客户端/P90/P95；
“优于”需基于至少 3 个种子和配对时间块 bootstrap 的差异均值及 95% CI。

## 7. 当前可成立与不可成立的结论

可成立：当前 NPZ 的 273 个节点和拓扑端点可与 SMART-DS v0.9/2018/Full_Texas/P10R
的指定 feeder 逐项匹配；旧图遗漏 Transformer 边；零负荷结构节点应保留为 relay；官方
设备图不需要 MST；官方 OpenDSS-native 独立候选已完成可复现重建和一致性验证。上述是拓扑/结构层面的事实。

不可成立：把旧 57 个分量说成真实断连；把节点名称或欧氏距离说成线路证据；把当前
legacy `load_ts` 宣称为官方 profile 的直接重建；在 blocker 未解除前运行正式模型并发布
精度；把联邦训练宣称为严格隐私保证；把合成网络空间划分结果外推为真实台区实测结论。
