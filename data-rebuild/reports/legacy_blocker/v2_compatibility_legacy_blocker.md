# PA-STFed v2 数据层兼容性与实验闸门报告

审计日期：2026-08-29  
当前状态：`BLOCKED_NO_VERIFIED_CANONICAL_SOURCE`  
本报告不包含任何正式模型训练结果。

原工程不是可用的 Git 仓库，因此本阶段采用独立 `data-rebuild` 工作目录隔离；旧文件和旧结果保持原位。

## 1. 当前项目结构与数据流

旧工程位于 `E:/pa_stfed_project`，核心链路为：

```text
smartds_graph.npz
  -> data.py/SmartDS 读取
  -> 92 个非零序列目标节点 + 181 个全年零序列节点
  -> 旧 raw 图 / 零负荷中继投影 / 历史 MST
  -> 15 min 序列窗口（history=96，horizon=12）
  -> PA-STFed 或基线模型
  -> LocalOnly/FedAvg/FedProx/Personalized
```

旧 NPZ 的 SHA256 为 `06a8c1d583159c1e3e972cdf12250e665e93f7f425fc83db57c772aa95cdb82f`；字段形状为 `node_coords=[273,2]`、`adj=[273,273]`、`edge_index=[2,432]`、`load_ts=[35040,273]`、`node_ids=[273]`。审计发现旧 `adj` 有 216 条无向边、57 个分量；这些数字描述二次处理文件，不能当作官方物理拓扑。

新链路固定为：

```text
官方 SMART-DS/OpenDSS raw
  -> 解析 Line/Transformer/Load/Buscoords 及来源文件/行号
  -> full physical graph（含零负荷结构节点）
  -> target_mask 选出有效负荷节点，不对结构节点的 0 插值
  -> target topology projection（记录原始路径 hop 与路径节点）
  -> metadata + NPZ + 自动一致性断言
  -> 下一阶段才适配 PA-STFed/Federated loader
```

## 2. 拓扑构建核查结果

| 方案 | 节点 | 无向边 | 连通分量 | 推断边 | canonical |
|---|---:|---:|---:|---:|:---:|
| legacy raw full | 273 | 216 | 57 | 0 | 否 |
| legacy target induced | 92 | 0 | 92 | 0 | 否 |
| legacy target relay projection | 92 | 54 | 56 | 0 | 否 |
| legacy full Euclidean MST | 273 | 272 | 1 | 56 | 否 |
| official full graph | 未生成 | 未生成 | 未生成 | 0 | 否 |

后三个旧方案的统计来自 `reports/topology_comparison.csv`。零负荷中继投影证明“直接删除零节点”会丢失一部分路径信息，但它仍然没有官方设备证据。历史 56 条 MST 边只用于数量对照，不能进入 canonical physical graph。

### 官方来源探针

从 OEDI 公共数据湖下载的探针为：

`s3://oedi-data-lake/SMART-DS/v1.0/2018/SFO/P10U/scenarios/base_timeseries/opendss/`

已保存根目录 `Buscoords.dss`、`Master.dss`，以及 feeder `p10uhs0_1247/p10uhs0_1247--p10udt2190/` 的 `Master.dss`、`Lines.dss`、`Transformers.dss`、`Loads.dss` 和坐标文件。解析得到 975 条 Line 端点关系、153 条 Transformer 端点关系和 412 个 Load 母线记录，但与当前 273 个 legacy 节点匹配数为 0/273。官方探针节点命名为 `p10udt...`/`p10udt...lv`，不等于当前 `p10rdt...`/`p10rdt...lv`；不能据此把两者视为同一 feeder。

因此当前事实是：

1. 当前 NPZ 的 SMART-DS region/substation/feeder/scenario/year 尚不能确定。
2. 56 对 `p10rdtXXXXX` 与 `p10rdtXXXXXlv` 全部 `verified=false`，详见 `rdt_rdtlv_verification.csv`。
3. 尚不能回答“旧 NPZ 是否遗漏 Transformer edges”；只能说当前已获取的官方候选文件没有匹配旧节点。
4. canonical 构建要求 273/273 节点完整匹配；未获得可靠匹配前，不能生成 `smartds_full_graph_v2.npz` 或 `smartds_target_graph_v2.npz`，也不能进入正式实验。

## 3. 节点角色与时间序列

`load_ts` 有 35040 行，按 15 分钟采样等于 365 天；文件没有可信绝对 timestamp，因此只能使用序列索引和 15 分钟周期，不能推断年份、星期、节假日或天气。审计确认 NaN=0、Inf=0、重复整行=0、最大值约 23.20；92 个非零节点没有发现中间零值。单位和缩放在 NPZ 元数据中没有标注。

`node_role_audit.csv` 目前只做证据分层：非零目标、全年零序列、结构节点候选、尚未验证节点。官方 Load/Bus/Transformer/Line 证据补齐后才可以把零序列节点最终归类为结构母线、Transformer side bus、数据映射错误或缺失负荷。任何结构节点的 0 都禁止插值。

## 4. 联邦参数共享机制核查

旧模型实现位置：`E:/pa_stfed_project/model.py` 的 `local_parameter_prefixes`、`shared_state_dict` 和 `load_shared_state`；联邦训练位置：`E:/pa_stfed_project/federated.py` 与 `run.py`。

| 参数组 | FedAvg/FedProx | Personalized | 说明 |
|---|:---:|:---:|---|
| `input_projection.*` | 共享 | 共享 | 输入特征到隐藏空间 |
| `physical.*` | 共享 | 共享 | 边感知物理图注意力，边张量在前向传入 |
| `functional.value.*` | 共享 | 共享 | 功能图消息变换 |
| `functional.embedding_1/2` | 本地 | 本地 | 节点数依客户端不同，静态功能图关系 |
| `spatial_gate.*` | 共享 | 共享 | 物理/功能表征的一级门控 |
| `temporal.*` | 共享 | 共享 | 时间 Transformer 与位置嵌入 |
| `temporal_gate.*` | 共享 | 共享 | 二级时空门控 |
| `head.*` | 共享 | 本地 | Personalized 仅将预测头留在客户端 |

该分组与设计意图一致：节点相关功能嵌入不能跨不同 `N_k` 的客户端聚合；PA-STFed 其他层形状不随客户端节点数改变。v2 loader 需要把 full graph 的官方 `edge_type`、target 映射和投影 hop 传给 `graph_view`，但 FedAvg/FedProx 的状态聚合本身与数据来源解耦。当前代码的 `hop_radius=2` 是旧邻域截断参数；v2 应先依据官方 target shortest-hop 分布审计后再决定是否保留，不能默认沿用。

## 5. 三组核心实验配置（下一阶段，当前不运行）

只有 v2 canonical topology 通过审计后才允许预注册以下实验。所有任务固定同一时间切分、输入输出窗口、节点集合、客户端划分和随机种子。

| 组别 | 配置 | 要回答的问题 |
|---|---|---|
| 拓扑 | `Forest`、`MST-no-tag`、`MST-tag`（若官方图仍真实不连通，MST 只能作为额外敏感性） | 原始可达关系、候选关系和来源标记是否有独立证据 |
| 双图 | `Physical-only`、`Functional-only`、`Physical+Functional` | 官方拓扑与本地数据驱动关系是否互补 |
| 联邦 | `LocalOnly`、`FedAvg`、`FedProx`、`Personalized` | 空间 Non-IID 下协同和个性化是否稳定改善尾部客户端 |

基线保留集中式 PA-STFed、LSTM、iTransformer、AGCRN-adapted 和 Graph-WaveNet-adapted；它们只能在相同 v2 输入协议下作方法对照。所有指标应输出 WAPE、sMAPE、RMSE、MAE、低负荷阈值 MAPE 以及客户端均值、标准差、最差客户端、P90/P95。任何“优于”必须基于至少 3 个随机种子和配对时间块 Bootstrap 的差异均值及 95% CI。

## 6. 建议的 loader 与脚本改动

1. 新 loader：优先读取 `smartds_target_graph_v2.npz` 的 `target_load_ts`、映射数组和投影边属性；训练 loss 只在 `target_mask` 上计算。
2. full graph 输入：若模型需要结构中继，使用 full graph 的 `edge_index`/`edge_type`；若只在目标节点运行注意力，使用 target projection，并显式标记 `edge_source=topology_projection`。
3. `data.py`：旧 `heuristic_topology_imputation` 和 `heuristic_projected_topology_imputation` 降级到 `legacy`/敏感性分支；不得在 canonical graph 构建中调用。
4. `model.py`：边感知层保留 `edge_type` 输入接口；`d_hop`、坐标仅作图关系和位置先验，不能解释为线路长度或阻抗。
5. `run.py`：正式训练入口增加 v2 metadata SHA256、拓扑版本、节点映射和 source manifest 校验；metadata 状态为 `BLOCKED` 时直接拒绝训练。
6. 自动测试：节点唯一、边界合法、官方来源存在、邻接一致、映射可逆、无 NaN/Inf、时间长度一致、所有 target 有 Load 证据、无未验证 MST、clean raw 重跑确定性一致。

## 7. 推荐执行顺序与计算量

当前阶段只运行：

```powershell
cd E:\pa_stfed_project\data-rebuild
python .\preprocess\audit_smartds_data.py
python .\preprocess\build_smartds_graph_v2.py
```

两条命令均为 CPU 数据审计，当前 NPZ 规模应在秒级到分钟级完成，不占用 GPU。官方来源、节点角色和 canonical 图通过后，再做单种子小批量 smoke test；之后按“拓扑三变体 -> 双图三变体 -> 联邦四变体 -> 多种子复核 -> 锁参最终测试”的顺序运行。审计未通过前的预计正式训练计算量为零。

## 8. 可能成立与可能失败的论文贡献

最可能成立的贡献是一个证据可追溯的工程框架：在拓扑信息不完备、空间划分造成负荷异构的合成配电网案例中，区分官方物理关系、拓扑投影关系和本地功能关系，并系统比较共享与个性化联邦机制。它不是“首次联邦负荷预测”、不是“首次双图”、不是“首次 MST 补边”，也不提供严格隐私保证。

最可能失败的模块是未经官方证据支持的 MST/来源标记，以及在 10--13 个节点的小客户端上叠加过多门控。如果消融无稳定增益，应将其降级为敏感性分析或删除，而不是为了保留原设计强行解释。

## 9. 结论闸门

当前唯一可审计结论为：旧 NPZ 的基本数组和序列完整性通过初筛，但官方来源与物理拓扑尚未匹配；56 对 `rdt-rdtlv` 尚不能确认；181 个零负荷节点角色尚未最终确认；canonical v2 图尚未生成。因此本任务在数据层 blocker 处停止，旧 `outputs/` 中的训练结果不得用于新拓扑论文结论。
