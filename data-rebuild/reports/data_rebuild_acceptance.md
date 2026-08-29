# PA-STFed SmartDS 数据重建验收报告

**审计日期：** 2026-08-29  
**适用阶段：** 数据来源、拓扑和序列审计；本阶段不运行模型训练。  
**验收状态：** `PASS_OFFICIAL_CANDIDATE / BLOCKED_LEGACY_MAPPING`

## 1. 结论摘要

依据官方 SMART-DS 用户指南（文档提交 `9cdf598733f94d72de09ce0015f4dda671982f9f`）和 OEDI 官方对象存储，当前 scope 已确定为：

```text
SMART-DS v0.9 / 2018 / Full_Texas / P10R / base_timeseries
substation: p10rhs0_1247
feeder:    p10rhs0_1247--p10rdt7719
```

官方 scope 的 `Master.dss` 明确 Redirect `LineCodes.dss`、`Lines.dss`、
`Transformers.dss`、`LoadShapes.dss`、`Loads.dss`，并使用 `Buscoords.dss`；
`Intermediates.txt` 只提供线路中间几何点。当前 15 个 OpenDSS 文件均已与 OEDI
远程对象逐项比较字节数和 SHA256，结果为 `15/15 PASS`。

## 2. 对验收问题的逐项回答

1. **官方子数据集：** SMART-DS v0.9、2018、Full_Texas、P10R、`base_timeseries`，指定 feeder 为 `p10rhs0_1247--p10rdt7719`。
2. **时间长度：** 35,040 个采样点；官方指南说明 timeseries 为 15 分钟分辨率、365 天。当前 legacy NPZ 没有绝对时间戳，因此只能使用序列索引，不能推断具体日期。
3. **92 个目标节点：** 92/92 个 legacy 非零节点可与官方 `Loads.dss` 母线匹配；官方独立候选中的 92 个 target 均有 Load 元件且序列至少一个非零点。
4. **181 个零负荷节点：** 官方图中主要是未挂载 Load 的结构母线或 transformer/line 中间母线，应保留作消息传播中继，不插值、不进入预测 loss。
5. **57 个 legacy 分量：** 旧邻接有 216 条边但没有写入 56 条 Transformer 边；加入官方 Line+Transformer 后为 1 个连通分量。因此 57 个分量主要是旧预处理漏读 Transformer connectivity，不是官方 feeder 的真实断连。
6. **56 对 `rdt-rdtlv`：** `rdt_rdtlv_verification.csv` 显示 56/56 对均在官方 `Transformers.dss` 找到设备名、文件路径和行号证据。
7. **旧 NPZ 是否遗漏 Transformer：** 是。旧图的 216 条无向边与官方 Line 数量一致，但遗漏 56 条官方 Transformer connectivity。
8. **官方完整拓扑：** 273 节点、216 条 Line 无向边、56 条 Transformer 无向边，共 272 条无向边，1 个连通分量。当前图中每条有向边保留 `edge_type` 和 `edge_source=file:line`。
9. **是否需要 MST：** 不需要。官方设备图已经连通；Euclidean MST、节点名称补边和距离推断边不得进入 canonical physical graph，只能作为历史敏感性记录。
10. **推荐图：** 下一阶段优先使用 full official physical graph，保留 181 个结构节点作为 relay，仅对 92 个 target 计算损失。target projection 图为 4,186 条无向边、密度 1.0，过密，不作为默认稀疏图。
11. **PA-STFed 适配：** loader 需要读取 `edge_index`、`edge_type`、`edge_source`、`target_mask`、`load_mask` 和 full/target 映射；物理图注意力只接收官方设备边；loss 只索引 target；旧 `hop_radius` 和 MST 逻辑降级为 legacy 诊断。
12. **不确定性：** legacy `load_ts` 的生成脚本、列映射、单位和缩放仍未确认。官方重建序列与 legacy 序列逐点对比为 `0/92` 精确通过，最大绝对误差约 21.9 kW，故 legacy 路径保持 blocker。

## 3. 产物与校验

官方独立候选目录：

```text
data/processed/official_v1_2018_P10R_p10rdt7719/
  smartds_full_graph_v2.npz
  smartds_target_graph_v2.npz
  smartds_metadata_v2.json
```

关键摘要：

| 项目 | 数值 |
|---|---:|
| full 节点 | 273 |
| target 节点 | 92 |
| 结构节点 | 181 |
| 时间点 | 35,040 |
| 采样间隔 | 15 min |
| Line 无向边 | 216 |
| Transformer 无向边 | 56 |
| full 无向边 | 272 |
| full 连通分量 | 1 |
| target projection 无向边 | 4,186 |

SHA256：

```text
official load series: c1205dda913b827a6dfecd35cb733927431eda946c30650cc9e22412b9194f7a
smartds_full_graph_v2.npz: 7997e0882f6f35b2c8af4bcdeeb43e011bcf9d45adfefd25f1a840c3151b940a
smartds_target_graph_v2.npz: 47abe7bf344821f152e27a2113c7b9d02f3e461a826b324fe7152aa272ee23c1
legacy smartds_graph.npz: 06a8c1d583159c1e3e972cdf12250e665e93f7f425fc83db57c772aa95cdb82f
```

验证命令（只做数据检查）：

```powershell
cd E:\pa_stfed_project\data-rebuild
python .\preprocess\validate_smartds_v2.py `
  --raw-root .\data\raw\SMARTDS\candidate_v09_2018_P10R_p10rdt7719_source `
  --processed .\data\processed\official_v1_2018_P10R_p10rdt7719 `
  --feeder-root p10rhs0_1247/p10rhs0_1247--p10rdt7719 `
  --report .\reports\official_v1_2018_P10R_p10rdt7719\validation_report.json
```

当前结果：`PASS`。检查包括节点唯一、边界合法、邻接一致、边属性对齐、官方
来源、无 MST、target 映射可逆、序列形状和有限性、target 非零、full 图连通性、
投影统计一致性和无 NaN/Inf。

legacy/provisional 目录仍会返回 `FAIL`，这是预期的保护行为：未验证的旧序列
不能静默进入模型。

## 4. 下一阶段闸门

在用户明确采用官方独立候选后，才允许修改主工程 loader；修改前先做一次不训练
的读取 smoke test，确认 full graph relay、92 个 target mask、节点映射和客户端
切分均正确。只有 smoke test 通过后，才可重新注册并运行 PA-STFed、FedAvg、
FedProx、Personalized 或拓扑消融实验。

SmartDS 是合成配电网，8 个客户端是空间划分模拟，不得把结果写成真实台区实测
结论。FedAvg/FedProx 只能描述为数据不出域的协同训练机制，不能据此宣称严格
隐私保证。

