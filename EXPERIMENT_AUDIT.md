# PA-STFed 实验完整性审计

**日期**：2026-08-29  
**项目**：基于时空注意力与联邦学习的配电网负荷预测  
**审计对象**：`E:\pa_stfed_project`  
**代码版本**：`20260829_topology_projection_v2`

## 总体结论：WARN（可继续，但不得引用旧结果证明修正方案）

SmartDS 数据和评估代码未发现伪造 ground truth 或自归一化指标；但当前 `outputs/`
中的 15 个结果文件全部来自旧版本 `20260826_literature_baselines_v4`、`graph=inf`，
且只评估验证集。修正后的拓扑、三组生死实验和最终测试尚无正式结果，因此论文目前只能
写方法设计和数据审计，不能写“PA-STFed 已优于基线”。

## A. Ground Truth 来源：PASS

- `run.py:737-779` 的基线使用 `LoadWindowDataset` 目标窗口；
- `run.py:737-781` 的模型评估从数据集 target 计算指标；
- `federated.py:31-58` 仅做反归一化和指标计算，没有从模型输出生成参考值。

结论：ground truth 来自 SmartDS 的负荷序列，属于 `simulation_only`（合成数据）评估。

## B. 指标归一化：PASS（需保留口径说明）

`metrics.py:17-49` 的 WAPE、sMAPE、MAPE 使用预测/目标原始负荷尺度；MAPE 仅排除低于
训练集节点阈值的位置，并返回有效覆盖率。未发现用模型自身最大值、均值或预测范围归一化
指标的代码。论文必须同时报告 MAPE 阈值定义和覆盖率，不能单独引用 MAPE 百分比。

## C. 结果文件与主张范围：WARN

- 15 个 `*_result.json` 均存在且可解析，但 `code_revision` 为旧版 v4；
- 旧文件中的 `graph_mode` 为历史 `inf`，不能代表 v2 的零负荷中继投影与有效节点 MST；
- `outputs/all_manifest.json` 当前仅记录 dry-run，没有完成的新拓扑/双图/联邦结果；
- 最终测试没有运行，不能在摘要或结论中报告测试精度。

行动：保留旧文件作为 `ARCHIVED`，新实验使用 v2 配置指纹和独立标签；`--resume` 只允许
跳过相同 `config_signature` 的结果。

## D. 评估实现与死代码：PASS/WARN

- `run.py:737-780` 的主评估函数被集中式和联邦路径调用；
- `run.py:782-850` 的固定时间块 WAPE 统计被最终结果调用，并由 `analyze_results.py`
  用于配对 Bootstrap；
- `metrics.py:52-78` 的门控统计在 PA-STFed 诊断路径调用。

风险：旧结果没有时间块充分统计量，不能事后补做严格配对 Bootstrap；必须使用 v2 重新运行。

## E. 范围充分性：WARN

当前仅有旧拓扑下的集中式验证结果，没有修正拓扑、双图和联邦三组核心实验，也没有多种子
统计检验。因此只能使用“初步验证”或“历史档案”，不能使用“全面”“稳健”“显著优于”等措辞。

## F. 评估类型：simulation_only

SmartDS 是合成配电网；8 个客户端是空间划分模拟。结果不能直接外推为真实台区实测结果。
FedAvg/FedProx 只表示原始窗口不集中上传的协同训练协议；除显式 DP 对照外，不提供形式化
隐私保证，DP 实现也不包含密码学安全聚合。

## 关键审计事实

- 原始图：273 节点、216 条无向边、57 个连通分量，满足森林恒等式；
- 有效节点：92；零负荷节点：181；训练时段有效节点同样为 92，`active_mask_stable_across_full_series=true`；
- 有效节点诱导图：92 个分量、0 条边；零负荷中继投影图：92 节点、54 条边、56 个分量；
- 修正 MST：增加 55 条有效节点候选桥接边，得到 92 节点、109 条边、1 个分量；
- 历史全节点 MST 的 56 条桥接边中，active-active=0、active-zero=1、zero-zero=55；
- 联邦 PA-STFed state_dict：47 个条目，FedAvg/FedProx 共享 45 个，功能图节点嵌入 2 个始终本地；
  `personalized_head=true` 时另有 6 个 `head.*` 条目本地保留；
- DP 会计：`privacy.py:68-99`，全客户端参与 (q=1)，均值噪声方差为
  \(\sigma^2 C^2/|\mathcal{S}_r|^2\)。

## 主张影响

| 主张 | 审计结论 |
|---|---|
| SmartDS 拓扑事实与零负荷中继投影 | 支持（数据事实 A） |
| MST 候选边提高预测精度 | 尚无 v2 结果，不支持 |
| 边来源标记带来独立增益 | 尚无 v2 结果，不支持 |
| 双图互补 | 尚无 v2 结果，不支持 |
| 联邦优于 LocalOnly/集中式 | 尚无 v2 结果，不支持 |
| FedAvg/FedProx 提供严格隐私 | 不支持；只能写数据留域协议 |

## 下一步

按 `refine-logs/EXPERIMENT_PLAN.md` 运行 B0--B5；任何“优于”结论必须附
`analyze_results.py` 的配对区块 Bootstrap 差异均值和 95% CI。若模块没有稳定增益，
按实验计划降级或删除其论文主张。
