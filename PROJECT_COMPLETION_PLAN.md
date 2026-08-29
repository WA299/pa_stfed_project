# PA-STFed 大创结项执行方案

## 目标

在不更换数据集、不泄漏测试集、不继续无边界调参的前提下，完成一份可复现、证据边界清楚的大创结项报告。项目结论限定为 SmartDS 合成配电网，不宣称真实台区实测精度或新的联邦优化器。

## 当前完成度

- 数据读取、拓扑审计、严格时间切分：已完成。
- 集中式 PA-STFed、AGCRN-adapted、Graph WaveNet-adapted、LSTM、iTransformer-style：旧拓扑口径下已有验证集三种子结果；不能与修正拓扑结果混报。
- 旧 FedAvg、FedProx 结果使用了会拆分重复负荷曲线的客户端划分，已作废。
- 新划分把完全重复曲线组作为不可拆分单元，审计确认跨客户端组数为 0。
- GPU 训练、断点续训、配置指纹、结果汇总：已完成。
- 新划分上的 LocalOnly、FedAvg、FedProx、个性化头、拓扑/双图生死消融、最终测试和 Bootstrap：未完成。

## 结项主线

研究问题写为：在 SmartDS 的空间 Non-IID 客户端划分下，原始负荷数据保留在本地时，共享时空编码器与客户端私有预测头是否能够改善联邦预测的平均误差和尾部客户端误差？

方法写为：PA-STFed 作为共享时空编码器；FedAvg 和 FedProx 是联邦优化基线；共享编码器加私有预测头是个性化适配实验。个性化头属于 FedPer/FedRep 范式的项目适配，不写成首次提出的联邦算法。

## 必做任务

### 1. 冻结验证配置

保留当前 `config.yaml` 的 `history=96`、`horizon=12`、`graph=mst_tag`、80/10/10 切分、重复曲线组不可拆分的客户端划分和三种子。正式拓扑生死实验另行使用 `forest`（即 `projected_raw`）与 `mst_no_tag`；禁止依据测试集再调学习率、窗口或模型结构。

### 2. 增加一个个性化联邦任务

运行 `personalized_head=true`：服务器聚合共享编码器，客户端预测头本地保留；同时用 `local_only` 作为无协作下限。不要再加入原型、动态采样、多中心聚合或新的门控模块。

### 3. 完成最小比较

集中式 PA、LocalOnly、FedAvg、FedProx、个性化头以及三组拓扑/双图消融必须报告：WAPE、sMAPE、MAPE、RMSE、MAE；联邦系统另报客户端标准差、P90/P95 和最差客户端。

### 4. 运行一次独立测试

锁定配置后，对集中式 PA、FedAvg 和个性化模型运行三种子测试。测试结果不得回写调参配置。若个性化实验未改善，测试表仍保留 FedAvg，并在讨论中报告负结果。

### 5. 补齐统计和隐私说明

按相同时间块进行配对 Bootstrap，报告 WAPE 差异均值及 95% 置信区间。DP 若作为项目要素保留，至少运行 DP-FedAvg 并记录 `(C, sigma, q, R, delta, epsilon)` 与 accountant 版本；否则将“隐私保护”改写为“原始负荷窗口不上传的联邦协议”。

## 建议分工（不写入论文姓名）

- 数据与工程：冻结环境、复查配置指纹、运行测试和保存日志。
- 实验与统计：生成每客户端指标、P90、Bootstrap 区间和消融表。
- 论文与答辩：整理方法图、实验协议、局限性、结果等级和答辩问答。

## 报告结构

1. 问题定义与 SmartDS 数据边界。
2. 启发式拓扑填充和 PA-STFed 模型。
3. 空间 Non-IID 联邦协议、FedAvg/FedProx 和个性化头。
4. 实验设置、指标、隐私六元组和统计方法。
5. 集中式、联邦、客户端公平性和 DP 结果。
6. 失败分析：PA 未超过 Graph WaveNet，FedProx 未超过 FedAvg，SmartDS 合成拓扑限制。
7. 结论：证明可行性和适配价值，不夸大算法原创性。

## 停止条件

- 个性化头运行完成并完成一次测试后，停止继续搜索新模型。
- 若个性化头没有改善，不再增加模块；以透明负结果完成报告。
- 所有结果、配置和日志归档后，开始写作，不再反复覆盖旧实验。

## 现有文件

- 文献依据：`literature/REFERENCES.md`。
- 拓扑核查：`PROJECT_AUDIT_20260829.md`、`outputs/topology_comparison.csv`、`outputs/federated_parameter_groups.csv`。
- 实验计划：`PROJECT_COMPLETION_PLAN.md`。
- 新结果：`outputs/`；带 `grouped` 标签的联邦结果属于修正后的客户端划分。
