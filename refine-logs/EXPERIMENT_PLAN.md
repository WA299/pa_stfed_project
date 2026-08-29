# PA-STFed 论文可发表性导向实验计划

**项目**：基于时空注意力与联邦学习的配电网负荷预测  
**数据**：SmartDS 合成配电网，15 min 采样  
**代码版本**：`20260829_topology_projection_v2`  
**计划日期**：2026-08-29

## 研究边界

1. SmartDS 是合成配电网，实验结论仅适用于该数据文件及其预处理口径，不能写成真实台区实测结果。
2. 原始 `adj` 在 273 个节点上有 57 个连通分量、216 条无向边，且 (216-273+57=0)，即 57 棵树组成的森林。181 个零负荷节点不进入预测损失，但保留为拓扑中继。
3. 有效节点诱导子图有 92 个孤立点；零负荷中继投影后为 92 节点、54 条边、56 个分量的 `projected_raw` 图。实验命令 `forest` 是其兼容别名，不预设投影图无环。
4. 只有投影后仍不连通的 56 个含负荷分量参与确定性 Kruskal MST，增加 55 条有效节点端点候选边。候选边是启发式拓扑填充关系，不能解释为真实线路、导线长度或阻抗。
5. 数据没有可可靠对齐的真实日期、星期、节假日和天气字段；输入仅为归一化负荷与日/周相对周期特征。历史长度 96（24 h），预测长度 12（3 h）。
6. 8 个客户端是空间划分模拟，不代表真实独立法人或物理隔离网络；重复负荷曲线组不得跨客户端。

## 主张地图

| 主张 | 最低证据 | 关联实验 | 失败时的表述 |
|---|---|---|---|
| C1：在拓扑信息不完备时，零负荷中继投影与有效节点候选桥接能改善或稳定短期预测；来源标记是否有独立价值由数据决定。 | `projected_raw`、`mst_no_tag`、`mst_tag` 在相同时间切分、预算、种子下的客户端/节点指标；配对时间块 Bootstrap 的 WAPE 差异均值及 95% CI。 | B1 | 若 MST 无稳定增益，将其降级为拓扑敏感性分析；若 `mst_tag` 不优于 `mst_no_tag`，不宣称边来源标记贡献。 |
| C2：在空间 Non-IID 客户端上，共享时空编码与本地关系/个性化头的联邦协议可在不集中上传原始窗口的条件下提供可行的协同折中。 | LocalOnly、FedAvg、FedProx、Personalized 的宏平均与客户端尾部指标；报告参数共享审计和 Bootstrap CI。 | B2 | 若 FedAvg/FedProx 不改善，结论限定为“实现了数据留域的协同训练协议”；若个性化头不改善，将其降级为对照。 |

**必须排除的反主张**：任何提升不能来自测试集调参、随机时间打乱、重复曲线跨客户端泄漏、模型参数量不公平或把启发式边写成真实线路。所有“优于”只能在配对区块 Bootstrap 95% CI 不跨 0 时使用。

## 模型和评价口径

### 模型集合

- 时间序列基线：Persistence、Daily-Naive、LSTM（负荷预测方向直接基线）、iTransformer-style（现代多变量序列基线）。
- 时空图基线：AGCRN-adapted、Graph WaveNet-adapted。代码保留自适应图/扩散图核心，但使用统一 SmartDS 输入、节点数、隐藏维度和输出协议；论文中不得称为官方实现逐行复现。
- 主模型：PA-STFed，包含边感知物理图、本地静态功能图、时间 Transformer、双级门控和可选残差锚定。除实验开关外不新增模块。

### 指标

统一报告 WAPE、sMAPE、RMSE、MAE 和带训练集低负荷阈值的 MAPE；同时报告 MAPE 有效覆盖率。联邦任务逐客户端报告上述指标及客户端均值、标准差、P90、P95、最大值/最差客户端。节点级结果和聚合负荷结果分开呈现。

WAPE 的配对 Bootstrap 使用训练代码保存的固定时间块充分统计量：每 96 个窗口约为一天，块内先求绝对误差和与目标绝对值和，再计算块 WAPE。`analyze_results.py` 的差异定义为 `right - left`；负值表示右侧实验误差较低，CI 跨 0 时不得写“显著优于”。

## 生死实验块

### B0：数据和实现烟测（必做）

- **目的**：确认拓扑、时间切分、指标和 CUDA 路径无漏洞。
- **运行**：`python run.py audit`；每个核心实验先用种子 2026 做烟测。
- **检查**：`outputs/audit_report.json`、`topology_comparison.csv`、`federated_parameter_groups.csv`、配置指纹、`py_compile`。
- **通过条件**：无 NaN、测试集未读取、拓扑计数符合 273/92/57/56/55、客户端重复曲线跨区数为 0。

### B1：拓扑来源敏感性（主论文必做）

| 变体 | 图模式 | 固定项 | 目标 |
|---|---|---|---|
| Forest | `forest`（`projected_raw`） | PA-STFed、80/10/10、96→12、同一训练预算 | 仅使用原始拓扑经零负荷中继投影后的关系 |
| MST-no-tag | `mst_no_tag` | 同上 | 检验候选桥接关系本身是否有效 |
| MST-tag | `mst_tag` | 同上 | 检验显式边来源标记是否提供额外信息 |

三种变体必须使用相同种子集合、同一验证选择指标和早停规则。结果不预设排序；若差异显著，报告差异均值、CI 和每客户端尾部变化。

### B2：物理图与功能图互补性（主论文必做）

固定 `graph=mst_tag`，比较：

1. `spatial_physical_only`：物理图开启、功能图关闭；
2. `spatial_functional_only`：功能图开启、物理图关闭；
3. `spatial_dual_graph`：两图及空间门控开启。

该块只回答“双图是否互补”。若双图不优于最强单图，删除“双图带来增益”的叙事；门控诊断仅用于解释负结果，不用于挑选有利样本。

### B3：联邦协同与个性化（主论文必做）

固定 `graph=mst_tag` 和 PA-STFed，比较：

1. `local_only`：8 个客户端独立训练，无服务器聚合；
2. `fedavg`：共享状态等权平均；
3. `fedprox_005`：共享状态等权平均，近端系数 \(\mu=0.005\)；
4. `personalized_head`：共享编码器，`head.*` 在客户端本地保留。

所有方案每轮 8/8 客户端参与，`local_epochs=1`，最多 20 轮，验证宏平均 WAPE 早停。参数分组来自实际 `state_dict`，不能把本地功能图嵌入或个性化头误写成服务器共享参数。

### B4：集中式基线（主论文必做，三种子）

在锁定的 `mst_tag` 图和同一时间切分下运行 PA-STFed、LSTM、iTransformer-style、AGCRN-adapted、Graph WaveNet-adapted，并补充 Persistence/Daily-Naive。基线用于校准任务难度和相对定位，不把交通论文的数值直接当作 SmartDS 的目标值。

### B5：最终测试（锁参后一次）

只有 B1--B4 完成且配置冻结后，运行 `centralized_test`、`fedavg_test`、必要时 `personalized_head_test`。测试集不参与模型选择、超参搜索、门控阈值选择或拓扑规则选择。

### B6：可选 DP 附录

DP 不是主贡献。若保留，单独运行 `dp_fedavg`，记录 `(C, sigma, q, R, delta, epsilon)`、均值噪声方差、`privacy.py:68-99` 的 accountant 版本和“无密码学安全聚合”的限制。不能把 FedAvg/FedProx 或该 DP 对照写成严格隐私保证。

## 执行顺序

| 里程碑 | 任务 | 配置/种子 | 决策门 | 计算量 |
|---|---|---|---|---|
| M0 | 审计、编译、B1--B3 单种子烟测 | 2026 | 任一拓扑/参数/指标错误即停止训练 | 分钟级 |
| M1 | B4 集中式基线验证 | 2026,2027,2028 | 确认基线结果文件版本为 v2；旧 `graph=inf` 结果只能归档 | 中等，5 个模型×3 种子 |
| M2 | B1 拓扑三变体 | 2026,2027,2028 | 无增益则 MST 降级；来源标记无增益则不保留为贡献 | 中等，3×3 个 PA 任务 |
| M3 | B2 双图三变体 | 2026,2027,2028 | 双图不改善则删去双图主张并保留最简单图 | 中等，3×3 个 PA 任务 |
| M4 | B3 联邦四变体 | 2026,2027,2028 | 观察宏平均与 P90/P95；FedProx/个性化无益则仅作对照 | 较高，4×3 个联邦任务 |
| M5 | Bootstrap、门控诊断、表格 | 完成 M1--M4 后 | 只对同一时间块配对，CI 跨 0 不写“优于” | 低 |
| M6 | B5 最终测试 | 锁定配置后 3 种子 | 测试结果只作最终一次报告 | 中等 |

建议命令（均在 `E:\pa_stfed_project` 执行）：

```powershell
$py = "E:\pa_stfed_gpu_env\Scripts\python.exe"
& $py -u run.py audit
& $py -u run.py all --experiments topology_forest,topology_mst_no_tag,topology_mst_tag --seeds 2026 --resume
& $py -u run.py all --experiments spatial_physical_only,spatial_functional_only,spatial_dual_graph --seeds 2026 --resume
& $py -u run.py all --experiments local_only,fedavg,fedprox_005,personalized_head --seeds 2026 --resume
```

烟测通过后，把三条实验命令中的 `--seeds 2026` 改为 `--seeds 2026,2027,2028`。基线和最终测试使用：

```powershell
& $py -u run.py all --experiments load_lstm_validation,itransformer_validation,agcrn,gwnet --seeds 2026,2027,2028 --resume
& $py -u run.py all --experiments centralized_test,fedavg_test,personalized_head_test --seeds 2026,2027,2028 --resume
```

配对 Bootstrap 示例（右侧减左侧）：

```powershell
& $py analyze_results.py `
  --left outputs/topology_forest_seed2026_centralized_result.json `
  --right outputs/topology_mst_tag_seed2026_centralized_result.json `
  --split validation --aggregation micro --samples 10000 --seed 2026 `
  --output outputs/bootstrap_forest_vs_mst_tag_seed2026.json
```

## 失败诊断和降级规则

- **MST 变体无改善**：保留拓扑审计和敏感性表，正文不写拓扑补全提升。
- **`mst_tag` 与 `mst_no_tag` 接近或反向**：边来源编码降级为实现细节，不能声称其独立贡献。
- **双图不改善**：删除双图主叙事，使用性能最好的单图模型，门控只在附录报告。
- **FedProx 不优于 FedAvg**：FedProx 作为 Non-IID 对照，不把系数 0.005 写成最优普适值。
- **Personalized 不改善**：报告负结果，说明共享头在当前客户端规模下已足够或本地头数据不足。
- **PA-STFed 不优于基线**：不继续堆叠模块；将 PA-STFed 定位为可审计的系统实现，论文贡献转为拓扑/联邦协议验证和限制分析。
- **WAPE 较低但节点尾部很差**：同时报告宏平均、最差客户端和 P90/P95，结论限定为系统级或平均性能。

## 复现和审计清单

- [ ] `outputs/audit_report.json` 的 SHA-256、拓扑计数、时间切分和客户端 Non-IID 统计已归档。
- [ ] 所有正式结果的 `code_revision` 为 `20260829_topology_projection_v2`，旧 v4 结果不混报。
- [ ] 每个配置至少 3 个种子；汇总均值、样本标准差和有效样本数。
- [ ] 结果 JSON 含带 `aggregation_level`、origin 哈希和尾块长度的 WAPE 区块；联邦结果同时含全局 `validation_wape_blocks_micro` 与客户端 `validation_wape_blocks_clients`。
- [ ] 每个 DP 结果含六元组、accountant 源码版本和无密码学安全聚合声明。
- [ ] 所有“优于”结论均有配对区块 Bootstrap 差异均值与 95% CI。
- [ ] 测试集仅在 B5 锁参后访问。
