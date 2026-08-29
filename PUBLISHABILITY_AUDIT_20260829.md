# PA-STFed 论文可发表性导向系统核查

**项目**：基于时空注意力与联邦学习的配电网负荷预测  
**数据**：SmartDS 合成配电网  
**代码版本**：`20260829_topology_projection_v2`  
**核查日期**：2026-08-29

## 1. 结论先行

当前代码已具备可复现实验的基本闭环，但论文证据仍处于“方法与审计完成、修正拓扑实验未完成”阶段。`outputs/` 中已有的 15 个结果文件全部来自 `20260826_literature_baselines_v4`、历史 `graph=inf` 和验证集-only 流程，不能作为修正方案 PA-STFed 的最终结果，也不能与新版本结果混合汇总。

本次核查不新增复杂模型模块，优先修正了结果可审计性：结果记录数据源 SHA-256、历史/预测长度、节点集合哈希和实际窗口起点哈希；联邦结果额外记录全局 micro-WAPE 区块，避免把集中式 micro-WAPE 与客户端 macro-WAPE 误作同一统计量；`analyze_results.py` 对版本、数据源、节点集合、客户端划分和时间窗口不一致采用 fail-fast。

## 2. 项目结构与数据流

| 文件 | 职责 |
|---|---|
| `data.py` | 读取 NPZ、识别有效节点、零负荷中继投影、有效节点 MST 候选桥接、时间切分、窗口和归一化 |
| `model.py` | PA-STFed、LSTM、iTransformer、AGCRN-adapted、Graph WaveNet-adapted |
| `federated.py` | Charbonnier 损失、本地训练、FedAvg/FedProx、可选中心 DP 更新聚合 |
| `run.py` | 审计、集中式训练、联邦训练、基线、早停、区块统计、实验矩阵调度 |
| `metrics.py` | MAE、RMSE、WAPE、sMAPE、带训练期低负荷阈值的 MAPE、门控诊断 |
| `privacy.py` | 客户端更新 L2 裁剪、均值高斯噪声、全参与率 `q=1` 的 RDP 会计 |
| `experiments.yaml` | 正式实验名、固定覆盖项、种子数和是否纳入默认矩阵 |
| `analyze_results.py` | 严格对齐的配对时间块 WAPE Bootstrap，不重新训练 |

数据流为：NPZ -> 92 个有效负荷节点和 181 个零负荷节点识别 -> 保留零节点中继并投影原始拓扑 -> 对仍不连通的有效节点分量执行确定性 Kruskal MST -> 训练期 robust median/IQR 归一化 -> 过去 96 步输入、未来 12 步目标 -> 集中式全节点模型或 8 个空间客户端 -> 验证集选择最佳 epoch/round -> 锁参后一次性访问测试集。

研究边界必须明确写入论文：SmartDS 是合成网络；8 个客户端是空间划分模拟，不代表真实法人或物理隔离；没有真实绝对时间戳、天气、节假日字段，不能引入外部日期/天气变量；MST 边和所有 `d_hop`、坐标距离只表示图关系和位置先验，不能解释为真实线路、导线长度或阻抗。

## 3. 拓扑构建核查

### 3.1 原始图与零负荷节点的作用

核查事实（证据等级 A）：

| 图方案 | 节点 | 无向边 | 连通分量 | 推断边 | 含义 |
|---|---:|---:|---:|---:|---|
| `raw_full` | 273 | 216 | 57 | 0 | 原始邻接图，含零负荷节点 |
| `active_induced` | 92 | 0 | 92 | 0 | 删除零负荷节点后的诱导图，仅作诊断 |
| `projected_raw`（命令别名 `forest`） | 92 | 54 | 56 | 0 | 零负荷中继投影后的原始可达关系 |
| `legacy_inf` | 273 | 272 | 1 | 56 | 历史全节点 MST，不再作为正式方案 |
| `mst_projected`（`mst_no_tag`/`mst_tag`） | 92 | 109 | 1 | 55 | 投影后仅以有效节点端点执行 MST |

原始图满足 (216-273+57=0)，因此数学上是 57 棵树组成的森林。57 个分量不是删除零负荷节点后才产生的；删除节点后，92 个有效节点之间的诱导边恰为 0，说明零负荷节点承担了原始拓扑中继作用。正式代码通过 `SmartDS.projected_topology()`（`data.py:216`）保留该作用：同一零节点连通块的有效邻居两两建立投影可达边，并记录完整原始图上的最短跳数。投影后可能出现环，因此不能把 `projected_raw` 断言为严格森林。

旧全节点 MST 的 56 条桥接边中，active-active=0、active-zero=1、zero-zero=55。旧方案在只预测 92 个有效节点的设定下几乎没有直接有效节点桥接信息，故不能继续作为主实验图。

### 3.2 正式 MST 规则

`SmartDS.heuristic_projected_topology_imputation()`（`data.py:284`）先以投影图的 56 个含负荷分量为节点。对任意两个分量 (C_a,C_b)，计算所有有效节点端点对的最小欧氏坐标距离：

\[
w(C_a,C_b)=\min_{i\in C_a,\,j\in C_b}\lVert p_i-p_j\rVert_2.
\]

候选边按 ((w,\min(i,j),\max(i,j))) 字典序排序，对分量级候选图执行 Kruskal 算法，直到选出 (56-1=55) 条边。该过程称为 **启发式拓扑填充（Heuristic Topology Imputation）**，是可检验的连通性假设，不是真实线路恢复。

投影原始图有效节点对的可达覆盖率为 54/4186=1.29%，跳数中位数为 1；修正 MST 后覆盖率为 100%，跳数中位数为 14、P90=29、最大 41。`d_hop` 是图论路径步数；`d_geo` 是节点坐标先验。二者均不得写成导线物理长度、电阻、电抗或阻抗。旧全节点 MST 的有效节点跳数中位数为 26、P90=51，不能与修正投影图的跳数分布混用。

正式拓扑生死实验固定模型、窗口、切分、训练预算和种子，仅改变图模式：

1. `topology_forest`：`projected_raw`，检验零负荷中继投影本身；
2. `topology_mst_no_tag`：投影图加 55 条候选边，第三个边属性恒为 0；
3. `topology_mst_tag`：同样加候选边，第三个边属性编码推断来源。

若 MST 没有稳定增益，将其降级为拓扑敏感性分析；若 `mst_tag` 不优于 `mst_no_tag`，删除“边来源标记贡献”的主张。

## 4. PA-STFed 数据与模型实现

输入张量为 (mathbf{X}\in\mathbb{R}^{B\times96\times N\times5})，其中 1 个通道为节点级 robust 归一化负荷，4 个通道为序列内部的日/周正余弦周期。目标为 (mathbf{Y}\in\mathbb{R}^{B\times12\times N})。模型前向位于 `model.py:477--600`：

1. `input_projection` 将 5 维特征映射到隐藏空间；
2. `physical` 使用边属性和邻接掩码执行边感知空间注意力；
3. `functional` 使用每客户端静态可学习关系矩阵
   \[
   \mathbf{A}^{A}_k=\operatorname{softmax}\left(\operatorname{ReLU}(\mathbf{E}_{k,1}\mathbf{E}_{k,2}^{\top})\right);
   \]
4. `spatial_gate` 对物理/功能表示做一级门控，`temporal` 为历史序列 Transformer，`temporal_gate` 做二级时空融合；
5. `head` 输出未来 12 步，残差锚定仅在显式消融中开启。

客户端仅有 10--13 个有效节点，静态功能图把逐时刻动态 TopK 构图从 (mathcal{O}(T N_k^2)) 降为一次性的 (mathcal{O}(N_k^2))，避免微型子图上的结构抖动和过拟合。该模块是否保留必须由 `spatial_physical_only`、`spatial_functional_only` 和 `spatial_dual_graph` 的结果决定。

本地训练使用 Charbonnier 损失（`federated.py:22`），小残差区近似二次且梯度连续，大残差区近似线性，适合含尖峰的负荷序列。所有模型统一在反归一化后的原始负荷尺度计算 WAPE、sMAPE、RMSE、MAE 和低负荷阈值 MAPE；MAPE 同时报告有效覆盖率。

## 5. 联邦参数共享机制

参数分组由 `model.py:17--23`、`model.py:603--621` 控制，训练/聚合由 `federated.py:81--219` 和 `run.py:1461--1900` 控制。

| 参数/对象 | FedAvg/FedProx | Personalized | 说明 |
|---|---|---|---|
| `input_projection.*`、`physical.*` | 服务器聚合 | 服务器聚合 | 通用输入和物理图编码 |
| `functional.value.*` | 服务器聚合 | 服务器聚合 | 功能图消息变换 |
| `spatial_gate.*`、`temporal.*`、`temporal_gate.*` | 服务器聚合 | 服务器聚合 | 空间/时间融合 |
| `head.*` | 服务器聚合 | **客户端本地** | 个性化预测头消融 |
| `functional.embedding_1/2` | **客户端本地** | **客户端本地** | 节点数不同，形状不能直接平均 |
| 图张量、节点划分、原始窗口、归一化统计量 | 不上传 | 不上传 | 非模型状态，本地构造 |

当前 PA-STFed 有 47 个 `state_dict` 条目：FedAvg/FedProx 聚合 45 个，功能图嵌入 2 个始终本地；`personalized_head=true` 时再把 6 个 `head.*` 条目留在本地，服务器聚合 39 个。上传的是模型状态/更新，不是负荷窗口。FedAvg/FedProx 只能表述为数据留域的协同训练协议，不能宣称严格隐私、加密安全或差分隐私；只有 `dp_fedavg` 显式开启时才应用客户端更新裁剪和中心高斯噪声。

## 6. 三组核心实验配置

### 6.1 拓扑来源组

`topology_forest`、`topology_mst_no_tag`、`topology_mst_tag`，集中式 PA-STFed，`history=96`、`horizon=12`、`hidden_dim=64`、`spatial_heads=4`、`transformer_layers=2`、`transformer_heads=4`、`dropout=0.10`、`AdamW lr=0.002`、`weight_decay=1e-4`、最大 30 epoch、早停 patience=7，种子 2026/2027/2028。

### 6.2 双图互补组

固定 `graph=mst_tag`，比较 `spatial_physical_only`、`spatial_functional_only`、`spatial_dual_graph`。单分支实验关闭对应空间门控，保证比较的是信息来源而非额外门控参数。主比较指标为节点 micro-WAPE 和客户端宏平均 WAPE，辅以 RMSE、MAE、sMAPE、MAPE 及门控/CKA 诊断。

### 6.3 联邦协同组

固定 `graph=mst_tag` 和 PA-STFed，比较 `local_only`、`fedavg`、`fedprox_005`、`personalized_head`。8 个客户端全参与、`local_epochs=1`、最大 20 轮、早停 patience=5；默认 `uniform_mean=true`，FedProx 使用 `mu=0.005`。报告全局 micro-WAPE、客户端 macro-WAPE、8 个客户端原始指标、均值、标准差、最差客户端、P90/P95。

## 7. 统一评价与统计协议

- 固定时间顺序 80%/10%/10%，当前边界为 train=28032、validation=31536、total=35040；窗口数为 27925/3493/3493。
- 节点集合仅由训练期稳定有效节点确认；当前训练期与全序列均为 92 个有效节点。
- 验证集选择 checkpoint，测试集仅在配置锁定后由 `*_test` 任务访问。
- `evaluate_wape_blocks()` 保存每个连续 96 个 rolling origin 的误差和与目标和，并记录尾块窗口数。相邻 rolling origin 存在重叠，因此 Bootstrap 解释为固定长度时间块的近似推断，不视为独立样本。
- 集中式和联邦跨模式比较使用全局 `micro_global` 区块；联邦客户端 `macro_client` 仅用于异构性描述。`analyze_results.py` 默认 `--aggregation micro`，跨源、历史长度、预测长度、节点集合、客户端划分或 origin 哈希不一致时拒绝运行。
- 多种子先逐 seed 配对，再报告 seed 间均值/标准差；不能把不同种子的时间块直接拼成独立样本池。多重两两比较应预先指定主比较或采用 Holm 校正。
- 所有 DP 表必须记录 `(C, sigma, q, R, delta, epsilon)` 和 `privacy.py` 会计版本。当前 DP 机制为全参与 `q=1`，均值噪声方差为 \(\sigma^2C^2/|\mathcal{S}_r|^2\)。

## 8. 推荐执行顺序与计算量

| 阶段 | 任务 | 运行规模 | 决策门 |
|---|---|---:|---|
| M0 | `audit`、编译、三组单种子烟测 | 分钟级 | 任一拓扑/窗口/指标错误即停止 |
| M1 | 集中式主模型与 LSTM、iTransformer、AGCRN-adapted、Graph WaveNet-adapted | 5 模型 × 3 seed | 确认均为 v2 且验证口径一致 |
| M2 | 拓扑三变体 | 3 × 3 seed | MST/来源标记无稳定增益则降级 |
| M3 | 双图三变体 | 3 × 3 seed | 双图无增益则删去双图主张 |
| M4 | LocalOnly/FedAvg/FedProx/Personalized | 4 × 3 seed | 观察 micro 与尾部客户端，不预设联邦更优 |
| M5 | Bootstrap、门控饱和率、客户端尾部图表 | 不训练 | 只输出证据支持的比较措辞 |
| M6 | 锁参后的集中式/FedAvg/必要时 Personalized 测试 | 3 × 3 seed | 测试集只访问一次 |

默认 `all` 只运行简单基线、集中式 PA、LocalOnly、FedAvg、FedProx、个性化头，共 16 个种子任务；拓扑、双图、额外基线、DP 和最终测试必须显式指定。集中式任务通常比联邦任务便宜；联邦任务的主要成本近似为“20 轮 × 8 客户端 × 本地窗口迭代”。应先用一个种子烟测确认吞吐和显存，再提交三种子批量任务，不要在证据不足时扩大模型或种子数量。

## 9. 论文主张边界与失败处理

最可能成立的贡献（证据等级 C，待 B/D 级实验验证）：

> 在拓扑信息不完备、零负荷节点承担中继作用且客户端负荷存在空间异构的 SmartDS 合成配电网中，构建可审计的零负荷中继投影—有效节点候选桥接—边来源感知建模流程，并比较物理关系、客户端本地功能关系与部分个性化联邦共享的作用边界。

不得写成“首次使用联邦学习/双图/MST 补边”或“实现隐私保护”。若 `mst_tag`、双图、FedProx 或个性化头没有稳定增益，分别降级为敏感性分析、实现细节或对照方法；若 PA-STFed 不优于强基线，则停止继续堆叠模块，把论文定位为“拓扑不完备与空间 Non-IID 条件下的可审计预测验证框架”，如实报告系统级与客户端级误差。

## 10. 当前状态

- 数据审计：完成（A 级事实）。
- 拓扑投影与有效节点 MST：已实现并通过审计（A/C 级）。
- 联邦参数边界：已实现并输出 `outputs/federated_parameter_groups.csv`（A 级代码事实）。
- 结果对齐与 micro/macro 统计：已实现，尚无新 v2 训练结果（C 级实现，B/D 级证据缺失）。
- 旧验证结果：仅作历史档案，不能支持修正方案优于基线的结论。
- 下一步：按 M0 -> M1 -> M2 -> M3 -> M4 -> M5 -> M6 执行；每组完成后再决定是否保留对应论文主张。
