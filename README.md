# PA-STFed 正式实验工程

本工程实现《基于时空注意力与联邦学习的配电网负荷预测》的可复现实验版本。
SmartDS 是合成配电网数据，所有结论必须限定为合成网络验证结果，不得写成真实台区实测结论。

## 文件说明

```text
run.py              正式实验入口、数据审计、训练和结果汇总
analyze_results.py  固定时间块 WAPE 配对 Bootstrap，不重新训练
config.yaml         共享基础配置与运行环境设置
experiments.yaml    当前正式矩阵、消融、DP 和文献基线筛查任务
data.py             SmartDS 读取、零负荷中继投影、MST 候选桥接、时间切分和窗口构造
model.py            PA-STFed、AGCRN、Graph WaveNet、LSTM、iTransformer
federated.py        FedAvg、FedProx、本地训练和 DP 更新聚合
metrics.py          MAE、RMSE、WAPE、sMAPE、MAPE 和门控诊断
privacy.py          L2 更新裁剪、中心高斯噪声和 RDP 会计
smartds_graph.npz   SmartDS 数据文件
outputs/            审计、结果、检查点和汇总文件
```

## 当前模型矩阵

- `Persistence`、`Daily-Naive`：无需训练的预测下限。
- `LSTM`：Kong 等人在 IEEE TSG 2019 的短期住宅负荷预测中采用的序列基线；本工程按节点独立编码。
- `iTransformer`：Liu 等人的 ICLR 2024 倒置 Transformer；将节点×输入特征视为变量 token。
- `AGCRN-adapted`、`Graph WaveNet-adapted`：仅作为显式调用的通用时空图迁移基线，原始论文主要面向交通；本工程是统一输入/输出协议下的适配实现，不声称官方代码逐行复现。
- `PA-STFed`：项目完整模型，集中式与联邦路径共享同一前向定义。

LSTM 是负荷预测方向的直接基线，iTransformer 是现代多变量时间序列基线；AGCRN 与 Graph WaveNet 只在需要时显式运行。所有模型使用相同 SmartDS 输入、时间切分、预测窗口和评价指标。完整 DOI 与迁移边界见 `literature/REFERENCES.md`。

## 运行前检查

在项目根目录执行。正式配置要求 CUDA；没有 CUDA 的环境会在训练开始前直接报错，不会静默使用 CPU。

```powershell
python run.py audit
python run.py all --seeds 2026,2027,2028 --dry-run
```

`--dry-run` 只解析并打印任务，不加载数据和训练。确认任务列表与配置无误后，运行非 DP 正式矩阵：

```powershell
python run.py all --seeds 2026,2027,2028 --resume
```

默认矩阵现在只包含 6 类结项主线任务（简单基线、集中式 PA、LocalOnly、FedAvg、FedProx、个性化头），共 16 个种子任务；不会自动运行拓扑/双图消融、额外模型基线、DP 或最终测试。

再运行 DP 对照：

```powershell
python run.py all --seeds 2026,2027,2028 --experiments dp_fedavg --resume
```

服务器中断后仍使用 `--resume`。结果文件含配置指纹，只有配置和代码版本一致才会跳过已有结果。

## 文献基线筛查

LSTM 与 iTransformer 使用三种子完整验证窗口筛查；结果可用于集中式对照，但测试集仍未读取：

```powershell
python run.py all --experiments load_lstm_validation,itransformer_validation --seeds 2026,2027,2028 --resume
```

显式基线：

```powershell
python run.py all --experiments agcrn,gwnet --seeds 2026,2027,2028 --resume
```

拓扑与双图生死实验（建议先用单种子烟测，再提交三种子）：

```powershell
python run.py all --experiments topology_forest,topology_mst_no_tag,topology_mst_tag --seeds 2026 --resume
python run.py all --experiments spatial_physical_only,spatial_functional_only,spatial_dual_graph --seeds 2026 --resume
```

确认烟测无误后，将 `--seeds 2026` 改为 `--seeds 2026,2027,2028`。联邦生死实验使用：

```powershell
python run.py all --experiments local_only,fedavg,fedprox_005,personalized_head --seeds 2026,2027,2028 --resume
```

锁定全部验证配置后，才运行最终测试：

```powershell
python run.py all --experiments centralized_test,fedavg_test,personalized_head_test --seeds 2026,2027,2028 --resume
```

该筛查任务只用于集中式对照，不把 LSTM/iTransformer 接入联邦路径。联邦实现目前只接受 `architecture=pa_stfed`，因为各客户端节点数不同，而 AGCRN/Graph WaveNet/LSTM/iTransformer 的集中式参数形状固定。

## 数据与研究边界

- 主实验固定使用 `graph=mst_tag`：先保留零负荷节点的路径并投影为92节点原始拓扑图，再在56个含负荷分量之间增加55条有效节点候选桥接边。投影图可能因中继块收缩产生环，不把它断言为森林。
- `graph=forest`、`graph=mst_no_tag`、`graph=mst_tag` 是正式拓扑生死实验；`graph=inf`/`legacy_inf` 仅保留历史全节点MST方案作审计对照。
- 原始273节点图有57个连通分量；删除181个零负荷节点后的诱导子图有92个孤立点。零负荷节点中继投影后为56个含负荷分量和54条投影边；正式代码将该图定义为 `projected_raw`，命令兼容别名为 `forest`。
- MST 候选边是可检验的连通性假设，不是 SmartDS 已确认的真实线路；不能解释为导线物理长度、阻抗或潮流关系。
- `d_hop` 和坐标距离只表示图关系与位置先验，不是导线物理长度、阻抗或电气参数。
- 数据没有真实绝对时间戳、天气和节假日字段；周期特征只能表示采样序列内部相位。
- 数据按时间顺序执行 80/10/10 切分，训练、验证、测试不随机混合；`evaluate_test=false` 时测试集不被读取。
- 8 个联邦客户端是合成网络的空间聚合验证，不代表真实独立法人或物理隔离电网。
- 完全相同的 SmartDS 负荷曲线组作为不可拆分单元；当前审计的跨客户端重复组数为 0。

## 输出审计

```text
outputs/audit_report.json       字段、拓扑、切分边界和客户端统计
outputs/topology_comparison.csv 各建图方案的节点、边和连通分量对比
outputs/bridge_edges.csv        修正方案的55条有效节点MST候选边
outputs/legacy_bridge_edges.csv 历史全节点MST的56条桥接边
outputs/federated_parameter_groups.csv 联邦状态参数逐条共享/本地分组
outputs/*_result.json           单个实验的指标、训练历史和隐私记录
outputs/*_model.pt              最佳验证 checkpoint
outputs/all_manifest.json       all 模式任务状态、跳过项和错误记录
outputs/all_summary.json        多随机种子均值、标准差和样本数
```

结果 JSON 还会记录 SmartDS 源文件 SHA-256、`history`/`horizon`、有效节点集合哈希、
验证/测试窗口起点哈希和区块长度。联邦结果同时保存：

- `validation_wape_blocks_micro` / `test_wape_blocks_micro`：跨客户端累加误差和目标和后得到的全局 micro-WAPE，可与集中式结果配对；
- `validation_wape_blocks_clients` / `test_wape_blocks_clients`：每个客户端的区块统计量，用于客户端 macro-WAPE 和尾部描述统计。

配对检验默认只允许严格对齐的全局 micro-WAPE：

```powershell
python analyze_results.py --left outputs/<left>_centralized_result.json `
  --right outputs/<right>_federated_result.json --split validation --aggregation micro
```

如需比较客户端等权误差，显式使用 `--aggregation macro`；该结果是 8 个合成客户端的
描述性统计，不应解释为真实总体分位数或隐私证明。

DP 实验结果必须同时保留 `(C, sigma, q, R, delta, epsilon)`；任何“优于”结论都应基于多随机种子结果及配对 Bootstrap 置信区间，而不是单次运行的最优轮次。
