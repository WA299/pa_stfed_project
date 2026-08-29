# Current Stage

正在冻结官方 SMART-DS canonical 数据口径，并确认 PA-STFed 的物理图输入设计。

# Data Status

`data/processed/smartds_full_graph_v2.npz` 为当前唯一正式候选：273 个节点、272 条官方物理边、92 个预测节点；本地一致性验证已建立。

# Model Status

PA-STFed 代码可解析配置和 dry-run，尚未基于 canonical 数据重新开展正式训练。

# Current Blocker

92 个目标节点在完整物理图上的消息传播表示仍需冻结；旧负荷序列与官方重建序列不一致，旧结果不可复用。

# Next Steps

1. 冻结 full-graph 到模型输入的映射规则。
2. 运行小规模非正式 smoke test。
3. 再确定正式多随机种子实验矩阵。

# Latest Commit

本文件所在提交；整理前备份为 `4d9656941e56cf9d3c79211c737f080507eaa645`。
