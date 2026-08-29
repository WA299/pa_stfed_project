# Current Stage

Formal experiment stage 0: target topology and client partition freeze.

# Data Status

`data/processed/smartds_full_graph_v2.npz` 已通过一致性验证；8-client 官方树切边划分及 k=2/4/6/8 拓扑诊断已完成记录。

# Model Status

PA-STFed 代码可解析配置和 dry-run，尚未基于 canonical 数据重新开展正式训练。

# Current Blocker

Topology-kNN 的 k 尚未选择；Stage 0 仅记录候选图和客户端连通性，不使用模型精度作决定。

# Next Steps

1. 冻结 full-graph 到模型输入的映射规则。
2. 运行小规模非正式 smoke test。
3. 再确定正式多随机种子实验矩阵。

# Latest Commit

本文件所在提交；整理前备份为 `4d9656941e56cf9d3c79211c737f080507eaa645`。
