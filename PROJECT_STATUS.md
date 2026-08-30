# Current Stage

Stage 2B centralized validation seeds 2026/2027/2028 completed.

# Data Status

Stage 0 已冻结 topology-based 8-client partition 与全局 symmetric topology-kNN（k=6）。Stage 2A/2B 已完成 centralized validation 对照实验（seeds=2026, 2027, 2028）。

# Model Status

PA-STFed、LSTM、iTransformer、AGCRN、Graph WaveNet、Physical-only、Functional-only 的三随机种子 centralized validation 已完成；未运行 federated 与 test。

# Current Blocker

当前无实现阻塞。Stage 2B 结果为 validation-only，尚不能替代最终 test 结果。

# Next Steps

1. 汇总三 seed validation 的均值、标准差与模型对比。
2. 审核后决定是否进入 federated validation（不改变已冻结拓扑）。
3. 通过审计后再规划 test 阶段。

# Latest Commit

本文件所在提交；整理前备份为 `4d9656941e56cf9d3c79211c737f080507eaa645`。
