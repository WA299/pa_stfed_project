# Current Stage

ModuleALA adaptive local aggregation performance path is validated with 2-round runtime smoke, inspired by/adapted from the FedALA mechanism; no formal long run was started.

# Data Status

共同 timestamp indexing 审计为 PASS：61 个官方 parquet 均给出 35040 个连续 15 分钟区间末时刻，canonical `load_ts` 可由官方 LoadShapes 精确重建。3 条商业 profile/parquet 数值不一致作为 caveat 保留。`smartds_calendar_v1.npz` 单独保存 timestamp、hour/day/weekend/month；temperature 标记为 unavailable。

# Model Status

模型主体、topology-kNN、客户端划分和既有 FedAvg/FedProx 算法未修改。正式 loader 输入冻结为 historical load + relative daily phase sin/cos + relative weekly phase sin/cos，共 5 维（`input_dim=5`）。ModuleALA 的 functional embeddings 始终本地，spatial/temporal gate 与 head 使用逐元素 alpha 个性化，第一轮关闭 ALA；不声称提出或复现原始 FedALA。ALA adaptation 现在对 train 滑窗做按全时间范围的确定性分层抽样，每 client/round 最多 2048 窗口，并缓存 local/global/static state，forward 使用与正式训练一致的 AMP。ModuleALA 与 VanillaFedALA 2-round runtime smoke 已通过，alpha 合法、global ALA 参数参与聚合、未访问 test。

# Current Blocker

无实现阻塞。Temperature 不可用且不再获取；不得用外部 NSRDB 或其他地区天气替代。verified timestamp/calendar sidecar 与审计报告继续保留，但 month/weekend 等额外日历变量不进入正式主模型。3 条 commercial profile/parquet 数值例外必须在论文数据说明中保留。

# Next Steps

1. 保持 ModuleALA、ModuleLocal 与 VanillaFedALA smoke/benchmark 结果仅作流程审计。
2. 将 8D calendar screening 与所有 personalization smoke 结果排除出正式主表。
3. 在明确下一阶段协议后再运行正式长训练。

# Latest Commit

本文件所在提交；整理前备份为 `4d9656941e56cf9d3c79211c737f080507eaa645`。
