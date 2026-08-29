# PA-STFed

项目：基于时空注意力与联邦学习的配电网短期负荷预测。

## Data

当前 canonical 数据来自 SMART-DS v0.9 的 2018 Full_Texas / P10R / `base_timeseries`，feeder 为 `p10rhs0_1247--p10rdt7719`。

- 273 个物理节点
- 216 条 Line 边和 56 条 Transformer 边，共 272 条无向边
- 1 个连通分量
- 92 个负荷预测节点，181 个零负荷结构节点
- 15 分钟采样，35,040 个时间步

旧版 `smartds_graph.npz` 漏掉了 56 条 Transformer 边且负荷序列无法与官方重建值对齐，已退出正式数据口径。当前物理图不使用欧氏 MST 补边。

## Forecast Task

过去 96 步预测未来 12 步，即使用过去 24 小时负荷预测未来 3 小时。

## Model

PA-STFed 包含物理拓扑关系、数据驱动功能关系、时间 Transformer、联邦协同与个性化预测头。当前阶段不对其效果作正式结论。

## Current Status

项目处于 canonical SmartDS 数据验证、负荷来源追溯和图设计确认阶段。正式实验尚未基于新数据重新开始。

## Run

安装依赖后，在项目根目录执行数据验证：

```powershell
python preprocess/validate_smartds_v2.py `
  --processed data/processed `
  --report reports/supporting/validation_report.json
```

只解析实验配置、不加载模型或训练：

```powershell
python code/run.py all --seeds 2026 --dry-run
```

正式训练必须等 canonical 数据与图设计冻结后再启动。
