# SmartDS 数据修复工作区

本目录是 PA-STFed 旧训练工程之外的独立数据审计与重建工作区。旧文件
`E:/pa_stfed_project/smartds_graph.npz`、`outputs/`、checkpoint、日志和
配置均不在本工作区内修改。

## 目录

```text
data-rebuild/
├── data/
│   ├── legacy/smartds_graph_legacy.npz   # 旧 NPZ 的只读封存副本
│   ├── raw/SMARTDS/                      # 官方 SMART-DS/OpenDSS 原始文件、profile 和最小 parquet
│   └── processed/                        # blocker 与官方独立候选分目录保存
├── preprocess/
│   ├── audit_smartds_data.py
│   ├── audit_official_load_data.py
│   ├── build_smartds_graph_v2.py
│   ├── build_official_load_series.py
│   ├── verify_official_raw.py
│   ├── fetch_smartds_profiles.py
│   ├── write_manifests.py
│   └── validate_smartds_v2.py
└── reports/
```

## 运行顺序（仅数据审计，不启动模型）

首次运行前安装审计依赖：

```powershell
python -m pip install -r .\requirements-audit.txt
```

```powershell
cd E:\pa_stfed_project\data-rebuild
python .\preprocess\fetch_smartds_profiles.py `
  --raw-root .\data\raw\SMARTDS\candidate_v09_2018_P10R_p10rdt7719_source `
  --feeder-root p10rhs0_1247/p10rhs0_1247--p10rdt7719 `
  --include-load-data
python .\preprocess\verify_official_raw.py `
  --raw-root .\data\raw\SMARTDS\candidate_v09_2018_P10R_p10rdt7719_source `
  --feeder-root p10rhs0_1247/p10rhs0_1247--p10rdt7719 `
  --report .\reports\official_raw_remote_verification.json
python .\preprocess\audit_official_load_data.py `
  --legacy .\data\legacy\smartds_graph_legacy.npz `
  --raw-root .\data\raw\SMARTDS\candidate_v09_2018_P10R_p10rdt7719_source `
  --reports .\reports\candidate_v09_2018_P10R_p10rdt7719 `
  --feeder-root p10rhs0_1247/p10rhs0_1247--p10rdt7719
python .\preprocess\audit_smartds_data.py `
  --raw-root .\data\raw\SMARTDS\candidate_v09_2018_P10R_p10rdt7719_source `
  --reports .\reports\candidate_v09_2018_P10R_p10rdt7719 `
  --feeder-root p10rhs0_1247/p10rhs0_1247--p10rdt7719
python .\preprocess\build_smartds_graph_v2.py `
  --raw-root .\data\raw\SMARTDS\candidate_v09_2018_P10R_p10rdt7719_source `
  --output .\data\processed\candidate_v09_2018_P10R_p10rdt7719 `
  --reports .\reports\candidate_v09_2018_P10R_p10rdt7719 `
  --feeder-root p10rhs0_1247/p10rhs0_1247--p10rdt7719
python .\preprocess\validate_smartds_v2.py `
  --raw-root .\data\raw\SMARTDS\candidate_v09_2018_P10R_p10rdt7719_source `
  --processed .\data\processed\candidate_v09_2018_P10R_p10rdt7719 `
  --feeder-root p10rhs0_1247/p10rhs0_1247--p10rdt7719
```

旧 NPZ 的数值映射若未通过，上述 legacy 路径预期返回 `FAIL`。如需构建
不依赖 legacy 数值的官方序列候选，使用独立输出目录：

```powershell
python .\preprocess\build_official_load_series.py `
  --raw-root .\data\raw\SMARTDS\candidate_v09_2018_P10R_p10rdt7719_source `
  --output .\data\processed\official_v1_2018_P10R_p10rdt7719 `
  --reports .\reports\official_v1_2018_P10R_p10rdt7719 `
  --feeder-root p10rhs0_1247/p10rhs0_1247--p10rdt7719
python .\preprocess\validate_smartds_v2.py `
  --raw-root .\data\raw\SMARTDS\candidate_v09_2018_P10R_p10rdt7719_source `
  --processed .\data\processed\official_v1_2018_P10R_p10rdt7719 `
  --feeder-root p10rhs0_1247/p10rhs0_1247--p10rdt7719 `
  --report .\reports\official_v1_2018_P10R_p10rdt7719\validation_report.json
```

legacy 审计状态为 `BLOCKED_LOAD_SERIES_MAPPING_UNVERIFIED` 时，旧流程只写入
blocker 元数据并隔离 provisional NPZ。官方独立构建流程输出的
`official_v1_2018_P10R_p10rdt7719` 已通过自身一致性闸门，但不代表 legacy
`load_ts` 已被追溯；正式模型 loader 必须显式选择官方候选路径，不能静默回退
到 `E:/pa_stfed_project/smartds_graph.npz`。

## 当前官方 scope

`SMART-DS v0.9 / 2018 / Full_Texas / P10R / base_timeseries`，substation
`p10rhs0_1247`，feeder `p10rhs0_1247--p10rdt7719`。官方用户指南
来源为 <https://github.com/openEDI/documentation/blob/main/SMART-DS/Readme.md>；
OEDI 对象存储前缀、文件 SHA256 和 profile 清单见 `data/raw/SMARTDS`。

## 拓扑规则

canonical physical graph 只能来自官方 OpenDSS `Line`、`Transformer` 等
明确设备定义；每条边必须保留 `edge_type` 和来源文件/行号。旧 NPZ 邻接矩阵、
节点名称模式和 Euclidean MST 只允许作为历史诊断或敏感性分析，不能进入
canonical topology。全年零负荷节点不做插值；在官方证据完成前不删除、不改写。

审计结论汇总见 `reports/data_audit_decision.md`；当前 legacy 状态为
`BLOCKED_LOAD_SERIES_MAPPING_UNVERIFIED`。官方独立候选的状态和校验见
`reports/official_v1_2018_P10R_p10rdt7719/official_series_rebuild.md` 及其
`validation_report.json`；本任务完成前仍不得启动正式训练。
