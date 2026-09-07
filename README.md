# PA-STFed

基于时空注意力与联邦学习的配电网短期负荷预测。

## Scope

- Canonical data: SMART-DS v0.9, 2018 Full_Texas / P10R, feeder `p10rhs0_1247--p10rdt7719`
- 273 physical nodes, 216 Lines, 56 Transformers, 272 edges, 1 connected component
- 92 target load nodes; 15-minute sampling; history 96 -> horizon 12
- Final input: historical load + relative daily/weekly phase sin/cos (5D)

## Current Development Status

The current node-level development candidate is `pa_horizon_decoder_wl1_dev` (seed 2026 validation WAPE approximately 27.9984). RMSE and feeder aggregate WAPE show a trade-off. The next required fairness control is GWN with the same WL1 objective. Final multi-seed confirmation and final test lock have not started.

## Run

```powershell
python preprocess/validate_smartds_v2.py --processed data/processed --report reports/supporting/validation_report.json
python analysis/forecastability_audit.py
python analysis/build_innovation1_ledger.py
python code/run.py all --seeds 2026 --dry-run
```

The analysis commands are read-only; the `run.py` command is a configuration dry-run and does not evaluate test.
