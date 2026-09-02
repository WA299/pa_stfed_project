# Forecastability Audit

Validation-only; no training, no test loader, and no existing result JSON was modified.

## Metric Sanity

- PA-STFed: recomputed WAPE/MAE/RMSE = {'wape': 29.373682022094727, 'mae': 0.34444817900657654, 'rmse': 0.714534342288971}; max difference = 0.00000381 (PASS)
- GWN: recomputed WAPE/MAE/RMSE = {'wape': 29.005434036254883, 'mae': 0.34012991189956665, 'rmse': 0.713005781173706}; max difference = 0.00000000 (PASS)

## Horizon-wise Metrics

| Method | Horizon | Node-micro WAPE | MAE | RMSE |
|---|---:|---:|---:|---:|
| PA-STFed | h1 | 18.3582 | 0.215099 | 0.457934 |
| PA-STFed | h3 | 23.7940 | 0.278829 | 0.600959 |
| PA-STFed | h6 | 26.6515 | 0.312388 | 0.669092 |
| PA-STFed | h12 | 29.3737 | 0.344448 | 0.714534 |
| GWN | h1 | 17.3894 | 0.203748 | 0.457321 |
| GWN | h3 | 23.1853 | 0.271697 | 0.600164 |
| GWN | h6 | 26.2692 | 0.307907 | 0.668225 |
| GWN | h12 | 29.0054 | 0.340130 | 0.713006 |
| Persistence | h1 | 18.0189 | 0.211123 | 0.516695 |
| Persistence | h3 | 26.7421 | 0.313377 | 0.724909 |
| Persistence | h6 | 33.3026 | 0.390347 | 0.862369 |
| Persistence | h12 | 40.9818 | 0.480569 | 0.991299 |
| Daily-lag naive | h1 | 48.0422 | 0.562900 | 1.092739 |
| Daily-lag naive | h3 | 48.0465 | 0.563032 | 1.092940 |
| Daily-lag naive | h6 | 48.0525 | 0.563233 | 1.093192 |
| Daily-lag naive | h12 | 48.0583 | 0.563552 | 1.093594 |

## Aggregation Effect

| Method | Horizon | Node WAPE | Feeder aggregate WAPE |
|---|---:|---:|---:|
| PA-STFed | h1 | 18.3582 | 4.7410 |
| PA-STFed | h3 | 23.7940 | 6.8742 |
| PA-STFed | h6 | 26.6515 | 8.1472 |
| PA-STFed | h12 | 29.3737 | 8.8903 |
| GWN | h1 | 17.3894 | 5.0082 |
| GWN | h3 | 23.1853 | 6.9185 |
| GWN | h6 | 26.2692 | 8.0402 |
| GWN | h12 | 29.0054 | 8.7677 |
| Persistence | h1 | 18.0189 | 5.2410 |
| Persistence | h3 | 26.7421 | 7.5164 |
| Persistence | h6 | 33.3026 | 9.9025 |
| Persistence | h12 | 40.9818 | 14.1932 |
| Daily-lag naive | h1 | 48.0422 | 16.0472 |
| Daily-lag naive | h3 | 48.0465 | 16.0453 |
| Daily-lag naive | h6 | 48.0525 | 16.0437 |
| Daily-lag naive | h12 | 48.0583 | 16.0348 |

## Node Difficulty

- PA-STFed: {'min': 2.762939929962158, 'p10': 12.90864372253418, 'p25': 25.412681579589844, 'median': 29.858762741088867, 'p75': 40.6666374206543, 'p90': 46.49767303466797, 'max': 49.01484680175781}; counts={'lt10': 8, 'lt20': 11, 'lt30': 47, 'ge40': 29}
- GWN: {'min': 3.839329719543457, 'p10': 14.086502075195312, 'p25': 23.372276306152344, 'median': 28.794334411621094, 'p75': 40.738258361816406, 'p90': 47.29705810546875, 'max': 49.12572479248047}; counts={'lt10': 8, 'lt20': 12, 'lt30': 49, 'ge40': 27}
- Persistence: {'min': 4.913691997528076, 'p10': 19.870603561401367, 'p25': 35.058555603027344, 'median': 44.3123779296875, 'p75': 54.34208679199219, 'p90': 66.28288269042969, 'max': 92.21017456054688}; counts={'lt10': 5, 'lt20': 10, 'lt30': 11, 'ge40': 56}
- Daily-lag naive: {'min': 7.026316165924072, 'p10': 19.96192169189453, 'p25': 39.092445373535156, 'median': 51.287315368652344, 'p75': 60.653995513916016, 'p90': 73.21785736083984, 'max': 95.0457534790039}; counts={'lt10': 3, 'lt20': 10, 'lt30': 13, 'ge40': 67}

## Forecastability Correlations

- PA-STFed: {'train_cv': 0.4158354257798238, 'lag96_autocorr': -0.6539908795744196, 'lag672_autocorr': -0.6765005554040376, 'mean_shift': -0.33968796761373926}
- GWN: {'train_cv': 0.37324434394467326, 'lag96_autocorr': -0.6908466193488729, 'lag672_autocorr': -0.7098102503560169, 'mean_shift': -0.35916808775806963}
- Persistence: {'train_cv': 0.5171424605561467, 'lag96_autocorr': -0.5772605993306497, 'lag672_autocorr': -0.5890282083314053, 'mean_shift': -0.3037523712580391}
- Daily-lag naive: {'train_cv': 0.4725705208285136, 'lag96_autocorr': -0.6697049615200726, 'lag672_autocorr': -0.7266614229090516, 'mean_shift': -0.506809173491263}

## Conclusions

1. PA-STFed node-micro WAPE is 29.37% and feeder-aggregate WAPE is 8.89%; this quantifies the aggregation-level effect.
2. PA-STFed h12 minus h1 WAPE is 11.02 percentage points, so horizon degradation is present.
3. The reported Spearman correlations quantify whether high error tracks CV, autocorrelation, or mean shift; no causal claim is made.
4. PA-STFed versus GWN h12 WAPE gap is 0.37 percentage points; horizon-wise and node-level tables above show where it concentrates.
