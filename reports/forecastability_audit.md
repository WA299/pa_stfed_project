# Forecastability Audit

Validation-only; no training, no test loader, and no existing result JSON was modified.

## Metric Sanity

- PA-STFed: recomputed WAPE/MAE/RMSE = {'wape': 29.373682022094727, 'mae': 0.34444817900657654, 'rmse': 0.714534342288971}; max difference = 0.00000381 (PASS)
- GWN: recomputed WAPE/MAE/RMSE = {'wape': 29.005434036254883, 'mae': 0.34012991189956665, 'rmse': 0.713005781173706}; max difference = 0.00000000 (PASS)
- PA-STFed residual-anchor: recomputed WAPE/MAE/RMSE = {'wape': 28.996524810791016, 'mae': 0.34002548456192017, 'rmse': 0.720468282699585}; max difference = 0.00000381 (PASS)

## Horizon-wise Metrics

prefix_h* are cumulative prefix metrics; step* are exact forecast steps.

| Method | Horizon | Node-micro WAPE | MAE | RMSE |
|---|---:|---:|---:|---:|
| PA-STFed | prefix_h1 | 18.3582 | 0.215099 | 0.457934 |
| PA-STFed | prefix_h3 | 23.7940 | 0.278829 | 0.600959 |
| PA-STFed | prefix_h6 | 26.6515 | 0.312388 | 0.669092 |
| PA-STFed | prefix_h12 | 29.3737 | 0.344448 | 0.714534 |
| PA-STFed | step1 | 18.3582 | 0.215099 | 0.457934 |
| PA-STFed | step3 | 27.5897 | 0.323354 | 0.692661 |
| PA-STFed | step6 | 30.1749 | 0.353822 | 0.740059 |
| PA-STFed | step12 | 33.3626 | 0.391552 | 0.769210 |
| PA-STFed | overall_12step | 29.3737 | 0.344448 | 0.714534 |
| PA-STFed residual-anchor | prefix_h1 | 17.6575 | 0.206889 | 0.464668 |
| PA-STFed residual-anchor | prefix_h3 | 23.2519 | 0.272477 | 0.606310 |
| PA-STFed residual-anchor | prefix_h6 | 26.2772 | 0.308001 | 0.674325 |
| PA-STFed residual-anchor | prefix_h12 | 28.9965 | 0.340025 | 0.720468 |
| PA-STFed residual-anchor | step1 | 17.6575 | 0.206889 | 0.464668 |
| PA-STFed residual-anchor | step3 | 27.2242 | 0.319070 | 0.697521 |
| PA-STFed residual-anchor | step6 | 29.9599 | 0.351301 | 0.745138 |
| PA-STFed residual-anchor | step12 | 32.8840 | 0.385935 | 0.775833 |
| PA-STFed residual-anchor | overall_12step | 28.9965 | 0.340025 | 0.720468 |
| GWN | prefix_h1 | 17.3894 | 0.203748 | 0.457321 |
| GWN | prefix_h3 | 23.1853 | 0.271697 | 0.600164 |
| GWN | prefix_h6 | 26.2692 | 0.307907 | 0.668225 |
| GWN | prefix_h12 | 29.0054 | 0.340130 | 0.713006 |
| GWN | step1 | 17.3894 | 0.203748 | 0.457321 |
| GWN | step3 | 27.3556 | 0.320610 | 0.692143 |
| GWN | step6 | 30.0644 | 0.352527 | 0.740252 |
| GWN | step12 | 32.8986 | 0.386107 | 0.765072 |
| GWN | overall_12step | 29.0054 | 0.340130 | 0.713006 |
| Persistence | prefix_h1 | 18.0189 | 0.211123 | 0.516695 |
| Persistence | prefix_h3 | 26.7421 | 0.313377 | 0.724909 |
| Persistence | prefix_h6 | 33.3026 | 0.390347 | 0.862369 |
| Persistence | prefix_h12 | 40.9818 | 0.480569 | 0.991299 |
| Persistence | step1 | 18.0189 | 0.211123 | 0.516695 |
| Persistence | step3 | 33.6876 | 0.394822 | 0.865190 |
| Persistence | step6 | 42.1124 | 0.493798 | 1.017970 |
| Persistence | step12 | 52.5797 | 0.617089 | 1.154756 |
| Persistence | overall_12step | 40.9818 | 0.480569 | 0.991299 |
| Daily-lag naive | prefix_h1 | 48.0422 | 0.562900 | 1.092739 |
| Daily-lag naive | prefix_h3 | 48.0465 | 0.563032 | 1.092940 |
| Daily-lag naive | prefix_h6 | 48.0525 | 0.563233 | 1.093192 |
| Daily-lag naive | prefix_h12 | 48.0583 | 0.563552 | 1.093594 |
| Daily-lag naive | step1 | 48.0422 | 0.562900 | 1.092739 |
| Daily-lag naive | step3 | 48.0506 | 0.563157 | 1.093112 |
| Daily-lag naive | step6 | 48.0601 | 0.563539 | 1.093551 |
| Daily-lag naive | step12 | 48.0635 | 0.564086 | 1.094253 |
| Daily-lag naive | overall_12step | 48.0583 | 0.563552 | 1.093594 |
| Weekly-lag naive | prefix_h1 | 57.4112 | 0.672675 | 1.267777 |
| Weekly-lag naive | prefix_h3 | 57.4145 | 0.672811 | 1.267937 |
| Weekly-lag naive | prefix_h6 | 57.4198 | 0.673030 | 1.268259 |
| Weekly-lag naive | prefix_h12 | 57.4246 | 0.673385 | 1.268739 |
| Weekly-lag naive | step1 | 57.4112 | 0.672675 | 1.267777 |
| Weekly-lag naive | step3 | 57.4187 | 0.672953 | 1.268111 |
| Weekly-lag naive | step6 | 57.4267 | 0.673369 | 1.268754 |
| Weekly-lag naive | step12 | 57.4229 | 0.673931 | 1.269427 |
| Weekly-lag naive | overall_12step | 57.4246 | 0.673385 | 1.268739 |
| Daily-weekly blend alpha=0.00 | prefix_h1 | 57.4112 | 0.672675 | 1.267777 |
| Daily-weekly blend alpha=0.00 | prefix_h3 | 57.4145 | 0.672811 | 1.267937 |
| Daily-weekly blend alpha=0.00 | prefix_h6 | 57.4198 | 0.673030 | 1.268259 |
| Daily-weekly blend alpha=0.00 | prefix_h12 | 57.4246 | 0.673385 | 1.268739 |
| Daily-weekly blend alpha=0.00 | step1 | 57.4112 | 0.672675 | 1.267777 |
| Daily-weekly blend alpha=0.00 | step3 | 57.4187 | 0.672953 | 1.268111 |
| Daily-weekly blend alpha=0.00 | step6 | 57.4267 | 0.673369 | 1.268754 |
| Daily-weekly blend alpha=0.00 | step12 | 57.4229 | 0.673931 | 1.269427 |
| Daily-weekly blend alpha=0.00 | overall_12step | 57.4246 | 0.673385 | 1.268739 |
| Daily-weekly blend alpha=0.25 | prefix_h1 | 50.6080 | 0.592963 | 1.096550 |
| Daily-weekly blend alpha=0.25 | prefix_h3 | 50.6118 | 0.593093 | 1.096696 |
| Daily-weekly blend alpha=0.25 | prefix_h6 | 50.6178 | 0.593301 | 1.097001 |
| Daily-weekly blend alpha=0.25 | prefix_h12 | 50.6237 | 0.593635 | 1.097460 |
| Daily-weekly blend alpha=0.25 | step1 | 50.6080 | 0.592963 | 1.096550 |
| Daily-weekly blend alpha=0.25 | step3 | 50.6162 | 0.593227 | 1.096855 |
| Daily-weekly blend alpha=0.25 | step6 | 50.6255 | 0.593620 | 1.097458 |
| Daily-weekly blend alpha=0.25 | step12 | 50.6261 | 0.594161 | 1.098141 |
| Daily-weekly blend alpha=0.25 | overall_12step | 50.6237 | 0.593635 | 1.097460 |
| Daily-weekly blend alpha=0.50 | prefix_h1 | 46.5071 | 0.544913 | 0.999306 |
| Daily-weekly blend alpha=0.50 | prefix_h3 | 46.5114 | 0.545043 | 0.999455 |
| Daily-weekly blend alpha=0.50 | prefix_h6 | 46.5180 | 0.545247 | 0.999743 |
| Daily-weekly blend alpha=0.50 | prefix_h12 | 46.5249 | 0.545570 | 1.000183 |
| Daily-weekly blend alpha=0.50 | step1 | 46.5071 | 0.544913 | 0.999306 |
| Daily-weekly blend alpha=0.50 | step3 | 46.5160 | 0.545172 | 0.999609 |
| Daily-weekly blend alpha=0.50 | step6 | 46.5266 | 0.545558 | 1.000163 |
| Daily-weekly blend alpha=0.50 | step12 | 46.5306 | 0.546095 | 1.000863 |
| Daily-weekly blend alpha=0.50 | overall_12step | 46.5249 | 0.545570 | 1.000183 |
| Daily-weekly blend alpha=0.75 | prefix_h1 | 45.1847 | 0.529420 | 0.997913 |
| Daily-weekly blend alpha=0.75 | prefix_h3 | 45.1894 | 0.529551 | 0.998082 |
| Daily-weekly blend alpha=0.75 | prefix_h6 | 45.1963 | 0.529755 | 0.998351 |
| Daily-weekly blend alpha=0.75 | prefix_h12 | 45.2035 | 0.530075 | 0.998771 |
| Daily-weekly blend alpha=0.75 | step1 | 45.1847 | 0.529420 | 0.997913 |
| Daily-weekly blend alpha=0.75 | step3 | 45.1941 | 0.529680 | 0.998242 |
| Daily-weekly blend alpha=0.75 | step6 | 45.2051 | 0.530061 | 0.998736 |
| Daily-weekly blend alpha=0.75 | step12 | 45.2109 | 0.530607 | 0.999443 |
| Daily-weekly blend alpha=0.75 | overall_12step | 45.2035 | 0.530075 | 0.998771 |
| Daily-weekly blend alpha=1.00 | prefix_h1 | 48.0422 | 0.562900 | 1.092739 |
| Daily-weekly blend alpha=1.00 | prefix_h3 | 48.0465 | 0.563032 | 1.092940 |
| Daily-weekly blend alpha=1.00 | prefix_h6 | 48.0525 | 0.563233 | 1.093192 |
| Daily-weekly blend alpha=1.00 | prefix_h12 | 48.0583 | 0.563552 | 1.093594 |
| Daily-weekly blend alpha=1.00 | step1 | 48.0422 | 0.562900 | 1.092739 |
| Daily-weekly blend alpha=1.00 | step3 | 48.0506 | 0.563157 | 1.093112 |
| Daily-weekly blend alpha=1.00 | step6 | 48.0601 | 0.563539 | 1.093551 |
| Daily-weekly blend alpha=1.00 | step12 | 48.0635 | 0.564086 | 1.094253 |
| Daily-weekly blend alpha=1.00 | overall_12step | 48.0583 | 0.563552 | 1.093594 |

## Aggregation Effect

Prefix and exact-step feeder aggregate WAPE are both reported.

| Method | Horizon | Node WAPE | Feeder aggregate WAPE |
|---|---:|---:|---:|
| PA-STFed | prefix_h1 | 18.3582 | 4.7410 |
| PA-STFed | prefix_h3 | 23.7940 | 6.8742 |
| PA-STFed | prefix_h6 | 26.6515 | 8.1472 |
| PA-STFed | prefix_h12 | 29.3737 | 8.8903 |
| PA-STFed | step1 | 18.3582 | 4.7410 |
| PA-STFed | step3 | 27.5897 | 8.7947 |
| PA-STFed | step6 | 30.1749 | 9.5295 |
| PA-STFed | step12 | 33.3626 | 9.9439 |
| PA-STFed | overall_12step | 29.3737 | 8.8903 |
| PA-STFed residual-anchor | prefix_h1 | 17.6575 | 4.9878 |
| PA-STFed residual-anchor | prefix_h3 | 23.2519 | 7.9207 |
| PA-STFed residual-anchor | prefix_h6 | 26.2772 | 9.4580 |
| PA-STFed residual-anchor | prefix_h12 | 28.9965 | 10.6623 |
| PA-STFed residual-anchor | step1 | 17.6575 | 4.9878 |
| PA-STFed residual-anchor | step3 | 27.2242 | 10.3468 |
| PA-STFed residual-anchor | step6 | 29.9599 | 11.2655 |
| PA-STFed residual-anchor | step12 | 32.8840 | 12.2408 |
| PA-STFed residual-anchor | overall_12step | 28.9965 | 10.6623 |
| GWN | prefix_h1 | 17.3894 | 5.0082 |
| GWN | prefix_h3 | 23.1853 | 6.9185 |
| GWN | prefix_h6 | 26.2692 | 8.0402 |
| GWN | prefix_h12 | 29.0054 | 8.7677 |
| GWN | step1 | 17.3894 | 5.0082 |
| GWN | step3 | 27.3556 | 8.3154 |
| GWN | step6 | 30.0644 | 9.3665 |
| GWN | step12 | 32.8986 | 9.7152 |
| GWN | overall_12step | 29.0054 | 8.7677 |
| Persistence | prefix_h1 | 18.0189 | 5.2410 |
| Persistence | prefix_h3 | 26.7421 | 7.5164 |
| Persistence | prefix_h6 | 33.3026 | 9.9025 |
| Persistence | prefix_h12 | 40.9818 | 14.1932 |
| Persistence | step1 | 18.0189 | 5.2410 |
| Persistence | step3 | 33.6876 | 9.4737 |
| Persistence | step6 | 42.1124 | 13.6675 |
| Persistence | step12 | 52.5797 | 21.8657 |
| Persistence | overall_12step | 40.9818 | 14.1932 |
| Daily-lag naive | prefix_h1 | 48.0422 | 16.0472 |
| Daily-lag naive | prefix_h3 | 48.0465 | 16.0453 |
| Daily-lag naive | prefix_h6 | 48.0525 | 16.0437 |
| Daily-lag naive | prefix_h12 | 48.0583 | 16.0348 |
| Daily-lag naive | step1 | 48.0422 | 16.0472 |
| Daily-lag naive | step3 | 48.0506 | 16.0436 |
| Daily-lag naive | step6 | 48.0601 | 16.0393 |
| Daily-lag naive | step12 | 48.0635 | 16.0231 |
| Daily-lag naive | overall_12step | 48.0583 | 16.0348 |
| Weekly-lag naive | prefix_h1 | 57.4112 | 26.3610 |
| Weekly-lag naive | prefix_h3 | 57.4145 | 26.3581 |
| Weekly-lag naive | prefix_h6 | 57.4198 | 26.3516 |
| Weekly-lag naive | prefix_h12 | 57.4246 | 26.3394 |
| Weekly-lag naive | step1 | 57.4112 | 26.3610 |
| Weekly-lag naive | step3 | 57.4187 | 26.3568 |
| Weekly-lag naive | step6 | 57.4267 | 26.3406 |
| Weekly-lag naive | step12 | 57.4229 | 26.3106 |
| Weekly-lag naive | overall_12step | 57.4246 | 26.3394 |
| Daily-weekly blend alpha=0.00 | prefix_h1 | 57.4112 | 26.3610 |
| Daily-weekly blend alpha=0.00 | prefix_h3 | 57.4145 | 26.3581 |
| Daily-weekly blend alpha=0.00 | prefix_h6 | 57.4198 | 26.3516 |
| Daily-weekly blend alpha=0.00 | prefix_h12 | 57.4246 | 26.3394 |
| Daily-weekly blend alpha=0.00 | step1 | 57.4112 | 26.3610 |
| Daily-weekly blend alpha=0.00 | step3 | 57.4187 | 26.3568 |
| Daily-weekly blend alpha=0.00 | step6 | 57.4267 | 26.3406 |
| Daily-weekly blend alpha=0.00 | step12 | 57.4229 | 26.3106 |
| Daily-weekly blend alpha=0.00 | overall_12step | 57.4246 | 26.3394 |
| Daily-weekly blend alpha=0.25 | prefix_h1 | 50.6080 | 21.3368 |
| Daily-weekly blend alpha=0.25 | prefix_h3 | 50.6118 | 21.3344 |
| Daily-weekly blend alpha=0.25 | prefix_h6 | 50.6178 | 21.3273 |
| Daily-weekly blend alpha=0.25 | prefix_h12 | 50.6237 | 21.3144 |
| Daily-weekly blend alpha=0.25 | step1 | 50.6080 | 21.3368 |
| Daily-weekly blend alpha=0.25 | step3 | 50.6162 | 21.3330 |
| Daily-weekly blend alpha=0.25 | step6 | 50.6255 | 21.3150 |
| Daily-weekly blend alpha=0.25 | step12 | 50.6261 | 21.2891 |
| Daily-weekly blend alpha=0.25 | overall_12step | 50.6237 | 21.3144 |
| Daily-weekly blend alpha=0.50 | prefix_h1 | 46.5071 | 17.3732 |
| Daily-weekly blend alpha=0.50 | prefix_h3 | 46.5114 | 17.3711 |
| Daily-weekly blend alpha=0.50 | prefix_h6 | 46.5180 | 17.3640 |
| Daily-weekly blend alpha=0.50 | prefix_h12 | 46.5249 | 17.3505 |
| Daily-weekly blend alpha=0.50 | step1 | 46.5071 | 17.3732 |
| Daily-weekly blend alpha=0.50 | step3 | 46.5160 | 17.3696 |
| Daily-weekly blend alpha=0.50 | step6 | 46.5266 | 17.3512 |
| Daily-weekly blend alpha=0.50 | step12 | 46.5306 | 17.3284 |
| Daily-weekly blend alpha=0.50 | overall_12step | 46.5249 | 17.3505 |
| Daily-weekly blend alpha=0.75 | prefix_h1 | 45.1847 | 15.1432 |
| Daily-weekly blend alpha=0.75 | prefix_h3 | 45.1894 | 15.1411 |
| Daily-weekly blend alpha=0.75 | prefix_h6 | 45.1963 | 15.1365 |
| Daily-weekly blend alpha=0.75 | prefix_h12 | 45.2035 | 15.1238 |
| Daily-weekly blend alpha=0.75 | step1 | 45.1847 | 15.1432 |
| Daily-weekly blend alpha=0.75 | step3 | 45.1941 | 15.1393 |
| Daily-weekly blend alpha=0.75 | step6 | 45.2051 | 15.1264 |
| Daily-weekly blend alpha=0.75 | step12 | 45.2109 | 15.1059 |
| Daily-weekly blend alpha=0.75 | overall_12step | 45.2035 | 15.1238 |
| Daily-weekly blend alpha=1.00 | prefix_h1 | 48.0422 | 16.0472 |
| Daily-weekly blend alpha=1.00 | prefix_h3 | 48.0465 | 16.0453 |
| Daily-weekly blend alpha=1.00 | prefix_h6 | 48.0525 | 16.0437 |
| Daily-weekly blend alpha=1.00 | prefix_h12 | 48.0583 | 16.0348 |
| Daily-weekly blend alpha=1.00 | step1 | 48.0422 | 16.0472 |
| Daily-weekly blend alpha=1.00 | step3 | 48.0506 | 16.0436 |
| Daily-weekly blend alpha=1.00 | step6 | 48.0601 | 16.0393 |
| Daily-weekly blend alpha=1.00 | step12 | 48.0635 | 16.0231 |
| Daily-weekly blend alpha=1.00 | overall_12step | 48.0583 | 16.0348 |

## Residual-anchor Differences

Differences are defined as residual-anchor minus the comparison method; negative values favor residual-anchor.

| Comparison | Horizon | dWAPE | dMAE | dRMSE | dFeeder WAPE |
|---|---:|---:|---:|---:|---:|
| residual-anchor - Original PA-STFed | prefix_h1 | -0.7007 | -0.008210 | 0.006735 | 0.2467 |
| residual-anchor - Original PA-STFed | prefix_h3 | -0.5421 | -0.006353 | 0.005351 | 1.0465 |
| residual-anchor - Original PA-STFed | prefix_h6 | -0.3743 | -0.004388 | 0.005233 | 1.3108 |
| residual-anchor - Original PA-STFed | prefix_h12 | -0.3772 | -0.004423 | 0.005934 | 1.7720 |
| residual-anchor - Original PA-STFed | step1 | -0.7007 | -0.008210 | 0.006735 | 0.2467 |
| residual-anchor - Original PA-STFed | step3 | -0.3655 | -0.004284 | 0.004860 | 1.5521 |
| residual-anchor - Original PA-STFed | step6 | -0.2150 | -0.002521 | 0.005078 | 1.7360 |
| residual-anchor - Original PA-STFed | step12 | -0.4786 | -0.005617 | 0.006623 | 2.2969 |
| residual-anchor - Original PA-STFed | overall_12step | -0.3772 | -0.004423 | 0.005934 | 1.7720 |
| residual-anchor - GWN | prefix_h1 | 0.2681 | 0.003141 | 0.007347 | -0.0205 |
| residual-anchor - GWN | prefix_h3 | 0.0666 | 0.000780 | 0.006146 | 1.0022 |
| residual-anchor - GWN | prefix_h6 | 0.0080 | 0.000094 | 0.006100 | 1.4178 |
| residual-anchor - GWN | prefix_h12 | -0.0089 | -0.000104 | 0.007463 | 1.8947 |
| residual-anchor - GWN | step1 | 0.2681 | 0.003141 | 0.007347 | -0.0205 |
| residual-anchor - GWN | step3 | -0.1314 | -0.001540 | 0.005378 | 2.0314 |
| residual-anchor - GWN | step6 | -0.1045 | -0.001225 | 0.004886 | 1.8990 |
| residual-anchor - GWN | step12 | -0.0146 | -0.000171 | 0.010761 | 2.5256 |
| residual-anchor - GWN | overall_12step | -0.0089 | -0.000104 | 0.007463 | 1.8947 |

## Node Difficulty

- PA-STFed: {'min': 2.762939929962158, 'p10': 12.90864372253418, 'p25': 25.412681579589844, 'median': 29.858762741088867, 'p75': 40.6666374206543, 'p90': 46.49767303466797, 'max': 49.01484680175781}; counts={'lt10': 8, 'lt20': 11, 'lt30': 47, 'ge40': 29}
- PA-STFed residual-anchor: {'min': 3.107927083969116, 'p10': 12.687405586242676, 'p25': 24.962587356567383, 'median': 29.210411071777344, 'p75': 40.534996032714844, 'p90': 46.20293045043945, 'max': 51.57440948486328}; counts={'lt10': 8, 'lt20': 12, 'lt30': 49, 'ge40': 27}
- GWN: {'min': 3.839329719543457, 'p10': 14.086502075195312, 'p25': 23.372276306152344, 'median': 28.794334411621094, 'p75': 40.738258361816406, 'p90': 47.29705810546875, 'max': 49.12572479248047}; counts={'lt10': 8, 'lt20': 12, 'lt30': 49, 'ge40': 27}
- Persistence: {'min': 4.913691997528076, 'p10': 19.870603561401367, 'p25': 35.058555603027344, 'median': 44.3123779296875, 'p75': 54.34208679199219, 'p90': 66.28288269042969, 'max': 92.21017456054688}; counts={'lt10': 5, 'lt20': 10, 'lt30': 11, 'ge40': 56}
- Daily-lag naive: {'min': 7.026316165924072, 'p10': 19.96192169189453, 'p25': 39.092445373535156, 'median': 51.287315368652344, 'p75': 60.653995513916016, 'p90': 73.21785736083984, 'max': 95.0457534790039}; counts={'lt10': 3, 'lt20': 10, 'lt30': 13, 'ge40': 67}
- Weekly-lag naive: {'min': 3.227156400680542, 'p10': 23.928659439086914, 'p25': 50.64219665527344, 'median': 61.87416076660156, 'p75': 71.95643615722656, 'p90': 81.28486633300781, 'max': 110.96682739257812}; counts={'lt10': 3, 'lt20': 7, 'lt30': 12, 'ge40': 79}
- Daily-weekly blend alpha=0.00: {'min': 3.227156400680542, 'p10': 23.928659439086914, 'p25': 50.64219665527344, 'median': 61.87416076660156, 'p75': 71.95643615722656, 'p90': 81.28486633300781, 'max': 110.96682739257812}; counts={'lt10': 3, 'lt20': 7, 'lt30': 12, 'ge40': 79}
- Daily-weekly blend alpha=0.25: {'min': 6.438445091247559, 'p10': 19.83623504638672, 'p25': 44.108299255371094, 'median': 56.037200927734375, 'p75': 64.21355438232422, 'p90': 71.00692749023438, 'max': 94.91107177734375}; counts={'lt10': 3, 'lt20': 10, 'lt30': 12, 'ge40': 76}
- Daily-weekly blend alpha=0.50: {'min': 9.043246269226074, 'p10': 17.60670280456543, 'p25': 39.13446044921875, 'median': 53.101234436035156, 'p75': 58.302764892578125, 'p90': 67.38572692871094, 'max': 94.17996978759766}; counts={'lt10': 4, 'lt20': 10, 'lt30': 12, 'ge40': 68}
- Daily-weekly blend alpha=0.75: {'min': 7.931594371795654, 'p10': 16.143129348754883, 'p25': 36.74297332763672, 'median': 50.27470397949219, 'p75': 57.6855354309082, 'p90': 68.76376342773438, 'max': 94.01577758789062}; counts={'lt10': 3, 'lt20': 10, 'lt30': 13, 'ge40': 61}
- Daily-weekly blend alpha=1.00: {'min': 7.026316165924072, 'p10': 19.96192169189453, 'p25': 39.092445373535156, 'median': 51.287315368652344, 'p75': 60.653995513916016, 'p90': 73.21785736083984, 'max': 95.04578399658203}; counts={'lt10': 3, 'lt20': 10, 'lt30': 13, 'ge40': 67}

## Weekly-lag Node Distribution

- quantiles: {'min': 3.227156400680542, 'p10': 23.928659439086914, 'p25': 50.64219665527344, 'median': 61.87416076660156, 'p75': 71.95643615722656, 'p90': 81.28486633300781, 'max': 110.96682739257812}
- PA-STFed has lower overall node WAPE than weekly-lag naive for 92 of 92 nodes; weekly-lag is lower or equal for 0 nodes.
- best10: [{'node_id': 'p10rlv7583', 'wape': 3.227156400680542}, {'node_id': 'p10rlv10217', 'wape': 3.227156400680542}, {'node_id': 'p10rlv30211', 'wape': 8.834614753723145}, {'node_id': 'p10rlv11105', 'wape': 12.923022270202637}, {'node_id': 'p10rlv13829', 'wape': 12.923022270202637}, {'node_id': 'p10rlv22134', 'wape': 12.923022270202637}, {'node_id': 'p10rlv28745', 'wape': 14.073055267333984}, {'node_id': 'p10rlv18297', 'wape': 22.602508544921875}, {'node_id': 'p10rlv20089', 'wape': 22.602508544921875}, {'node_id': 'p10rlv30788', 'wape': 23.575763702392578}]
- worst10: [{'node_id': 'p10rlv29910', 'wape': 110.96682739257812}, {'node_id': 'p10rlv29902', 'wape': 98.4835433959961}, {'node_id': 'p10rlv27605', 'wape': 97.00413513183594}, {'node_id': 'p10rlv18002', 'wape': 97.00413513183594}, {'node_id': 'p10rlv3062', 'wape': 97.00413513183594}, {'node_id': 'p10rlv4224', 'wape': 84.53926086425781}, {'node_id': 'p10rlv679', 'wape': 84.53926086425781}, {'node_id': 'p10rlv24611', 'wape': 83.37914276123047}, {'node_id': 'p10rlv21569', 'wape': 81.43507385253906}, {'node_id': 'p10rlv19213', 'wape': 81.42317199707031}]

## Forecastability Correlations

- PA-STFed: {'train_cv': 0.4158354257798238, 'lag96_autocorr': -0.6539908795744196, 'lag672_autocorr': -0.6765005554040376, 'mean_shift': -0.33968796761373926}
- PA-STFed residual-anchor: {'train_cv': 0.461432936920985, 'lag96_autocorr': -0.6224387894474926, 'lag672_autocorr': -0.6433604539000897, 'mean_shift': -0.3186044185062032}
- GWN: {'train_cv': 0.37324434394467326, 'lag96_autocorr': -0.6908466193488729, 'lag672_autocorr': -0.7098102503560169, 'mean_shift': -0.35916808775806963}
- Persistence: {'train_cv': 0.5171424605561467, 'lag96_autocorr': -0.5772605993306497, 'lag672_autocorr': -0.5890282083314053, 'mean_shift': -0.3037523712580391}
- Daily-lag naive: {'train_cv': 0.4725705208285136, 'lag96_autocorr': -0.6697049615200726, 'lag672_autocorr': -0.7266614229090516, 'mean_shift': -0.506809173491263}
- Weekly-lag naive: {'train_cv': 0.6428846836009191, 'lag96_autocorr': -0.3015006400468853, 'lag672_autocorr': -0.47206156788352677, 'mean_shift': -0.48834806212310483}
- Daily-weekly blend alpha=0.00: {'train_cv': 0.6428846836009191, 'lag96_autocorr': -0.3015006400468853, 'lag672_autocorr': -0.47206156788352677, 'mean_shift': -0.48834806212310483}
- Daily-weekly blend alpha=0.25: {'train_cv': 0.637980227949228, 'lag96_autocorr': -0.42653341353197916, 'lag672_autocorr': -0.5709526673761162, 'mean_shift': -0.51507580314317}
- Daily-weekly blend alpha=0.50: {'train_cv': 0.5980814016255648, 'lag96_autocorr': -0.5423124971082218, 'lag672_autocorr': -0.6368852079766805, 'mean_shift': -0.49797189962831007}
- Daily-weekly blend alpha=0.75: {'train_cv': 0.5112355218309966, 'lag96_autocorr': -0.6416817039127684, 'lag672_autocorr': -0.7067968352380509, 'mean_shift': -0.5005320871697588}
- Daily-weekly blend alpha=1.00: {'train_cv': 0.4725705208285136, 'lag96_autocorr': -0.6697049615200726, 'lag672_autocorr': -0.7266614229090516, 'mean_shift': -0.506809173491263}

## Conclusions

1. PA-STFed node-micro WAPE is 29.37% and feeder-aggregate WAPE is 8.89%; this quantifies the aggregation-level effect.
2. PA-STFed overall prefix WAPE minus exact step1 WAPE is 11.02 percentage points, so horizon degradation is present.
3. The reported Spearman correlations quantify whether high error tracks CV, autocorrelation, or mean shift; no causal claim is made.
4. PA-STFed versus GWN overall-12-step WAPE gap is 0.37 percentage points; horizon-wise and node-level tables above show where it concentrates.
