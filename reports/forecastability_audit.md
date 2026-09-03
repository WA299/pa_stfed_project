# Forecastability Audit

Validation-only; no training, no test loader, and no existing result JSON was modified.

## Metric Sanity

- PA-STFed: recomputed WAPE/MAE/RMSE = {'wape': 29.373682022094727, 'mae': 0.34444817900657654, 'rmse': 0.714534342288971}; max difference = 0.00000381 (PASS)
- GWN: recomputed WAPE/MAE/RMSE = {'wape': 29.005434036254883, 'mae': 0.34012991189956665, 'rmse': 0.713005781173706}; max difference = 0.00000000 (PASS)
- PA-STFed residual-anchor: recomputed WAPE/MAE/RMSE = {'wape': 28.996524810791016, 'mae': 0.34002548456192017, 'rmse': 0.720468282699585}; max difference = 0.00000381 (PASS)
- PA-STFed residual-scale-loss: recomputed WAPE/MAE/RMSE = {'wape': 28.94508171081543, 'mae': 0.33942222595214844, 'rmse': 0.7123457193374634}; max difference = 0.00000000 (PASS)
- PA-STFed residual-multilevel-loss: recomputed WAPE/MAE/RMSE = {'wape': 29.182409286499023, 'mae': 0.3422051966190338, 'rmse': 0.7109360694885254}; max difference = 0.00000191 (PASS)
- PA-STFed residual-multilevel-lambda0.02: recomputed WAPE/MAE/RMSE = {'wape': 29.10218620300293, 'mae': 0.3412645161151886, 'rmse': 0.7116377353668213}; max difference = 0.00000006 (PASS)
- PA-STFed residual-multilevel-lambda0.05: recomputed WAPE/MAE/RMSE = {'wape': 29.041759490966797, 'mae': 0.34055590629577637, 'rmse': 0.7117523550987244}; max difference = 0.00000000 (PASS)

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
| PA-STFed residual-scale-loss | prefix_h1 | 17.7820 | 0.208347 | 0.463068 |
| PA-STFed residual-scale-loss | prefix_h3 | 23.3636 | 0.273786 | 0.600243 |
| PA-STFed residual-scale-loss | prefix_h6 | 26.3053 | 0.308330 | 0.667148 |
| PA-STFed residual-scale-loss | prefix_h12 | 28.9451 | 0.339422 | 0.712346 |
| PA-STFed residual-scale-loss | step1 | 17.7820 | 0.208347 | 0.463068 |
| PA-STFed residual-scale-loss | step3 | 27.3262 | 0.320265 | 0.689506 |
| PA-STFed residual-scale-loss | step6 | 29.8486 | 0.349996 | 0.736674 |
| PA-STFed residual-scale-loss | step12 | 32.6839 | 0.383587 | 0.767869 |
| PA-STFed residual-scale-loss | overall_12step | 28.9451 | 0.339422 | 0.712346 |
| PA-STFed residual-multilevel-loss | prefix_h1 | 17.8854 | 0.209560 | 0.463564 |
| PA-STFed residual-multilevel-loss | prefix_h3 | 23.5162 | 0.275574 | 0.600269 |
| PA-STFed residual-multilevel-loss | prefix_h6 | 26.5371 | 0.311047 | 0.666213 |
| PA-STFed residual-multilevel-loss | prefix_h12 | 29.1824 | 0.342205 | 0.710936 |
| PA-STFed residual-multilevel-loss | step1 | 17.8854 | 0.209560 | 0.463564 |
| PA-STFed residual-multilevel-loss | step3 | 27.4578 | 0.321808 | 0.688634 |
| PA-STFed residual-multilevel-loss | step6 | 30.2030 | 0.354151 | 0.734610 |
| PA-STFed residual-multilevel-loss | step12 | 32.9239 | 0.386403 | 0.766036 |
| PA-STFed residual-multilevel-loss | overall_12step | 29.1824 | 0.342205 | 0.710936 |
| PA-STFed residual-multilevel-lambda0.02 | prefix_h1 | 17.8257 | 0.208860 | 0.463468 |
| PA-STFed residual-multilevel-lambda0.02 | prefix_h3 | 23.4380 | 0.274658 | 0.599990 |
| PA-STFed residual-multilevel-lambda0.02 | prefix_h6 | 26.4498 | 0.310024 | 0.666497 |
| PA-STFed residual-multilevel-lambda0.02 | prefix_h12 | 29.1022 | 0.341265 | 0.711638 |
| PA-STFed residual-multilevel-lambda0.02 | step1 | 17.8257 | 0.208860 | 0.463468 |
| PA-STFed residual-multilevel-lambda0.02 | step3 | 27.4461 | 0.321671 | 0.689090 |
| PA-STFed residual-multilevel-lambda0.02 | step6 | 30.1079 | 0.353037 | 0.735669 |
| PA-STFed residual-multilevel-lambda0.02 | step12 | 32.7655 | 0.384544 | 0.765710 |
| PA-STFed residual-multilevel-lambda0.02 | overall_12step | 29.1022 | 0.341265 | 0.711638 |
| PA-STFed residual-multilevel-lambda0.05 | prefix_h1 | 17.9718 | 0.210571 | 0.462524 |
| PA-STFed residual-multilevel-lambda0.05 | prefix_h3 | 23.5480 | 0.275947 | 0.600430 |
| PA-STFed residual-multilevel-lambda0.05 | prefix_h6 | 26.4686 | 0.310244 | 0.666881 |
| PA-STFed residual-multilevel-lambda0.05 | prefix_h12 | 29.0418 | 0.340556 | 0.711752 |
| PA-STFed residual-multilevel-lambda0.05 | step1 | 17.9718 | 0.210571 | 0.462524 |
| PA-STFed residual-multilevel-lambda0.05 | step3 | 27.4774 | 0.322037 | 0.690513 |
| PA-STFed residual-multilevel-lambda0.05 | step6 | 29.9776 | 0.351508 | 0.735831 |
| PA-STFed residual-multilevel-lambda0.05 | step12 | 32.6563 | 0.383263 | 0.767057 |
| PA-STFed residual-multilevel-lambda0.05 | overall_12step | 29.0418 | 0.340556 | 0.711752 |
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
| PA-STFed residual-scale-loss | prefix_h1 | 17.7820 | 4.9895 |
| PA-STFed residual-scale-loss | prefix_h3 | 23.3636 | 7.0119 |
| PA-STFed residual-scale-loss | prefix_h6 | 26.3053 | 8.1205 |
| PA-STFed residual-scale-loss | prefix_h12 | 28.9451 | 9.0346 |
| PA-STFed residual-scale-loss | step1 | 17.7820 | 4.9895 |
| PA-STFed residual-scale-loss | step3 | 27.3262 | 8.6301 |
| PA-STFed residual-scale-loss | step6 | 29.8486 | 9.4696 |
| PA-STFed residual-scale-loss | step12 | 32.6839 | 10.5541 |
| PA-STFed residual-scale-loss | overall_12step | 28.9451 | 9.0346 |
| PA-STFed residual-multilevel-loss | prefix_h1 | 17.8854 | 4.8553 |
| PA-STFed residual-multilevel-loss | prefix_h3 | 23.5162 | 6.7189 |
| PA-STFed residual-multilevel-loss | prefix_h6 | 26.5371 | 7.6390 |
| PA-STFed residual-multilevel-loss | prefix_h12 | 29.1824 | 8.4795 |
| PA-STFed residual-multilevel-loss | step1 | 17.8854 | 4.8553 |
| PA-STFed residual-multilevel-loss | step3 | 27.4578 | 8.2269 |
| PA-STFed residual-multilevel-loss | step6 | 30.2030 | 8.7262 |
| PA-STFed residual-multilevel-loss | step12 | 32.9239 | 9.9349 |
| PA-STFed residual-multilevel-loss | overall_12step | 29.1824 | 8.4795 |
| PA-STFed residual-multilevel-lambda0.02 | prefix_h1 | 17.8257 | 4.9775 |
| PA-STFed residual-multilevel-lambda0.02 | prefix_h3 | 23.4380 | 6.8683 |
| PA-STFed residual-multilevel-lambda0.02 | prefix_h6 | 26.4498 | 7.8756 |
| PA-STFed residual-multilevel-lambda0.02 | prefix_h12 | 29.1022 | 8.7875 |
| PA-STFed residual-multilevel-lambda0.02 | step1 | 17.8257 | 4.9775 |
| PA-STFed residual-multilevel-lambda0.02 | step3 | 27.4461 | 8.3547 |
| PA-STFed residual-multilevel-lambda0.02 | step6 | 30.1079 | 9.0613 |
| PA-STFed residual-multilevel-lambda0.02 | step12 | 32.7655 | 10.3408 |
| PA-STFed residual-multilevel-lambda0.02 | overall_12step | 29.1022 | 8.7875 |
| PA-STFed residual-multilevel-lambda0.05 | prefix_h1 | 17.9718 | 4.9513 |
| PA-STFed residual-multilevel-lambda0.05 | prefix_h3 | 23.5480 | 6.8400 |
| PA-STFed residual-multilevel-lambda0.05 | prefix_h6 | 26.4686 | 7.8500 |
| PA-STFed residual-multilevel-lambda0.05 | prefix_h12 | 29.0418 | 8.8223 |
| PA-STFed residual-multilevel-lambda0.05 | step1 | 17.9718 | 4.9513 |
| PA-STFed residual-multilevel-lambda0.05 | step3 | 27.4774 | 8.3735 |
| PA-STFed residual-multilevel-lambda0.05 | step6 | 29.9776 | 9.1207 |
| PA-STFed residual-multilevel-lambda0.05 | step12 | 32.6563 | 10.5884 |
| PA-STFed residual-multilevel-lambda0.05 | overall_12step | 29.0418 | 8.8223 |
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

## Scale-aware and Multilevel Loss Differences

Negative values favor the first method named in each comparison; node WAPE and feeder-aggregate WAPE are shown together.

| Comparison | Horizon | dWAPE | dMAE | dRMSE | dFeeder WAPE |
|---|---:|---:|---:|---:|---:|
| scale-aware - residual | prefix_h1 | 0.1245 | 0.001459 | -0.001600 | 0.0017 |
| scale-aware - residual | prefix_h3 | 0.1117 | 0.001309 | -0.006067 | -0.9088 |
| scale-aware - residual | prefix_h6 | 0.0281 | 0.000329 | -0.007177 | -1.3375 |
| scale-aware - residual | prefix_h12 | -0.0514 | -0.000603 | -0.008123 | -1.6277 |
| scale-aware - residual | step1 | 0.1245 | 0.001459 | -0.001600 | 0.0017 |
| scale-aware - residual | step3 | 0.1020 | 0.001195 | -0.008016 | -1.7168 |
| scale-aware - residual | step6 | -0.1113 | -0.001305 | -0.008463 | -1.7959 |
| scale-aware - residual | step12 | -0.2001 | -0.002349 | -0.007965 | -1.6867 |
| scale-aware - residual | overall_12step | -0.0514 | -0.000603 | -0.008123 | -1.6277 |
| multilevel - scale-aware | prefix_h1 | 0.1035 | 0.001212 | 0.000496 | -0.1342 |
| multilevel - scale-aware | prefix_h3 | 0.1526 | 0.001788 | 0.000027 | -0.2930 |
| multilevel - scale-aware | prefix_h6 | 0.2318 | 0.002717 | -0.000935 | -0.4814 |
| multilevel - scale-aware | prefix_h12 | 0.2373 | 0.002783 | -0.001410 | -0.5551 |
| multilevel - scale-aware | step1 | 0.1035 | 0.001212 | 0.000496 | -0.1342 |
| multilevel - scale-aware | step3 | 0.1316 | 0.001542 | -0.000871 | -0.4031 |
| multilevel - scale-aware | step6 | 0.3544 | 0.004155 | -0.002065 | -0.7435 |
| multilevel - scale-aware | step12 | 0.2400 | 0.002816 | -0.001833 | -0.6192 |
| multilevel - scale-aware | overall_12step | 0.2373 | 0.002783 | -0.001410 | -0.5551 |
| multilevel - GWN | prefix_h1 | 0.4960 | 0.005812 | 0.006243 | -0.1530 |
| multilevel - GWN | prefix_h3 | 0.3309 | 0.003877 | 0.000106 | -0.1996 |
| multilevel - GWN | prefix_h6 | 0.2679 | 0.003140 | -0.002012 | -0.4012 |
| multilevel - GWN | prefix_h12 | 0.1770 | 0.002075 | -0.002070 | -0.2882 |
| multilevel - GWN | step1 | 0.4960 | 0.005812 | 0.006243 | -0.1530 |
| multilevel - GWN | step3 | 0.1022 | 0.001197 | -0.003508 | -0.0885 |
| multilevel - GWN | step6 | 0.1385 | 0.001624 | -0.005642 | -0.6403 |
| multilevel - GWN | step12 | 0.0252 | 0.000296 | 0.000964 | 0.2197 |
| multilevel - GWN | overall_12step | 0.1770 | 0.002075 | -0.002070 | -0.2882 |

## Multilevel Lambda Table

Exact forecast steps and overall 12-step; lambda=0 is the scale-aware node loss without the feeder term.

| Lambda | Horizon | Node-micro WAPE | Feeder aggregate WAPE |
|---:|---:|---:|---:|
| lambda=0.00 (scale-aware) | step1 | 17.7820 | 4.9895 |
| lambda=0.00 (scale-aware) | step3 | 27.3262 | 8.6301 |
| lambda=0.00 (scale-aware) | step6 | 29.8486 | 9.4696 |
| lambda=0.00 (scale-aware) | step12 | 32.6839 | 10.5541 |
| lambda=0.00 (scale-aware) | overall_12step | 28.9451 | 9.0346 |
| lambda=0.02 | step1 | 17.8257 | 4.9775 |
| lambda=0.02 | step3 | 27.4461 | 8.3547 |
| lambda=0.02 | step6 | 30.1079 | 9.0613 |
| lambda=0.02 | step12 | 32.7655 | 10.3408 |
| lambda=0.02 | overall_12step | 29.1022 | 8.7875 |
| lambda=0.05 | step1 | 17.9718 | 4.9513 |
| lambda=0.05 | step3 | 27.4774 | 8.3735 |
| lambda=0.05 | step6 | 29.9776 | 9.1207 |
| lambda=0.05 | step12 | 32.6563 | 10.5884 |
| lambda=0.05 | overall_12step | 29.0418 | 8.8223 |
| lambda=0.10 | step1 | 17.8854 | 4.8553 |
| lambda=0.10 | step3 | 27.4578 | 8.2269 |
| lambda=0.10 | step6 | 30.2030 | 8.7262 |
| lambda=0.10 | step12 | 32.9239 | 9.9349 |
| lambda=0.10 | overall_12step | 29.1824 | 8.4795 |

## New Lambda Differences

Differences are left method minus right method; negative values favor the tested lambda.

| Comparison | Horizon | dWAPE | dMAE | dRMSE | dFeeder WAPE |
|---|---:|---:|---:|---:|---:|
| lambda0.02_minus_scale_aware_lambda0 | step1 | 0.0438 | 0.000513 | 0.000400 | -0.0119 |
| lambda0.02_minus_scale_aware_lambda0 | step3 | 0.1199 | 0.001405 | -0.000415 | -0.2753 |
| lambda0.02_minus_scale_aware_lambda0 | step6 | 0.2593 | 0.003040 | -0.001005 | -0.4083 |
| lambda0.02_minus_scale_aware_lambda0 | step12 | 0.0816 | 0.000958 | -0.002158 | -0.2133 |
| lambda0.02_minus_scale_aware_lambda0 | overall_12step | 0.1571 | 0.001842 | -0.000708 | -0.2471 |
| lambda0.02_minus_lambda0.1 | step1 | -0.0597 | -0.000700 | -0.000096 | 0.1222 |
| lambda0.02_minus_lambda0.1 | step3 | -0.0117 | -0.000137 | 0.000456 | 0.1278 |
| lambda0.02_minus_lambda0.1 | step6 | -0.0951 | -0.001115 | 0.001059 | 0.3351 |
| lambda0.02_minus_lambda0.1 | step12 | -0.1584 | -0.001859 | -0.000325 | 0.4059 |
| lambda0.02_minus_lambda0.1 | overall_12step | -0.0802 | -0.000941 | 0.000702 | 0.3080 |
| lambda0.02_minus_gwn | step1 | 0.4363 | 0.005112 | 0.006147 | -0.0307 |
| lambda0.02_minus_gwn | step3 | 0.0905 | 0.001060 | -0.003052 | 0.0393 |
| lambda0.02_minus_gwn | step6 | 0.0435 | 0.000510 | -0.004583 | -0.3052 |
| lambda0.02_minus_gwn | step12 | -0.1331 | -0.001562 | 0.000639 | 0.6256 |
| lambda0.02_minus_gwn | overall_12step | 0.0968 | 0.001135 | -0.001368 | 0.0198 |
| lambda0.05_minus_scale_aware_lambda0 | step1 | 0.1898 | 0.002224 | -0.000544 | -0.0382 |
| lambda0.05_minus_scale_aware_lambda0 | step3 | 0.1512 | 0.001772 | 0.001007 | -0.2566 |
| lambda0.05_minus_scale_aware_lambda0 | step6 | 0.1290 | 0.001512 | -0.000843 | -0.3490 |
| lambda0.05_minus_scale_aware_lambda0 | step12 | -0.0276 | -0.000324 | -0.000811 | 0.0343 |
| lambda0.05_minus_scale_aware_lambda0 | overall_12step | 0.0967 | 0.001134 | -0.000593 | -0.2123 |
| lambda0.05_minus_lambda0.1 | step1 | 0.0863 | 0.001012 | -0.001040 | 0.0960 |
| lambda0.05_minus_lambda0.1 | step3 | 0.0196 | 0.000230 | 0.001878 | 0.1466 |
| lambda0.05_minus_lambda0.1 | step6 | -0.2254 | -0.002643 | 0.001222 | 0.3945 |
| lambda0.05_minus_lambda0.1 | step12 | -0.2676 | -0.003141 | 0.001022 | 0.6535 |
| lambda0.05_minus_lambda0.1 | overall_12step | -0.1406 | -0.001649 | 0.000816 | 0.3428 |
| lambda0.05_minus_gwn | step1 | 0.5824 | 0.006824 | 0.005203 | -0.0570 |
| lambda0.05_minus_gwn | step3 | 0.1217 | 0.001427 | -0.001630 | 0.0581 |
| lambda0.05_minus_gwn | step6 | -0.0869 | -0.001019 | -0.004420 | -0.2458 |
| lambda0.05_minus_gwn | step12 | -0.2424 | -0.002844 | 0.001986 | 0.8732 |
| lambda0.05_minus_gwn | overall_12step | 0.0363 | 0.000426 | -0.001253 | 0.0546 |

## Loss Ablation Node WAPE Differences

Node deltas are left method minus right method in percentage points; negative values favor the left method.
- scale_minus_residual: left better=32, right better=60, ties=0; quantiles={'min': -2.055980682373047, 'p10': -0.6800617218017578, 'median': 0.13433837890625, 'p90': 1.3193685531616228, 'max': 4.956027984619141}
- multilevel_minus_scale: left better=34, right better=58, ties=0; quantiles={'min': -2.764209747314453, 'p10': -0.3735698699951172, 'median': 0.12675905227661133, 'p90': 0.9894680023193362, 'max': 1.9545669555664062}
- multilevel_minus_gwn: left better=26, right better=66, ties=0; quantiles={'min': -1.9338302612304688, 'p10': -0.8182716369628906, 'median': 0.39148902893066406, 'p90': 3.8209986686706547, 'max': 7.118358612060547}

| Comparison | Node | dWAPE (pp) |
|---|---|---:|
| scale_minus_residual | p10rlv92 | 0.461422 |
| scale_minus_residual | p10rlv679 | 0.399654 |
| scale_minus_residual | p10rlv696 | 0.641895 |
| scale_minus_residual | p10rlv943 | 0.680426 |
| scale_minus_residual | p10rlv1542 | 0.389896 |
| scale_minus_residual | p10rlv1819 | 0.447393 |
| scale_minus_residual | p10rlv2169 | 0.579613 |
| scale_minus_residual | p10rlv2790 | -0.447861 |
| scale_minus_residual | p10rlv3062 | 4.714306 |
| scale_minus_residual | p10rlv3393 | 0.920979 |
| scale_minus_residual | p10rlv3394 | 0.730228 |
| scale_minus_residual | p10rlv3936 | 1.047989 |
| scale_minus_residual | p10rlv4224 | 0.097784 |
| scale_minus_residual | p10rlv4862 | -0.147980 |
| scale_minus_residual | p10rlv5170 | 0.237862 |
| scale_minus_residual | p10rlv5780 | 0.734550 |
| scale_minus_residual | p10rlv5781 | -0.103130 |
| scale_minus_residual | p10rlv6371 | 0.265854 |
| scale_minus_residual | p10rlv6373 | 0.090622 |
| scale_minus_residual | p10rlv6659 | -0.530704 |
| scale_minus_residual | p10rlv7262 | 0.370317 |
| scale_minus_residual | p10rlv7286 | 0.215782 |
| scale_minus_residual | p10rlv7583 | -0.141769 |
| scale_minus_residual | p10rlv7584 | -0.093414 |
| scale_minus_residual | p10rlv7860 | -1.384163 |
| scale_minus_residual | p10rlv7861 | 0.329884 |
| scale_minus_residual | p10rlv8456 | 0.129990 |
| scale_minus_residual | p10rlv9050 | 0.292049 |
| scale_minus_residual | p10rlv9051 | 0.406456 |
| scale_minus_residual | p10rlv9646 | -0.966719 |
| scale_minus_residual | p10rlv9907 | 0.000122 |
| scale_minus_residual | p10rlv9923 | -0.067820 |
| scale_minus_residual | p10rlv10217 | -0.092543 |
| scale_minus_residual | p10rlv11104 | 0.130173 |
| scale_minus_residual | p10rlv11105 | 4.514129 |
| scale_minus_residual | p10rlv11392 | 1.832096 |
| scale_minus_residual | p10rlv12002 | -0.166260 |
| scale_minus_residual | p10rlv12029 | 0.302563 |
| scale_minus_residual | p10rlv12030 | -0.059792 |
| scale_minus_residual | p10rlv12345 | 0.268938 |
| scale_minus_residual | p10rlv13496 | 1.050102 |
| scale_minus_residual | p10rlv13829 | 4.594760 |
| scale_minus_residual | p10rlv15288 | 0.174240 |
| scale_minus_residual | p10rlv15908 | 0.485081 |
| scale_minus_residual | p10rlv16523 | 0.082668 |
| scale_minus_residual | p10rlv17088 | -1.068325 |
| scale_minus_residual | p10rlv17370 | 0.300428 |
| scale_minus_residual | p10rlv17685 | 0.132008 |
| scale_minus_residual | p10rlv18002 | 4.956028 |
| scale_minus_residual | p10rlv18296 | 0.945160 |
| scale_minus_residual | p10rlv18297 | 0.040838 |
| scale_minus_residual | p10rlv18611 | 0.407667 |
| scale_minus_residual | p10rlv18908 | 0.044548 |
| scale_minus_residual | p10rlv19213 | -0.014622 |
| scale_minus_residual | p10rlv19495 | -0.662807 |
| scale_minus_residual | p10rlv19496 | 0.246454 |
| scale_minus_residual | p10rlv19818 | -1.071997 |
| scale_minus_residual | p10rlv20089 | 0.002290 |
| scale_minus_residual | p10rlv20103 | -1.033758 |
| scale_minus_residual | p10rlv20105 | 0.686644 |
| scale_minus_residual | p10rlv20976 | -0.230642 |
| scale_minus_residual | p10rlv21568 | -1.564898 |
| scale_minus_residual | p10rlv21569 | -0.472950 |
| scale_minus_residual | p10rlv21869 | 0.237801 |
| scale_minus_residual | p10rlv22134 | 3.970173 |
| scale_minus_residual | p10rlv22442 | 0.362103 |
| scale_minus_residual | p10rlv23067 | 2.480146 |
| scale_minus_residual | p10rlv23375 | 0.296192 |
| scale_minus_residual | p10rlv24009 | 0.679714 |
| scale_minus_residual | p10rlv24610 | 0.056515 |
| scale_minus_residual | p10rlv24611 | -0.488495 |
| scale_minus_residual | p10rlv24936 | -0.195450 |
| scale_minus_residual | p10rlv25536 | -0.882328 |
| scale_minus_residual | p10rlv26432 | 0.287472 |
| scale_minus_residual | p10rlv26433 | 1.043381 |
| scale_minus_residual | p10rlv27290 | -0.300007 |
| scale_minus_residual | p10rlv27605 | 4.877911 |
| scale_minus_residual | p10rlv27606 | 0.058399 |
| scale_minus_residual | p10rlv28435 | 1.349287 |
| scale_minus_residual | p10rlv28744 | 0.239342 |
| scale_minus_residual | p10rlv28745 | 0.038299 |
| scale_minus_residual | p10rlv29054 | -0.939249 |
| scale_minus_residual | p10rlv29354 | -0.665560 |
| scale_minus_residual | p10rlv29657 | 0.136669 |
| scale_minus_residual | p10rlv29902 | -2.055981 |
| scale_minus_residual | p10rlv29910 | -0.160099 |
| scale_minus_residual | p10rlv29912 | 1.524275 |
| scale_minus_residual | p10rlv29913 | 0.045418 |
| scale_minus_residual | p10rlv29914 | -0.207062 |
| scale_minus_residual | p10rlv30210 | -0.439686 |
| scale_minus_residual | p10rlv30211 | -0.681673 |
| scale_minus_residual | p10rlv30788 | -0.196280 |
| multilevel_minus_scale | p10rlv92 | 0.191048 |
| multilevel_minus_scale | p10rlv679 | 0.529039 |
| multilevel_minus_scale | p10rlv696 | -0.054325 |
| multilevel_minus_scale | p10rlv943 | 0.081934 |
| multilevel_minus_scale | p10rlv1542 | 1.032047 |
| multilevel_minus_scale | p10rlv1819 | -0.181314 |
| multilevel_minus_scale | p10rlv2169 | 0.475231 |
| multilevel_minus_scale | p10rlv2790 | 1.313408 |
| multilevel_minus_scale | p10rlv3062 | -0.214958 |
| multilevel_minus_scale | p10rlv3393 | 0.369869 |
| multilevel_minus_scale | p10rlv3394 | 0.021652 |
| multilevel_minus_scale | p10rlv3936 | -0.055824 |
| multilevel_minus_scale | p10rlv4224 | 0.524311 |
| multilevel_minus_scale | p10rlv4862 | 0.500101 |
| multilevel_minus_scale | p10rlv5170 | 0.091162 |
| multilevel_minus_scale | p10rlv5780 | 0.098549 |
| multilevel_minus_scale | p10rlv5781 | 0.214417 |
| multilevel_minus_scale | p10rlv6371 | 0.390759 |
| multilevel_minus_scale | p10rlv6373 | -0.105175 |
| multilevel_minus_scale | p10rlv6659 | 0.018223 |
| multilevel_minus_scale | p10rlv7262 | 0.689404 |
| multilevel_minus_scale | p10rlv7286 | -0.294746 |
| multilevel_minus_scale | p10rlv7583 | -0.043071 |
| multilevel_minus_scale | p10rlv7584 | 0.994587 |
| multilevel_minus_scale | p10rlv7860 | 0.299393 |
| multilevel_minus_scale | p10rlv7861 | 1.182335 |
| multilevel_minus_scale | p10rlv8456 | 0.704052 |
| multilevel_minus_scale | p10rlv9050 | -0.382328 |
| multilevel_minus_scale | p10rlv9051 | 0.247335 |
| multilevel_minus_scale | p10rlv9646 | -0.023224 |
| multilevel_minus_scale | p10rlv9907 | 0.052994 |
| multilevel_minus_scale | p10rlv9923 | 0.437328 |
| multilevel_minus_scale | p10rlv10217 | -0.105353 |
| multilevel_minus_scale | p10rlv11104 | 0.303019 |
| multilevel_minus_scale | p10rlv11105 | -0.193711 |
| multilevel_minus_scale | p10rlv11392 | -2.655338 |
| multilevel_minus_scale | p10rlv12002 | 0.135372 |
| multilevel_minus_scale | p10rlv12029 | 0.217604 |
| multilevel_minus_scale | p10rlv12030 | -0.133942 |
| multilevel_minus_scale | p10rlv12345 | 0.596056 |
| multilevel_minus_scale | p10rlv13496 | -0.680363 |
| multilevel_minus_scale | p10rlv13829 | -0.169044 |
| multilevel_minus_scale | p10rlv15288 | 0.600323 |
| multilevel_minus_scale | p10rlv15908 | -0.009905 |
| multilevel_minus_scale | p10rlv16523 | 0.751827 |
| multilevel_minus_scale | p10rlv17088 | 1.333210 |
| multilevel_minus_scale | p10rlv17370 | -1.702183 |
| multilevel_minus_scale | p10rlv17685 | -0.142269 |
| multilevel_minus_scale | p10rlv18002 | -0.124741 |
| multilevel_minus_scale | p10rlv18296 | 0.516403 |
| multilevel_minus_scale | p10rlv18297 | 0.198961 |
| multilevel_minus_scale | p10rlv18611 | 0.353537 |
| multilevel_minus_scale | p10rlv18908 | -0.096951 |
| multilevel_minus_scale | p10rlv19213 | 1.063988 |
| multilevel_minus_scale | p10rlv19495 | 0.027639 |
| multilevel_minus_scale | p10rlv19496 | 0.197073 |
| multilevel_minus_scale | p10rlv19818 | 1.954567 |
| multilevel_minus_scale | p10rlv20089 | -0.264002 |
| multilevel_minus_scale | p10rlv20103 | -0.234871 |
| multilevel_minus_scale | p10rlv20105 | -2.173010 |
| multilevel_minus_scale | p10rlv20976 | -0.040482 |
| multilevel_minus_scale | p10rlv21568 | 1.753342 |
| multilevel_minus_scale | p10rlv21569 | 0.424541 |
| multilevel_minus_scale | p10rlv21869 | -1.598185 |
| multilevel_minus_scale | p10rlv22134 | 0.235808 |
| multilevel_minus_scale | p10rlv22442 | 0.105934 |
| multilevel_minus_scale | p10rlv23067 | -2.764210 |
| multilevel_minus_scale | p10rlv23375 | -0.182877 |
| multilevel_minus_scale | p10rlv24009 | 0.943398 |
| multilevel_minus_scale | p10rlv24610 | 1.145695 |
| multilevel_minus_scale | p10rlv24611 | 0.081341 |
| multilevel_minus_scale | p10rlv24936 | 0.767467 |
| multilevel_minus_scale | p10rlv25536 | -0.933609 |
| multilevel_minus_scale | p10rlv26432 | 1.342709 |
| multilevel_minus_scale | p10rlv26433 | -0.042419 |
| multilevel_minus_scale | p10rlv27290 | 0.078651 |
| multilevel_minus_scale | p10rlv27605 | -0.105114 |
| multilevel_minus_scale | p10rlv27606 | 0.901501 |
| multilevel_minus_scale | p10rlv28435 | -0.383528 |
| multilevel_minus_scale | p10rlv28744 | 0.133091 |
| multilevel_minus_scale | p10rlv28745 | -0.124899 |
| multilevel_minus_scale | p10rlv29054 | 0.142452 |
| multilevel_minus_scale | p10rlv29354 | 0.028391 |
| multilevel_minus_scale | p10rlv29657 | 0.443447 |
| multilevel_minus_scale | p10rlv29902 | 0.555168 |
| multilevel_minus_scale | p10rlv29910 | 0.774618 |
| multilevel_minus_scale | p10rlv29912 | -2.418961 |
| multilevel_minus_scale | p10rlv29913 | 0.705727 |
| multilevel_minus_scale | p10rlv29914 | 0.458691 |
| multilevel_minus_scale | p10rlv30210 | -0.218605 |
| multilevel_minus_scale | p10rlv30211 | 0.188076 |
| multilevel_minus_scale | p10rlv30788 | 0.120427 |
| multilevel_minus_gwn | p10rlv92 | 0.208996 |
| multilevel_minus_gwn | p10rlv679 | 1.783897 |
| multilevel_minus_gwn | p10rlv696 | 0.472904 |
| multilevel_minus_gwn | p10rlv943 | 0.431725 |
| multilevel_minus_gwn | p10rlv1542 | 0.754827 |
| multilevel_minus_gwn | p10rlv1819 | 0.382095 |
| multilevel_minus_gwn | p10rlv2169 | 0.147629 |
| multilevel_minus_gwn | p10rlv2790 | 2.130283 |
| multilevel_minus_gwn | p10rlv3062 | 7.054760 |
| multilevel_minus_gwn | p10rlv3393 | 0.910507 |
| multilevel_minus_gwn | p10rlv3394 | 0.337811 |
| multilevel_minus_gwn | p10rlv3936 | 0.047443 |
| multilevel_minus_gwn | p10rlv4224 | 1.293333 |
| multilevel_minus_gwn | p10rlv4862 | -0.203342 |
| multilevel_minus_gwn | p10rlv5170 | 0.085047 |
| multilevel_minus_gwn | p10rlv5780 | 0.386604 |
| multilevel_minus_gwn | p10rlv5781 | 0.396374 |
| multilevel_minus_gwn | p10rlv6371 | 0.189739 |
| multilevel_minus_gwn | p10rlv6373 | -0.636452 |
| multilevel_minus_gwn | p10rlv6659 | 0.102543 |
| multilevel_minus_gwn | p10rlv7262 | 0.649309 |
| multilevel_minus_gwn | p10rlv7286 | -0.822636 |
| multilevel_minus_gwn | p10rlv7583 | -1.189265 |
| multilevel_minus_gwn | p10rlv7584 | -0.276295 |
| multilevel_minus_gwn | p10rlv7860 | -0.664207 |
| multilevel_minus_gwn | p10rlv7861 | 1.671288 |
| multilevel_minus_gwn | p10rlv8456 | 0.525105 |
| multilevel_minus_gwn | p10rlv9050 | 0.273415 |
| multilevel_minus_gwn | p10rlv9051 | 0.563765 |
| multilevel_minus_gwn | p10rlv9646 | -1.933830 |
| multilevel_minus_gwn | p10rlv9907 | -0.775192 |
| multilevel_minus_gwn | p10rlv9923 | -0.011209 |
| multilevel_minus_gwn | p10rlv10217 | -1.131623 |
| multilevel_minus_gwn | p10rlv11104 | 0.481043 |
| multilevel_minus_gwn | p10rlv11105 | 3.825757 |
| multilevel_minus_gwn | p10rlv11392 | 3.778173 |
| multilevel_minus_gwn | p10rlv12002 | 0.288490 |
| multilevel_minus_gwn | p10rlv12029 | 0.554148 |
| multilevel_minus_gwn | p10rlv12030 | -0.241196 |
| multilevel_minus_gwn | p10rlv12345 | 1.327787 |
| multilevel_minus_gwn | p10rlv13496 | -1.387413 |
| multilevel_minus_gwn | p10rlv13829 | 3.896427 |
| multilevel_minus_gwn | p10rlv15288 | -0.219948 |
| multilevel_minus_gwn | p10rlv15908 | 0.626801 |
| multilevel_minus_gwn | p10rlv16523 | 0.781982 |
| multilevel_minus_gwn | p10rlv17088 | 1.457386 |
| multilevel_minus_gwn | p10rlv17370 | 4.946363 |
| multilevel_minus_gwn | p10rlv17685 | 0.001831 |
| multilevel_minus_gwn | p10rlv18002 | 7.118359 |
| multilevel_minus_gwn | p10rlv18296 | 0.041138 |
| multilevel_minus_gwn | p10rlv18297 | -1.064841 |
| multilevel_minus_gwn | p10rlv18611 | 1.478081 |
| multilevel_minus_gwn | p10rlv18908 | -0.728497 |
| multilevel_minus_gwn | p10rlv19213 | 1.482109 |
| multilevel_minus_gwn | p10rlv19495 | 0.129368 |
| multilevel_minus_gwn | p10rlv19496 | 0.511936 |
| multilevel_minus_gwn | p10rlv19818 | 1.415422 |
| multilevel_minus_gwn | p10rlv20089 | -1.885691 |
| multilevel_minus_gwn | p10rlv20103 | -0.626190 |
| multilevel_minus_gwn | p10rlv20105 | 4.869461 |
| multilevel_minus_gwn | p10rlv20976 | 0.403128 |
| multilevel_minus_gwn | p10rlv21568 | -0.369701 |
| multilevel_minus_gwn | p10rlv21569 | 0.514099 |
| multilevel_minus_gwn | p10rlv21869 | 4.267412 |
| multilevel_minus_gwn | p10rlv22134 | 3.699916 |
| multilevel_minus_gwn | p10rlv22442 | 0.246706 |
| multilevel_minus_gwn | p10rlv23067 | 5.097631 |
| multilevel_minus_gwn | p10rlv23375 | 1.013531 |
| multilevel_minus_gwn | p10rlv24009 | 1.735558 |
| multilevel_minus_gwn | p10rlv24610 | 1.349247 |
| multilevel_minus_gwn | p10rlv24611 | 0.603451 |
| multilevel_minus_gwn | p10rlv24936 | 0.203636 |
| multilevel_minus_gwn | p10rlv25536 | -0.778996 |
| multilevel_minus_gwn | p10rlv26432 | 1.531216 |
| multilevel_minus_gwn | p10rlv26433 | 0.332886 |
| multilevel_minus_gwn | p10rlv27290 | -0.054161 |
| multilevel_minus_gwn | p10rlv27605 | 7.051208 |
| multilevel_minus_gwn | p10rlv27606 | 0.746239 |
| multilevel_minus_gwn | p10rlv28435 | 1.842827 |
| multilevel_minus_gwn | p10rlv28744 | -0.042019 |
| multilevel_minus_gwn | p10rlv28745 | -0.371659 |
| multilevel_minus_gwn | p10rlv29054 | -0.009274 |
| multilevel_minus_gwn | p10rlv29354 | 0.138756 |
| multilevel_minus_gwn | p10rlv29657 | 0.464630 |
| multilevel_minus_gwn | p10rlv29902 | -1.104954 |
| multilevel_minus_gwn | p10rlv29910 | 0.023998 |
| multilevel_minus_gwn | p10rlv29912 | 4.408396 |
| multilevel_minus_gwn | p10rlv29913 | 0.491661 |
| multilevel_minus_gwn | p10rlv29914 | 0.588671 |
| multilevel_minus_gwn | p10rlv30210 | -1.110312 |
| multilevel_minus_gwn | p10rlv30211 | -0.959465 |
| multilevel_minus_gwn | p10rlv30788 | 0.284543 |

## Node Difficulty

- PA-STFed: {'min': 2.762939929962158, 'p10': 12.90864372253418, 'p25': 25.412681579589844, 'median': 29.858762741088867, 'p75': 40.6666374206543, 'p90': 46.49767303466797, 'max': 49.01484680175781}; counts={'lt10': 8, 'lt20': 11, 'lt30': 47, 'ge40': 29}
- PA-STFed residual-anchor: {'min': 3.107927083969116, 'p10': 12.687405586242676, 'p25': 24.962587356567383, 'median': 29.210411071777344, 'p75': 40.534996032714844, 'p90': 46.20293045043945, 'max': 51.57440948486328}; counts={'lt10': 8, 'lt20': 12, 'lt30': 49, 'ge40': 27}
- PA-STFed residual-scale-loss: {'min': 2.966158390045166, 'p10': 12.760388374328613, 'p25': 25.166942596435547, 'median': 30.08745574951172, 'p75': 40.65315628051758, 'p90': 47.164737701416016, 'max': 56.314857482910156}; counts={'lt10': 5, 'lt20': 12, 'lt30': 44, 'ge40': 27}
- PA-STFed residual-multilevel-loss: {'min': 2.9211020469665527, 'p10': 12.846092224121094, 'p25': 25.29302215576172, 'median': 29.420238494873047, 'p75': 41.36602020263672, 'p90': 47.111663818359375, 'max': 56.19011688232422}; counts={'lt10': 5, 'lt20': 12, 'lt30': 49, 'ge40': 29}
- PA-STFed residual-multilevel-lambda0.02: {'min': 3.2899539470672607, 'p10': 13.094197273254395, 'p25': 25.393531799316406, 'median': 29.494644165039062, 'p75': 40.71490478515625, 'p90': 47.05120086669922, 'max': 55.75004959106445}; counts={'lt10': 5, 'lt20': 12, 'lt30': 50, 'ge40': 29}
- PA-STFed residual-multilevel-lambda0.05: {'min': 2.977898359298706, 'p10': 12.520881652832031, 'p25': 25.3557071685791, 'median': 29.725997924804688, 'p75': 40.629051208496094, 'p90': 47.267295837402344, 'max': 56.28422546386719}; counts={'lt10': 5, 'lt20': 12, 'lt30': 49, 'ge40': 29}
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
- PA-STFed residual-scale-loss: {'train_cv': 0.46554943243229185, 'lag96_autocorr': -0.622654635654078, 'lag672_autocorr': -0.6425279042461175, 'mean_shift': -0.3155209012692692}
- PA-STFed residual-multilevel-loss: {'train_cv': 0.4367339638531439, 'lag96_autocorr': -0.6449330476909261, 'lag672_autocorr': -0.6593793259459616, 'mean_shift': -0.3144725054087117}
- PA-STFed residual-multilevel-lambda0.02: {'train_cv': 0.4533232865878487, 'lag96_autocorr': -0.6272644939232943, 'lag672_autocorr': -0.6470452569982257, 'mean_shift': -0.31251447196325854}
- PA-STFed residual-multilevel-lambda0.05: {'train_cv': 0.45875027692485243, 'lag96_autocorr': -0.6234255149633114, 'lag672_autocorr': -0.642265805280978, 'mean_shift': -0.30978555920857204}
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
