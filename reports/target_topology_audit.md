# Target Topology Audit

## Protocol

- Canonical source: `E:\pa_stfed_project\data\processed\smartds_full_graph_v2.npz`
- Distance graph: official 273-node Line+Transformer graph.
- Targets: 92 non-zero load nodes; structural nodes remain shortest-path relays.
- Candidate k values: 2, 4, 6, 8 (diagnostic only; no k is selected here).
- Symmetrization: retain an undirected edge when either endpoint selects the other.
- Tie break: ascending `node_id` after ascending official shortest-path hop distance.
- Client partition: existing `SmartDS.client_partitions(8)`.
- Client partition SHA-256: `2b3737ccf4ea25943bd5c392aaa7cc5dcaecc98ec43dab4554be4b4da79139a3`

## Global Target Graph Statistics

| k | Edges | Density | Avg degree | Max degree | Components | Isolated | Hop min | Hop median | Hop mean | Hop P75 | Hop P90 | Hop max | Cross-client ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 120 | 0.028667 | 2.608696 | 7 | 10 | 0 | 2.000000 | 5.000000 | 4.791667 | 6.000000 | 7.000000 | 10.000000 | 0.600000 |
| 4 | 242 | 0.057812 | 5.260870 | 11 | 2 | 0 | 2.000000 | 6.000000 | 5.867769 | 7.000000 | 8.000000 | 13.000000 | 0.648760 |
| 6 | 352 | 0.084090 | 7.652174 | 16 | 1 | 0 | 2.000000 | 7.000000 | 6.559659 | 8.000000 | 9.000000 | 16.000000 | 0.667614 |
| 8 | 455 | 0.108696 | 9.891304 | 23 | 1 | 0 | 2.000000 | 7.000000 | 7.129670 | 8.500000 | 10.000000 | 17.000000 | 0.694505 |

## Client-induced Subgraph Statistics

| k | Client | Targets | Edges | Density | Avg degree | Max degree | Components | Isolated |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0 | 10 | 9 | 0.200000 | 1.800000 | 5 | 4 | 2 |
| 2 | 1 | 11 | 7 | 0.127273 | 1.272727 | 2 | 5 | 1 |
| 2 | 2 | 12 | 4 | 0.060606 | 0.666667 | 1 | 8 | 4 |
| 2 | 3 | 11 | 6 | 0.109091 | 1.090909 | 2 | 7 | 5 |
| 2 | 4 | 12 | 3 | 0.045455 | 0.500000 | 1 | 9 | 6 |
| 2 | 5 | 12 | 6 | 0.090909 | 1.000000 | 2 | 7 | 3 |
| 2 | 6 | 12 | 4 | 0.060606 | 0.666667 | 2 | 8 | 5 |
| 2 | 7 | 12 | 9 | 0.136364 | 1.500000 | 3 | 5 | 2 |
| 4 | 0 | 10 | 16 | 0.355556 | 3.200000 | 6 | 3 | 2 |
| 4 | 1 | 11 | 13 | 0.236364 | 2.363636 | 5 | 3 | 1 |
| 4 | 2 | 12 | 8 | 0.121212 | 1.333333 | 2 | 5 | 1 |
| 4 | 3 | 11 | 7 | 0.127273 | 1.272727 | 2 | 6 | 3 |
| 4 | 4 | 12 | 7 | 0.106061 | 1.166667 | 3 | 6 | 2 |
| 4 | 5 | 12 | 9 | 0.136364 | 1.500000 | 3 | 6 | 3 |
| 4 | 6 | 12 | 9 | 0.136364 | 1.500000 | 4 | 6 | 4 |
| 4 | 7 | 12 | 16 | 0.242424 | 2.666667 | 4 | 3 | 0 |
| 6 | 0 | 10 | 23 | 0.511111 | 4.600000 | 7 | 2 | 1 |
| 6 | 1 | 11 | 19 | 0.345455 | 3.454545 | 5 | 3 | 1 |
| 6 | 2 | 12 | 10 | 0.151515 | 1.666667 | 2 | 5 | 1 |
| 6 | 3 | 11 | 13 | 0.236364 | 2.363636 | 4 | 4 | 2 |
| 6 | 4 | 12 | 9 | 0.136364 | 1.500000 | 3 | 5 | 2 |
| 6 | 5 | 12 | 11 | 0.166667 | 1.833333 | 3 | 5 | 2 |
| 6 | 6 | 12 | 14 | 0.212121 | 2.333333 | 5 | 5 | 3 |
| 6 | 7 | 12 | 18 | 0.272727 | 3.000000 | 4 | 3 | 0 |
| 8 | 0 | 10 | 30 | 0.666667 | 6.000000 | 8 | 2 | 1 |
| 8 | 1 | 11 | 23 | 0.418182 | 4.181818 | 7 | 2 | 1 |
| 8 | 2 | 12 | 12 | 0.181818 | 2.000000 | 3 | 4 | 1 |
| 8 | 3 | 11 | 15 | 0.272727 | 2.727273 | 4 | 4 | 2 |
| 8 | 4 | 12 | 12 | 0.181818 | 2.000000 | 5 | 5 | 2 |
| 8 | 5 | 12 | 13 | 0.196970 | 2.166667 | 3 | 5 | 2 |
| 8 | 6 | 12 | 15 | 0.227273 | 2.500000 | 5 | 5 | 3 |
| 8 | 7 | 12 | 19 | 0.287879 | 3.166667 | 5 | 2 | 0 |

## Scope

These tables are topology diagnostics only. They contain no model accuracy, do not select k, and do not change the formal graph configuration.
