# Target Topology Audit

## Protocol

- Canonical source: `E:\pa_stfed_project\data\processed\smartds_full_graph_v2.npz`
- Distance graph: official 273-node Line+Transformer graph.
- Targets: 92 non-zero load nodes; structural nodes remain shortest-path relays.
- Candidate k values: 2, 4, 6, 8 (diagnostic only; no k is selected here).
- Symmetrization: retain an undirected edge when either endpoint selects the other.
- Tie break: ascending `node_id` after ascending official shortest-path hop distance.
- Client partition: deterministic official-tree partition from seven cut edges.
- Partition objective: minimize total 10--13 range violation, then total squared deviation from 11.5, then the number of out-of-range regions.
- Partition inputs: official adjacency, target positions, and node IDs only.
- Client partition SHA-256: `d50d8071b19b22d79828c72cb76f1308ce373c74c0685e780e99e73a85b8b1a6`

## Topology-based Client Partition

- Target counts by client: `[15, 12, 9, 10, 11, 12, 11, 12]`
- Legacy duplicate-aware target counts: `[10, 11, 12, 11, 12, 12, 12, 12]`
- Duplicate load-curve groups: 18
- Duplicate groups crossing new clients: 16
- Duplicate groups crossing legacy clients: 0

The exact tree dynamic program found no seven-edge cut whose eight region counts all lie in 10--13. The minimum total range violation is 3 targets, yielding [15, 12, 9, 10, 11, 12, 11, 12].

The duplicate-curve statistics are computed only after the topology partition is frozen; they do not affect any cut decision.

### Seven Cut Official Tree Edges

| Left index | Left node ID | Right index | Right node ID | Type | Official source |
|---:|---|---:|---|---|---|
| 105 | `p10rdt13742` | 41 | `p10rdt2638` | line | `p10rhs0_1247/p10rhs0_1247--p10rdt7719/Lines.dss:361` |
| 119 | `p10rdt16751` | 256 | `p10rdt16751-p10rdt4426x` | line | `p10rhs0_1247/p10rhs0_1247--p10rdt7719/Lines.dss:355` |
| 125 | `p10rdt17567` | 75 | `p10rdt6810` | line | `p10rhs0_1247/p10rhs0_1247--p10rdt7719/Lines.dss:319` |
| 137 | `p10rdt19045` | 85 | `p10rdt8556` | line | `p10rhs0_1247/p10rhs0_1247--p10rdt7719/Lines.dss:281` |
| 51 | `p10rdt3891` | 260 | `p10rdt3891-p10rdt9361x` | line | `p10rhs0_1247/p10rhs0_1247--p10rdt7719/Lines.dss:399` |
| 67 | `p10rdt6364` | 69 | `p10rdt6599` | line | `p10rhs0_1247/p10rhs0_1247--p10rdt7719/Lines.dss:333` |
| 79 | `p10rdt7719` | 268 | `p10rdt7719-p10rdt9792x` | line | `p10rhs0_1247/p10rhs0_1247--p10rdt7719/Lines.dss:389` |

## Global Target Graph Statistics

| k | Edges | Density | Avg degree | Max degree | Components | Isolated | Hop min | Hop median | Hop mean | Hop P75 | Hop P90 | Hop max | Cross-client ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 120 | 0.028667 | 2.608696 | 7 | 10 | 0 | 2.000000 | 5.000000 | 4.791667 | 6.000000 | 7.000000 | 10.000000 | 0.083333 |
| 4 | 242 | 0.057812 | 5.260870 | 11 | 2 | 0 | 2.000000 | 6.000000 | 5.867769 | 7.000000 | 8.000000 | 13.000000 | 0.148760 |
| 6 | 352 | 0.084090 | 7.652174 | 16 | 1 | 0 | 2.000000 | 7.000000 | 6.559659 | 8.000000 | 9.000000 | 16.000000 | 0.204545 |
| 8 | 455 | 0.108696 | 9.891304 | 23 | 1 | 0 | 2.000000 | 7.000000 | 7.129670 | 8.500000 | 10.000000 | 17.000000 | 0.261538 |

## Client-induced Subgraph Statistics

| k | Client | Targets | Edges | Density | Avg degree | Max degree | Components | Isolated |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0 | 15 | 19 | 0.180952 | 2.533333 | 7 | 3 | 1 |
| 2 | 1 | 12 | 12 | 0.181818 | 2.000000 | 3 | 3 | 0 |
| 2 | 2 | 9 | 11 | 0.305556 | 2.444444 | 3 | 1 | 0 |
| 2 | 3 | 10 | 12 | 0.266667 | 2.400000 | 5 | 1 | 0 |
| 2 | 4 | 11 | 13 | 0.236364 | 2.363636 | 4 | 2 | 0 |
| 2 | 5 | 12 | 15 | 0.227273 | 2.500000 | 4 | 3 | 0 |
| 2 | 6 | 11 | 13 | 0.236364 | 2.363636 | 4 | 2 | 0 |
| 2 | 7 | 12 | 15 | 0.227273 | 2.500000 | 4 | 2 | 0 |
| 4 | 0 | 15 | 37 | 0.352381 | 4.933333 | 7 | 1 | 0 |
| 4 | 1 | 12 | 26 | 0.393939 | 4.333333 | 6 | 1 | 0 |
| 4 | 2 | 9 | 20 | 0.555556 | 4.444444 | 6 | 1 | 0 |
| 4 | 3 | 10 | 20 | 0.444444 | 4.000000 | 7 | 1 | 0 |
| 4 | 4 | 11 | 21 | 0.381818 | 3.818182 | 7 | 2 | 0 |
| 4 | 5 | 12 | 29 | 0.439394 | 4.833333 | 8 | 1 | 0 |
| 4 | 6 | 11 | 25 | 0.454545 | 4.545455 | 8 | 1 | 0 |
| 4 | 7 | 12 | 28 | 0.424242 | 4.666667 | 6 | 1 | 0 |
| 6 | 0 | 15 | 50 | 0.476190 | 6.666667 | 8 | 1 | 0 |
| 6 | 1 | 12 | 34 | 0.515152 | 5.666667 | 9 | 1 | 0 |
| 6 | 2 | 9 | 27 | 0.750000 | 6.000000 | 8 | 1 | 0 |
| 6 | 3 | 10 | 26 | 0.577778 | 5.200000 | 7 | 1 | 0 |
| 6 | 4 | 11 | 34 | 0.618182 | 6.181818 | 10 | 1 | 0 |
| 6 | 5 | 12 | 42 | 0.636364 | 7.000000 | 11 | 1 | 0 |
| 6 | 6 | 11 | 31 | 0.563636 | 5.636364 | 8 | 1 | 0 |
| 6 | 7 | 12 | 36 | 0.545455 | 6.000000 | 9 | 1 | 0 |
| 8 | 0 | 15 | 57 | 0.542857 | 7.600000 | 14 | 1 | 0 |
| 8 | 1 | 12 | 47 | 0.712121 | 7.833333 | 11 | 1 | 0 |
| 8 | 2 | 9 | 32 | 0.888889 | 7.111111 | 8 | 1 | 0 |
| 8 | 3 | 10 | 33 | 0.733333 | 6.600000 | 9 | 1 | 0 |
| 8 | 4 | 11 | 42 | 0.763636 | 7.636364 | 10 | 1 | 0 |
| 8 | 5 | 12 | 48 | 0.727273 | 8.000000 | 11 | 1 | 0 |
| 8 | 6 | 11 | 38 | 0.690909 | 6.909091 | 9 | 1 | 0 |
| 8 | 7 | 12 | 39 | 0.590909 | 6.500000 | 10 | 1 | 0 |

## Scope

These tables are topology diagnostics only. They contain no model accuracy, do not select k, and do not change the formal graph configuration.
