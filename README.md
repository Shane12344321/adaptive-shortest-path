# Adaptive Shortest-Path Framework (ASPF)

> **Runtime algorithm selection for single-source shortest-path problems.**
>
> Instead of hard-coding a single algorithm, ASPF inspects the input graph at
> runtime and automatically dispatches to the best strategy — BFS, Dial's
> bucket queue, array-based Dijkstra, or heap-based Dijkstra — with negligible
> analysis overhead.

---

## Table of Contents

1. [Project Description](#project-description)
2. [Architecture](#architecture)
3. [Algorithm Portfolio](#algorithm-portfolio)
4. [Key Design Decisions](#key-design-decisions)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Demo Command](#demo-command)
8. [Test Command](#test-command)
9. [Configuration](#configuration)
10. [Project Structure](#project-structure)

---

## Project Description

The Adaptive Shortest-Path Framework (ASPF) solves the Single-Source Shortest
Path (SSSP) problem with non-negative edge weights.  Rather than always
applying a general-purpose algorithm, ASPF performs a two-tier analysis of the
input graph and selects the most suitable algorithm from a portfolio of four:

| Input characteristic            | Selected algorithm        |
|---------------------------------|---------------------------|
| Unit-weight (metadata flag)     | BFS                       |
| Small integer weights           | Dial's bucket queue       |
| Dense graph, moderate size      | Array-scan Dijkstra       |
| Everything else                 | Heap Dijkstra (default)   |

The framework is built around three design patterns:

* **Adapter pattern** — a zero-copy `GraphView` interface decouples the
  analysis and solver code from any specific graph representation.
* **Strategy pattern** — each algorithm implements the same
  `ShortestPathSolver` interface.
* **Algorithm Selection Problem** (Rice 1976) — graph features are mapped to
  strategies through a decision tree with machine-calibrated thresholds.

---

## Architecture

```
External Graph Data
        │
        ▼
┌───────────────────┐   zero-copy reference
│  Input Adapters   │──────────────────────────────────────────────┐
│  (adapters.py)    │                                              │
│  AdjListAdapter   │   GraphView interface                        │
│  NetworkXAdapter  │                                              │
└────────┬──────────┘                                              │
         │ GraphView                                               │
         ▼                                                         │
┌───────────────────────────────────────────────┐                  │
│          Two-Tier Graph Analyzer              │                  │
│              (analyzer.py)                   │                  │
│                                               │                  │
│  Tier 1 — O(1)                                │                  │
│    ├─ Read V, E from header                   │                  │
│    ├─ Compute density = E / (V*(V-1))         │                  │
│    └─ Check metadata["unweighted"] → BFS      │                  │
│                                               │                  │
│  Tier 2 — O(√E), only if Tier 1 inconclusive │                  │
│    ├─ Reservoir-sample √E edges               │                  │
│    ├─ Detect is_unit_weight_sample            │                  │
│    ├─ Detect is_integer, max_weight_sample    │                  │
│    └─ Compute degree_variance_sample          │                  │
└────────┬──────────────────────────────────────┘                  │
         │ feature dict                                            │
         ▼                                                         │
┌───────────────────────────────────────────────┐                  │
│         Strategy Selector                     │                  │
│            (selector.py)                      │                  │
│                                               │                  │
│  Loads thresholds from default_config.json    │                  │
│  Decision tree:                               │                  │
│    1. is_unit_weight (metadata only) → BFS    │                  │
│    2. is_integer + small max_weight  → DIAL   │                  │
│    3. dense + small V               → ARRAY   │                  │
│    4. else (safe default)           → HEAP    │                  │
└────────┬──────────────────────────────────────┘                  │
         │ strategy string                                         │
         ▼                                                         │
┌───────────────────────────────────────────────┐                  │
│          Execution Engine                     │◄─────────────────┘
│             (solvers.py)                      │   GraphView (zero-copy)
│                                               │
│  BFSSolver          HeapDijkstraSolver        │
│  ArrayDijkstraSolver  DialSolver              │
└────────┬──────────────────────────────────────┘
         │ dist dict  {vertex → float}
         ▼
┌───────────────────────────────────────────────┐
│       Output + Benchmarking                   │
│          (benchmark.py)                       │
│                                               │
│  BenchmarkRunner.run(graph, source)           │
│    • Median timing (perf_counter_ns)          │
│    • 95% Bootstrap CI                         │
│    • Optional compare_all mode                │
└───────────────────────────────────────────────┘
```

---

## Algorithm Portfolio

| Strategy             | When Selected                                             | Complexity      |
|----------------------|-----------------------------------------------------------|-----------------|
| **BFS**              | `metadata["unweighted"] is True` (Tier 1 only)           | O(V + E)        |
| **DIAL**             | Integer weights, `max_weight ≤ bucket_threshold`         | O(V + E + C)    |
| **ARRAY_DIJKSTRA**   | `density > dense_threshold` and `V ≤ size_limit`         | O(V²)           |
| **HEAP_DIJKSTRA**    | All other cases (safe general-purpose default)           | O((V+E) log V)  |

Where C = maximum edge weight for Dial's algorithm.

---

## Key Design Decisions

### BFS Safety Invariant (Critical Correctness Requirement)

**BFS produces incorrect results on weighted graphs.**  Therefore:

> **BFS is dispatched *only* when the graph has an explicit `unweighted=True`
> metadata flag (Tier 1 check).  Tier 2 sampling may *suggest* unit weights
> (`is_unit_weight_sample=True`), but this information is purely informational
> and is never used to dispatch to BFS.**

This is a **correctness invariant**, not a performance optimisation.  A graph
without the metadata flag could have been sampled at edges that happen to all
weigh 1, while other edges carry different weights.  Dispatching to BFS in
that scenario would silently return wrong distances.

### Zero-Copy Adapter Layer

The `GraphView` adapters hold a *reference* to the original graph object.  No
adjacency data is ever duplicated.  This makes the framework safe to use on
graphs that occupy a large fraction of available memory.

### Two-Tier Lazy Analysis

| Tier   | Cost    | Action                                      |
|--------|---------|---------------------------------------------|
| Tier 1 | O(1)    | Metadata check + size/density from header   |
| Tier 2 | O(√E)   | Reservoir sampling of edges                 |

Tier 2 is only entered when Tier 1 is inconclusive.  Total analysis overhead
is at most O(√E), which is dominated by any shortest-path algorithm on
non-trivial graphs.

### Confidence Margin

When graph density falls within `confidence_margin` of `dense_threshold`
(i.e., in the interval `(dense_threshold, dense_threshold + confidence_margin]`),
the selector defaults to **HEAP_DIJKSTRA** instead of ARRAY_DIJKSTRA.  This
prevents misclassification near decision boundaries where cache effects make
the empirical crossover uncertain.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/Shane12344321/adaptive-shortest-path.git
cd adaptive-shortest-path

# Install dependencies
pip install -r requirements.txt
```

Dependencies:
* `networkx >= 3.0` — NetworkX adapter support
* `pytest >= 7.0` — test suite

---

## Usage

### Using the Adaptive Pipeline Directly

```python
from aspf.adapters import AdjListAdapter
from aspf.analyzer import GraphAnalyzer
from aspf.selector import StrategySelector
from aspf.benchmark import _solver_for_strategy

# Build a graph
adj = {
    0: [(1, 4), (2, 1)],
    1: [(3, 1)],
    2: [(1, 2), (3, 5)],
    3: [(4, 3)],
    4: [],
}
graph = AdjListAdapter(adj)

# Analyse → select → solve
features = GraphAnalyzer(seed=42).analyse(graph)
strategy = StrategySelector().select(features)
solver = _solver_for_strategy(strategy, features)
distances = solver.solve(graph, source=0)

print(f"Strategy: {strategy}")
print(f"Distances: {distances}")
```

### Using BenchmarkRunner

```python
from aspf.adapters import AdjListAdapter
from aspf.benchmark import BenchmarkRunner

graph = AdjListAdapter(adj)
runner = BenchmarkRunner(total_runs=10, warmup_runs=2)
result = runner.run(graph, source=0)

print(f"Strategy  : {result['strategy']}")
print(f"Distances : {result['distances']}")
print(f"Median ns : {result['median_ns']:.0f}")
print(f"95% CI    : {result['ci_95']}")
```

### Using a NetworkX Graph

```python
import networkx as nx
from aspf.adapters import NetworkXAdapter
from aspf.analyzer import GraphAnalyzer
from aspf.selector import StrategySelector

G = nx.DiGraph(unweighted=True)
G.add_edges_from([(0, 1), (1, 2), (2, 3)])
graph = NetworkXAdapter(G)

features = GraphAnalyzer().analyse(graph)
strategy = StrategySelector().select(features)
print(f"Strategy: {strategy}")   # → BFS
```

### Using an Unweighted Graph (BFS path)

```python
from aspf.adapters import AdjListAdapter

# The metadata flag is the ONLY safe path to BFS dispatch.
graph = AdjListAdapter(adj, meta={"unweighted": True})
```

---

## Demo Command

```bash
python -m aspf.main
```

This builds four synthetic graphs (unweighted, dense weighted, sparse
weighted, small-integer weighted), runs the full adaptive pipeline on each,
prints the selected strategy, sample distances, median timing, and 95%
confidence intervals, and finishes with a comparison-mode run on the sparse
graph.

---

## Test Command

```bash
pytest tests/ -v
```

The test suite covers 83 tests across six files:

| File                        | Coverage area                                          |
|-----------------------------|--------------------------------------------------------|
| `tests/test_adapters.py`    | AdjListAdapter, NetworkXAdapter, zero-copy, metadata  |
| `tests/test_analyzer.py`    | Tier 1 + Tier 2 analysis, BFS safety invariant        |
| `tests/test_selector.py`    | Decision tree, confidence margin, config loading       |
| `tests/test_solvers.py`     | All four solvers, cross-solver consistency             |
| `tests/test_benchmark.py`   | BenchmarkRunner, helper functions, timing              |
| `tests/test_integration.py` | End-to-end pipeline, NetworkX integration             |

---

## Configuration

Thresholds are loaded from `aspf/default_config.json`:

```json
{
    "bucket_threshold": 100,
    "dense_threshold": 0.5,
    "size_limit": 10000,
    "confidence_margin": 0.05
}
```

| Parameter           | Description                                                | Default |
|---------------------|------------------------------------------------------------|---------|
| `bucket_threshold`  | Maximum integer weight for DIAL to be selected             | 100     |
| `dense_threshold`   | Minimum density for ARRAY_DIJKSTRA to be considered        | 0.5     |
| `size_limit`        | Maximum vertex count for ARRAY_DIJKSTRA to be selected     | 10000   |
| `confidence_margin` | Density band near threshold that defaults to HEAP_DIJKSTRA | 0.05    |

To use a custom config:

```python
from aspf.selector import StrategySelector
selector = StrategySelector(config_path="my_config.json")
```

Missing or corrupt config files are silently ignored; built-in defaults apply.

---

## Project Structure

```
adaptive-shortest-path/
├── aspf/
│   ├── __init__.py          # Package exports
│   ├── adapters.py          # GraphView ABC + AdjListAdapter + NetworkXAdapter
│   ├── analyzer.py          # Two-tier GraphAnalyzer + strategy constants
│   ├── selector.py          # StrategySelector with JSON config
│   ├── solvers.py           # BFS, ArrayDijkstra, HeapDijkstra, Dial solvers
│   ├── benchmark.py         # BenchmarkRunner + helper functions
│   ├── main.py              # Demo entry point (python -m aspf.main)
│   └── default_config.json  # Default threshold configuration
├── tests/
│   ├── __init__.py
│   ├── test_adapters.py     # Adapter layer tests
│   ├── test_analyzer.py     # Analyzer tests (including BFS safety invariant)
│   ├── test_selector.py     # Strategy selector tests
│   ├── test_solvers.py      # Solver correctness + cross-solver consistency
│   ├── test_benchmark.py    # Benchmark runner + helper function tests
│   └── test_integration.py  # End-to-end pipeline integration tests
├── requirements.txt
└── README.md
```

---

## References

1. E. W. Dijkstra, "A note on two problems in connexion with graphs," *Numerische Mathematik*, vol. 1, 1959.
2. M. L. Fredman and R. E. Tarjan, "Fibonacci heaps and their uses in improved network optimization algorithms," *Journal of the ACM*, vol. 34, 1987.
3. R. B. Dial, "Algorithm 360: Shortest-path forest with topological ordering," *Communications of the ACM*, vol. 12, 1969.
4. J. R. Rice, "The algorithm selection problem," *Advances in Computers*, vol. 15, 1976.
5. E. Gamma, R. Helm, R. Johnson, and J. Vlissides, *Design Patterns: Elements of Reusable Object-Oriented Software*. Addison-Wesley, 1994.
