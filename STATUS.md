# STATUS — PGE JAX Port

## Summary

The full PGE search loop has been ported from `pypge/pypge/` to `pge_jax/`. All 86 tests pass.

## What Was Built

### 8 New Modules

| Module | Lines | Description |
|---|---|---|
| `pge_jax/filters.py` | ~126 | Expression validity filters (6 filter functions + `default_filters`) |
| `pge_jax/algebra.py` | ~75 | Symbolic manipulation (`manip_model`, `do_simp` for expand/factor/simplify) |
| `pge_jax/memoize.py` | ~75 | Hash-based expression deduplication (`Memoizer` class) |
| `pge_jax/selection.py` | ~440 | NSGA-II, SPEA-II, tournament selection, log-ND sort (from DEAP) |
| `pge_jax/fitness_funcs.py` | ~145 | Multi-objective fitness construction (normalized + raw) |
| `pge_jax/search_model.py` | ~350 | Search-loop-aware model with state flags, size metrics, JAX wrapper |
| `pge_jax/expand.py` | ~480 | `Grower` class — grammar-based expression enumeration (5 operators) |
| `pge_jax/search.py` | ~690 | `PGE` class — main search loop orchestration |

### Updated

| File | Change |
|---|---|
| `pge_jax/__init__.py` | Added 20 new exports (filters, algebra, memoize, selection, fitness, expand, search) |
| `AGENTS.md` | Added full PGE architecture reference, porting order, design decisions |

### Tests

| File | Tests | Coverage |
|---|---|---|
| `tests/test_search.py` | 45 | SearchModel, filters, algebra, memoize, selection, fitness, expand |

**Total: 86 tests passing** (41 existing + 45 new)

## Architecture

```
sympy.Expr → SearchModel → Grower.first_exprs() → Filter → Memoize → Algebra
                                                        → Filter → Memoize
                                                        → PGE._eval_models() (peek)
                                                        → PGE._peek_pop() (NSGA-II)
                                                        → PGE._eval_models() (full)
                                                        → PGE._final_push() → final list
```

### Key Design Decisions

1. **No DEAP dependency** — fitness values stored as tuples on model objects (`fitness_values`, `wvalues`)
2. **No multiprocessing** — JAX/XLA handles parallelism; skipped for now
3. **No remote evaluation** — all computation is local via JAX
4. **No lmfit/sklearn** — replaced by JAX-native LM optimizer + JAX metrics
5. **Two model classes**: `JAXModel` (pure JAX evaluation) + `SearchModel` (search loop state)
6. **`sortLogNondominated`** expects `wvalues` as plain tuples (not objects)

## Remaining Work

### Low Priority

- **Integration test** — end-to-end `PGE.fit(X, Y)` with real data
- **`ExpanderConfig` usage** — multi-expander parameter wiring
- **Progress logging** — tqdm progress bars in the search loop
- **`print_best()`** — formatting improvements, column output
- **`get_best_model()`** — validation that it returns the RMSE-optimal model

### Not Ported (Out of Scope)

- `parallel.py` — multiprocessing workers (JAX/XLA handles this)
- `remote evaluation` — WebSocket-based remote workers
- `timer.py` — timing utilities (can use `time` module directly)
- `base.py` / `creator.py` — DEAP foundations (no longer needed)
- `benchmark problems` — Koza, Lipson, Nguyen benchmarks (can add later)

## Quick Start

```python
import numpy as np
from pge_jax import PGE

# Generate synthetic data
np.random.seed(42)
X = np.random.randn(100, 2)
Y = 3.0 * X[:, 0] + 1.5 * X[:, 1]**2 - 0.5 * np.sin(X[:, 0])

# Run PGE search
pge = PGE(
    usable_vars=["x0", "x1"],
    usable_funcs=["sin", "cos", "exp", "log"],
    max_iter=10,
    pop_count=3,
    peek_npts=16,
)
pge.fit(X, Y)

# Get results
paretos = pge.get_final_paretos()
best = pge.get_best_model()
print(best.pretty_expr())
```

## Test Commands

```bash
# All tests
python -m pytest tests/ -v

# New search loop tests only
python -m pytest tests/test_search.py -v

# Lint
ruff check pge_jax/ tests/

# Format
ruff format pge_jax/ tests/
```
