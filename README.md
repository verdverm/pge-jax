# pge-jax

JAX implementation of the **Prioritized Grammar Enumeration (PGE)** algorithm for symbolic regression.

## Overview

pge-jax is a complete symbolic regression system that automatically discovers mathematical formulas from data. It enumerates candidate expressions from a grammar, fits their coefficients using JAX-native Levenberg-Marquardt optimization, and selects the best models using multi-objective evolutionary algorithms (NSGA-II).

The key advantage over prior implementations (pypge, go-pge) is a fully JAX-native evaluation pipeline, enabling:

- **GPU/TPU acceleration** of model evaluation and Jacobian computation
- **JIT compilation** via `jax.jit` and `jax.vmap`
- **Automatic differentiation** for efficient gradient-based optimization
- **No external ML dependencies** — no scikit-learn, lmfit, or DEAP required

## Quick Start

```python
import jax
jax.config.update("jax_enable_x64", True)

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
    peek_npts=16,       # subset size for fast partial evaluation
)
pge.fit(X, Y)

# Get results
best = pge.get_best_model()
print(best.pretty_expr())  # e.g. "3.0*x0 + 1.5*x1**2 - 0.5*sin(x0)"

paretos = pge.get_final_paretos()  # list of Pareto fronts
```

## Architecture

```
pge_jax/
├── __init__.py          # Public API exports
├── model.py             # JAXModel — sympy expression → JAX evaluation wrapper
├── optimize.py          # Levenberg-Marquardt + BFGS optimizers (JAX-native)
├── metrics.py           # JAX-native regression metrics (RMSE, R², AIC, etc.)
├── evaluate.py          # High-level fit/predict/evaluate pipeline
├── search_model.py      # SearchModel — expression state, size metrics, fitness
├── filters.py           # Expression validity filters (size, coefficients, powers)
├── algebra.py           # Symbolic expand/factor/simplify
├── memoize.py           # Hash-based expression deduplication
├── fitness_funcs.py     # Multi-objective fitness construction
├── selection.py         # NSGA-II, SPEA-II, log-ND sort (from DEAP)
├── expand.py            # Grower — grammar-based expression enumeration
└── search.py            # PGE — main search loop orchestration
```

## Public API

### `PGE` — End-to-End Search

The primary entry point. Wraps the full search pipeline:

```python
from pge_jax import PGE

pge = PGE(
    usable_vars=["x0", "x1"],   # or "x0 x1" or list of sympy.Symbol
    usable_funcs=["sin", "cos", "exp", "log", "tan", "sqrt"],
    max_iter=100,               # search iterations
    pop_count=3,                # models expanded per expander per iteration
    peek_count=6,               # models selected from peek heap for full eval
    peek_npts=16,               # data points for fast partial (peek) evaluation
    max_size=64,                # max expression tree size
    max_power=5,                # max power exponent
    algebra_methods=["expand", "factor"],
    err_method="mse",           # error metric for score
    random_seed=23,
)

pge.fit(X_train, Y_train)          # sklearn-style: returns self
best = pge.get_best_model()        # SearchModel with lowest score
paretos = pge.get_final_paretos()  # list[list[SearchModel]]
```

### `SearchModel` — Expression State

Wraps a sympy expression with lifecycle state, size metrics, and fitness:

```python
from pge_jax import SearchModel
import sympy

x = sympy.Symbol("x")
c0 = sympy.Symbol("C_0")
expr = c0 * x + 1

model = SearchModel(expr, xs=[x], cs=[c0])
model.rewrite_coeff()   # builds JAXModel wrapper, converts bare C → C_0, C_1, ...

model.size()            # tree size
model.psz               # penalised size (+2 per function node)
model.jpsz              # penalised Jacobian size
model.score             # RMSE after fitting
model.r2                # R-squared
model.pretty_expr()     # expression with fitted coefficients substituted
```

### `JAXModel` — Pure JAX Evaluation

For users who want to evaluate a specific expression without the search loop:

```python
from pge_jax import JAXModel, fit_model, evaluate
import jax.numpy as jnp
import sympy

x = sympy.Symbol("x")
c0, c1 = sympy.symbols("C_0 C_1")
expr = c0 * x + c1

model = JAXModel(expr)
result = fit_model(model, jnp.array([3.0, 5.0, 7.0]), jnp.array(x_data))
print(result.coefficients)  # [2.0, 1.0]

eval_result = evaluate(model, jnp.array(y_true), result.predictions)
print(f"R²: {eval_result.r2:.4f}")
```

### Individual Components

All modules are importable independently:

```python
from pge_jax import (
    # Filters
    filter_models, default_filters,

    # Algebra
    manip_model, do_simp,

    # Memoization
    Memoizer,

    # Selection
    selNSGA2, selSPEA2, sortLogNondominated, isDominated, assignCrowdingDist,

    # Fitness
    build_fitness_calc, build_fitness_weights, build_value_extractor,

    # Expansion
    Grower, map_names_to_funcs,

    # Metrics
    rmse, mae, mse, r2, explained_variance, aic, bic, chisqr, redchi, rmae,

    # Optimizers
    fit_levenberg_marquardt, fit_least_squares,
)
```

## Design Decisions

### Why sympy + JAX?

Sympy handles expression representation, tree manipulation, expansion, and simplification. JAX replaces the evaluation and optimization pipeline, providing GPU acceleration and automatic differentiation.

### No DEAP Dependency

Fitness values are stored directly as tuples on `SearchModel` objects (`fitness_values`, `wvalues`, `crowding_dist`). Selection functions expect these attributes instead of DEAP's `Fitness` wrapper.

### Two Model Classes

- **`JAXModel`** — Pure JAX evaluation wrapper (sympy → JAX, predict, jacobian)
- **`SearchModel`** — Search-loop state machine (lifecycle flags, size metrics, fitness, selection compatibility)

### Levenberg-Marquardt

The default optimizer because it's well-suited for least-squares problems, uses the JAX-computed Jacobian efficiently, and matches the original pypge approach.

### Progressive Evaluation

The search uses `peek_npts` (default 16) data points for fast partial evaluation of candidate expressions, then fully evaluates only the most promising ones on all training data. This dramatically reduces the number of expensive full evaluations.

## Installation

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install from source
pip install -e ".[dev]"

# Or with benchmark dependencies (for pypge compatibility)
pip install -e ".[dev,benchmark]"
```

### Optional Dependencies

| Group | Packages | Purpose |
|-------|----------|---------|
| `dev` | pytest, ruff, mypy, pre-commit | Development tooling |
| `benchmark` | scikit-learn, lmfit, pandas | Compatibility with pypge benchmarks |
| `notebook` | jupyter, matplotlib | Interactive exploration |

## Development

```bash
# Run all tests
python -m pytest tests/ -v

# Lint
ruff check pge_jax/ tests/

# Format
ruff format pge_jax/ tests/

# Type check
mypy pge_jax/
```

### Configuration

- **Line length:** 120 characters
- **Target Python:** 3.11+
- **Linter:** ruff (E, F, I, N, W, NPY rules)
- **Type checker:** mypy (non-strict mode)
- **JAX float64:** Enable with `jax.config.update("jax_enable_x64", True)` or `JAX_ENABLE_X64=1` env var

## Prior Art

- **pypge**: Original Python implementation — <https://github.com/verdverm/pypge>
- **go-pge**: Go implementation with performance optimizations — <https://github.com/verdverm/go-pge>
- **PySR**: Symbolic regression with JAX — <https://github.com/MilesCranmer/pysr>
- **FFX**: Fast feature selection for symbolic regression — <https://github.com/ffx-org/ffx>

## Citation

> "Prioritized Grammar Enumeration" — Best Paper, GECCO 2013
> <http://dl.acm.org/citation.cfm?id=2463486>
