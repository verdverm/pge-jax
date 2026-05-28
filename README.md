# pge-jax

JAX implementation of the **Prioritized Grammar Enumeration (PGE)** algorithm for symbolic regression.

## Overview

pge-jax is a complete symbolic regression system that automatically discovers mathematical formulas from data. It enumerates candidate expressions from a grammar, fits their coefficients using JAX-native Levenberg-Marquardt optimization, and selects the best models using a multi-objective pareto front (NSGA-II).

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
    peek_fraction=0.25,   # 25% of data for fast partial evaluation
)
pge.fit(X, Y)

# Get results
best = pge.get_best_model()
print(best.pretty_expr())  # e.g. "3.0*x0 + 1.5*x1**2 - 0.5*sin(x0)"

paretos = pge.get_final_paretos()  # list of Pareto fronts
```

## Algorithm

### Symbolic Regression

Given data points $(x_1, y_1), \dots, (x_N, y_N)$, symbolic regression searches the space of all valid mathematical expressions for one that fits the data well — but unlike standard regression, it returns an interpretable formula rather than opaque coefficients.

The search space is combinatorial: with $V$ variables, $F$ functions, and a max tree depth $D$, the number of candidate expressions grows exponentially. Exhaustive search is infeasible. Instead, symbolic regression uses an iterative loop that cycles through three phases:

```
  data       ┌──────────┐    ┌──────────┐    ┌──────────┐
 ──────►     │ Explore  │───►│ Evaluate │───►│  Select  │
             └──────────┘    └──────────┘    └──────────┘
                   ▲                               │
                   │       ┌───────────────┐       │
                   └───────│    "best"     │◄──────┘
                           │ C_0*x0+C_1    │        
                           │ C_0*x0+C_1*x1 │      
                           │ C_0*sin(x0)   │        
                           └───────────────┘          
```

**Explore** generates candidate expressions from a grammar of valid forms. **Evaluate** fits coefficients and computes accuracy. **Select** keeps the best candidates and feeds them back into exploration, gradually improving the set of discovered formulas.

### PGE Algorithm

Prioritized Grammar Enumeration (PGE) is a deterministic symbolic regression algorithm that replaces genetic operators and randomness with grammar production rules and systematic enumeration. PGE enumerates expressions in order of increasing complexity, pruning through multi-objective selection. Algebraic canonicalization normalizes operands using associativity and commutativity rules before memoization so equivalent expressions are detected as duplicates, and additional optimizations include early termination on diverging fits and intermediate value bounds checking.

PGE combines grammar-based expression generation with evolutionary multi-objective optimization:

1. **Grammar-based generation** — A context-free grammar defines valid expressions from variables, functions, constants, and arithmetic operators. The `Grower` class enumerates candidates using five expansion operators: variable substitution, addition extension, multiplication extension, coefficient-scaled functions, and shrink.

2. **Filtering** — Expressions are rejected for violating size limits, having integer coefficients, or exceeding power bounds.

3. **Memoization** — Structural deduplication skips expressions already explored, avoiding redundant work.

4. **Algebraic manipulation** — Valid expressions are expanded, factored, and simplified to discover equivalent forms that may have better coefficient fits.

5. **Progressive evaluation** — Candidates are first evaluated on a small data subset (`peek_fraction`). Only the most promising survive via NSGA-II selection to full evaluation on all training data.

6. **Multi-objective optimization** — Each candidate is scored on multiple objectives (RMSE, complexity, AIC/BIC). NSGA-II maintains a Pareto front of non-dominated solutions across iterations.

7. **Coefficient fitting** — Levenberg-Marquardt optimization tunes free coefficients for each expression, enabling fair comparison between structurally different candidates.

The result is a Pareto front of trade-off solutions — from simple approximate formulas to complex high-accuracy ones — from which the user can select the best interpretation for their problem.

## How It Works

### Two-Layer Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        PGE Search Loop                          │
│  (search.py — orchestration, selection, expansion)              │
│                                                                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐   │
│  │ Generate │───▶│  Filter  │───▶│ Memoize  │───▶│  Algebra │   │
│  │ (Grower) │    │          │    │          │    │          │   │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘   │
│                                                        │        │
│                                                        ▼        │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐   │
│  │  Final   │◀───│  Full    │◀───│  Peek    │◀───│ Select   │   │
│  │  Push    │    │ Evaluate │    │ Evaluate │    │ (NSGA-II)│   │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘   │
└─────────────────────────────────────────────────────────────────┘
         │
         │ SearchModel (expression + state + fitness)
         │
┌────────┴────────────────────────────────────────────────────────┐
│                    Evaluation Layer                             │
│  (model.py + optimize.py + metrics.py)                          │
│                                                                 │
│  SearchModel ──▶ JAXModel ──▶ fit_model() ──▶ evaluate()        │
│                                    │              │             │
│                                    ▼              ▼             │
│                               FitResult      EvalResult         │
│                               (coeffs,       (RMSE, R²,         │
│                                cost)         AIC, BIC, ...)     │
└─────────────────────────────────────────────────────────────────┘
```

### Progressive Evaluation

```
All candidate expressions from Grower
            │
            ▼
    ┌───────────────┐
     │  Peek Eval    │  ← Uses peek_fraction of training data (off by default)
    │  (fast subset)│
    └───────┬───────┘
            │
            ▼
    ┌───────────────┐
    │ NSGA-II       │  ← Multi-objective selection keeps
    │ _peek_pop()   │     only the most promising candidates
    └───────┬───────┘
            │
            ▼
    ┌───────────────┐
    │  Full Eval    │  ← Uses ALL training data
    │  (expensive)  │     only on selected candidates
    └───────┬───────┘
            │
            ▼
    ┌───────────────┐
    │  _final_push()│  ← Accumulate into Pareto front
    └───────────────┘
```

### Evaluation Pipeline

```
sympy.Expr
    │
    ▼
JAXModel.__init__()
    │  - wraps sympy expression
    │  - compiles jax_fun + jac_fun via sympy.lambdify
    ▼
fit_levenberg_marquardt()
    │  - optimizes coefficients (C_0, C_1, ...)
    │  - uses JAX-computed Jacobian
    ▼
FitResult
    │  - coefficients: [2.0, 1.0]
    │  - predictions: model output at optimized coeffs
    │  - cost: sum of squared residuals
    ▼
evaluate()
    │  - computes regression metrics
    ▼
EvalResult
    │  - score (RMSE), r2, aic, bic, chisqr, ...
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
    peek_fraction=0.25,         # fraction of data for fast partial evaluation (0 = off)
    max_size=64,                # max expression tree size
    max_power=5,                # max power exponent
    algebra_methods=["expand", "factor"],
    err_method="mse",           # error metric for score
    random_seed=23,
    expanders=[                 # optional: multiple expanders with different configs
        ExpanderConfig(pop_count=3, max_size=32),
        ExpanderConfig(pop_count=2, max_size=64),
    ],
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

The search uses `peek_fraction` (default 0.0, off) to use a fraction of training data for fast partial evaluation of candidate expressions, then fully evaluates only the most promising ones on all training data. This dramatically reduces the number of expensive full evaluations when enabled.

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

## Note from "Author"

> [!NOTE]
> This project is an experiment in using agents to modernize my PhD work.
> The stack is OpenCode + Qwen-3.6 35B A3B MoE (unsloth@UD-Q8_K_XL) + DGX Spark.
> Things seem to be working as before so far, but it is largely a blackbox reimplementation.
> You can see history in the .sessions/ directory, but OpenCode doesn't include user message in /export... ¯\\\_(ツ)\_/¯
> The intention is to continue this experiment into new research with open weight qwen36moe (unless we get a simlar size 3.7 soon)

> [!TIP]
> Little qwen36moe is an awesome model and my daily driver

## Prior Art

- **pypge**: Original Python implementation — <https://github.com/verdverm/pypge>
- **go-pge**: Go implementation with performance optimizations — <https://github.com/verdverm/go-pge>
- **PySR**: Symbolic regression with JAX — <https://github.com/MilesCranmer/pysr>
- **FFX**: Fast feature selection for symbolic regression — <https://github.com/ffx-org/ffx>

## Citation

> "Prioritized Grammar Enumeration" — Best Paper, GECCO 2013
> <http://dl.acm.org/citation.cfm?id=2463486>
