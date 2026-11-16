# pge-jax

JAX-based implementation of Prioritized Grammar Enumeration (PGE) for symbolic regression.

pge-jax enumerates mathematical expressions, fits their coefficients with JAX-native optimization, and maintains Pareto fronts that trade off accuracy and complexity. Sympy provides expression representation and manipulation; JAX provides prediction, automatic differentiation, and optimization.

## Features

- Grammar-based expression enumeration with configurable functions and growth policies
- Raw and expanded expression variants for broader search coverage
- JAX evaluation through `sympy.lambdify`, with automatic differentiation for Jacobians
- Levenberg-Marquardt fitting and a BFGS-based least-squares alternative
- Structural deduplication, expression filters, and algebraic transformations
- Optional partial-data coefficient fitting through progressive evaluation
- Multi-objective selection with NSGA-II, SPEA-II, and log-nondominated sorting
- Multiple independently configured expanders
- Per-model, per-stage, and per-iteration instrumentation

No scikit-learn, lmfit, or DEAP dependency is required by the core package.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Optional dependency groups:

| Group | Packages | Purpose |
|-------|----------|---------|
| `dev` | pytest, ruff, mypy, pre-commit | Development tooling |
| `benchmark` | scikit-learn, lmfit, pandas | Compatibility with pypge benchmarks |
| `notebook` | jupyter, matplotlib | Interactive exploration |

JAX enables 32-bit arithmetic by default on many platforms. Enable 64-bit arithmetic before importing numerical code when precision requires it:

```python
import jax

jax.config.update("jax_enable_x64", True)
```

The `JAX_ENABLE_X64=1` environment variable provides the same setting.

## Quick Start

```python
import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
from pge_jax import PGE

np.random.seed(42)
X = np.random.randn(100, 2)
Y = 3.0 * X[:, 0] + 1.5 * X[:, 1] ** 2 - 0.5 * np.sin(X[:, 0])

pge = PGE(
    usable_vars=["x0", "x1"],
    usable_funcs=["sin", "cos", "exp", "log"],
    max_iter=10,
    pop_count=3,
    peek_fraction=0.25,
)
pge.fit(X, Y)

best = pge.get_best_model()
print(best.pretty_expr())

fronts = pge.get_final_paretos()
```

`peek_fraction=0.0`, the default, disables progressive evaluation. A value strictly between 0 and 1 fits candidates on a random subset before selecting models for full-data fitting and evaluation. Do not use other nonzero values until the out-of-range dispatch issue in `REVIEW.md` is resolved.

## Algorithm

PGE uses grammar production rules to generate candidates rather than genetic mutation and crossover. Each search iteration:

1. Selects parent models with NSGA-II.
2. Applies growth operators to produce expression variants.
3. Rejects invalid or oversized expressions.
4. Deduplicates candidates and assigns model IDs.
5. Generates algebraic variants and filters and deduplicates them.
6. Optionally fits candidates on the peek subset and selects promising models.
7. Fits and evaluates selected models on all training data.
8. Adds evaluated models to the final results and search population.

The five growth operators are variable substitution, addition extension, multiplication extension, top-level addition extension, and shrinking. Each generated expression is rewritten from bare `C` placeholders to numbered coefficient symbols. Both its raw form and expanded form enter the candidate set because their different tree structures expose different future growth sites.

Coefficient fitting minimizes squared residuals. Fitness combines expression size and regression metrics, and Pareto selection preserves trade-offs rather than collapsing them into one score.

See `STATUS.md` for planned changes and `REVIEW.md` for current technical findings.

## Architecture

### Evaluation layer

- `JAXModel`: wraps a sympy expression, identifies coefficients and variables, and constructs prediction and Jacobian functions.
- `fit_model()`: fits coefficients with Levenberg-Marquardt by default or BFGS with `method="ls"`.
- `evaluate()`: computes RMSE, R-squared, explained variance, AIC, BIC, chi-squared, reduced chi-squared, MAE, and RMAE.
- `FitResult` and `EvalResult`: contain fitting and evaluation outputs.

### Search layer

- `SearchModel`: stores raw and lazily expanded expressions, lifecycle state, size metrics, fitness values, and evaluation results.
- `Grower`: constructs grammar term pools and generates candidate expressions.
- `PGE`: orchestrates generation, filtering, memoization, algebra, evaluation, and selection.
- Selection and fitness modules: implement multi-objective optimization without DEAP.

`SearchModel.expr` expands on first access. `SearchModel.build_jax_model()` constructs the evaluation wrapper on demand. During growth, coefficient rewriting is separated from JAX construction so rejected children do not require evaluation functions. Initial expressions and algebraic variants still use `rewrite_coeff()`, which constructs the JAX wrapper immediately.

## Usage

### Configure a search

```python
from pge_jax import ExpanderConfig, PGE

pge = PGE(
    usable_vars=["x0", "x1"],
    usable_funcs=["sin", "cos", "exp", "log", "tan", "sqrt"],
    max_iter=100,
    pop_count=3,
    peek_count=6,
    peek_fraction=0.25,
    max_size=64,
    max_power=5,
    algebra_methods=["expand", "factor"],
    random_seed=23,
)

# Optional: use separate expander populations and growth policies.
pge = PGE(
    usable_vars=["x0", "x1"],
    max_iter=100,
    multi_expander_params=[
        ExpanderConfig(pop_count=3, usable_funcs=["sin"], grow_params={"subs_level": "low"}),
        {"pop_count": 2, "usable_funcs": ["cos", "exp"], "grow_params": {"subs_level": "high"}},

    ],
)
```

`fit(X_train, Y_train)` returns the `PGE` instance. `get_best_model()` returns the lowest-scoring finalized model, and `get_final_paretos()` returns the computed Pareto fronts.

### Inspect a search model

```python
import sympy
from pge_jax import SearchModel

x = sympy.Symbol("x")
c0 = sympy.Symbol("C_0")
model = SearchModel(c0 * x + 1, xs=[x], cs=[c0])

model.size()
print(model.psz)
print(model.jpsz)

model.build_jax_model()
# Fit through fit_model(), then model.jax_model.c_values can be set.
print(model.pretty_expr())
```

`rewrite_coeff()` rewrites bare `C` atoms in `orig`, updates `cs`, and builds the JAX wrapper immediately. `rewrite_coeff_only(expr)` performs only the symbol transformation. Coefficient inference recognizes names beginning with `C_` or `C[`; a bare `C` is otherwise treated as an input variable.

### Evaluate an expression directly

```python
import jax.numpy as jnp
import sympy
from pge_jax import JAXModel, evaluate, fit_model

x = sympy.Symbol("x")
c0, c1 = sympy.symbols("C_0 C_1")
model = JAXModel(c0 * x + c1)

x_data = jnp.array([0.0, 1.0, 2.0])
y_data = jnp.array([1.0, 3.0, 5.0])
fit = fit_model(model, y_data, x_data)
metrics = evaluate(model, y_data, fit.predictions)

print(fit.coefficients)
print(fit.predictions)
print(metrics.r2)
```

### Use individual components

```python
from pge_jax import (
    Grower,
    Memoizer,
    SearchModel,
    aic,
    assignCrowdingDist,
    bic,
    build_fitness_calc,
    build_fitness_weights,
    build_value_extractor,
    default_filters,
    do_simp,
    explained_variance,
    filter_models,
    fit_least_squares,
    fit_levenberg_marquardt,
    isDominated,
    mae,
    manip_model,
    map_names_to_funcs,
    mse,
    r2,
    rmse,
    selNSGA2,
    selSPEA2,
    sortLogNondominated,
)
```

`chisqr`, `redchi`, and `rmae` are available from `pge_jax.metrics`.

## Numerical and evaluation behavior

- Evaluation arrays request `float64`; JAX truncates them with warnings when x64 is disabled.
- `sympy.lambdify(..., modules="jax")` falls back to NumPy if JAX lambdification fails.
- Levenberg-Marquardt starts from zero coefficients by default and uses a Python iteration loop around JAX operations.
- During peek, coefficients are fitted on the subset, but metrics are calculated on predictions over the full training data. Selected candidates are then fitted again with normal full evaluation.
- Memoization uses the hash of `SearchModel.orig`. Raw and expanded variants with different original structures remain distinct.
- Failing model builds, fits, or evaluations mark a model as errored and omit it from returned evaluation results.

## Development

```bash
source .venv/bin/activate

python -m pytest tests/ -v
python -m pytest tests/test_search.py -v
python -m pytest tests/test_search.py::TestSearchModel -v

ruff check pge_jax/ tests/
ruff format pge_jax/ tests/
mypy pge_jax/
```

See `AGENTS.md` for code conventions and implementation guidance. Planning documents are under `.design/`; `.sessions/` contains session records, and the reference implementation directories contain third-party documentation.

## Prior Art

- [pypge](https://github.com/verdverm/pypge): original Python implementation
- [go-pge](https://github.com/verdverm/go-pge): Go implementation
- [PySR](https://github.com/MilesCranmer/PySR): symbolic regression for Python and Julia
- [FFX](https://github.com/ffx-org/ffx): fast function extraction

## Citation

Tony Worm. "Prioritized Grammar Enumeration: Symbolic Regression by Grammar-based Deterministic Search." GECCO 2013.
<https://dl.acm.org/doi/10.1145/2463372.2463486>
