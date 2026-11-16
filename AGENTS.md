# pge-jax - Agent Instructions

## Scope

`pge_jax/` and `tests/` are the implementation. `ffx/`, `go-pge/`, `pypge/`, and `pysr/` are reference codebases. `.design/` contains active design notes, `STATUS.md` tracks planned work, `REVIEW.md` tracks current findings, and `.sessions/` contains historical session records.

Do not rewrite third-party documentation or historical session records as if they described the current package.

## Project Structure

```text
pge-jax/
├── pge_jax/
│   ├── __init__.py       # Curated public exports
│   ├── model.py          # JAXModel expression evaluation and Jacobians
│   ├── optimize.py       # Levenberg-Marquardt and BFGS fitting
│   ├── metrics.py        # Regression metrics
│   ├── evaluate.py       # fit_model, predict, evaluate, result types
│   ├── search_model.py   # SearchModel and instrumentation dataclasses
│   ├── filters.py        # Expression filters and rejection counts
│   ├── algebra.py        # Symbolic expand, factor, simplify
│   ├── memoize.py        # Standalone expression memoizer
│   ├── fitness_funcs.py  # Multi-objective fitness construction
│   ├── selection.py      # NSGA-II, SPEA-II, log-nondominated sort
│   ├── expand.py         # Grammar-based grow operators
│   └── search.py         # PGE orchestration and console reporting
├── tests/                # Pytest suite
├── .design/              # Design and technical notes
├── pyproject.toml        # Package and tool configuration
├── README.md             # User documentation
├── STATUS.md             # Current work queue
└── REVIEW.md             # Active technical review
```

There is no `pge_jax/pipeline.py`; the pipeline abstraction remains planned.

## Commands

```bash
source .venv/bin/activate

python -m pytest tests/ -v
python -m pytest tests/test_search.py -v
python -m pytest tests/test_search.py::TestSearchModel -v

ruff check pge_jax/ tests/
ruff format pge_jax/ tests/
mypy pge_jax/
```

Run library and test linting separately when diagnosing failures. The skipped test modules contain existing ruff violations.

## Conventions

- Target Python 3.11+ and use modern type syntax such as `X | Y`.
- Keep lines within 120 characters.
- Use Google-style docstrings for public interfaces.
- Order standard-library, third-party, and `pge_jax` imports alphabetically within groups.
- Use `jax.numpy as jnp` for array operations. `import jax` is appropriate for configuration and transformations such as `jax.jacfwd`.
- Add type hints to public functions and new class attributes.
- Keep existing sections and IDs when updating documentation so session and design references remain valid.
- Preserve terminal output behavior unless a task explicitly changes reporting. `search.py` deliberately uses console reporting; new reusable services should return values or use logging.
- Do not use file and line references in long-lived prose unless necessary; they become stale quickly.

## Architecture

### Evaluation

1. `JAXModel` expands a sympy expression and identifies coefficient and variable symbols.
2. `sympy.lambdify` builds the prediction function, with a NumPy fallback when JAX conversion fails.
3. `fit_model()` initializes coefficients to zero and fits with Levenberg-Marquardt or BFGS.
4. `evaluate()` computes regression metrics and returns `EvalResult`.

`JAXModel` is the reusable evaluation wrapper. `SearchModel` is the search-state wrapper.

### Search

`PGE._preloop()` generates initial expressions, filters and memoizes them, creates algebraic variants, optionally performs peek selection, and fully evaluates candidates.

`PGE._loop()` selects parents for each expander, grows raw and expanded children, filters them, memoizes candidates, creates and processes algebraic variants, optionally performs peek selection, and fully evaluates selected models. Exact ordering is configuration-dependent; read `_loop()` before changing it.

The main grow operators are `var_sub`, `add_extend`, `mul_extend`, `add_xtop`, and `shrink`. `Grower.grow_with_stats()` is the instrumented path used by the search loop; keep it semantically aligned with `Grower.grow()`.

`PGE._memoize_models()` owns the active search index, model IDs, and `PGE.models`. The standalone `Memoizer` is a separate public utility. Both hash `SearchModel.orig` rather than canonicalizing algebraic equivalents.

### Progressive evaluation

For `0 < peek_fraction < 1`, `_set_data()` selects `max(1, int(peek_fraction * eval_npts))` random training rows. Peek fits coefficients on this subset and computes metrics over all training rows. Selected models then undergo normal fitting and evaluation. Data setup uses full arrays outside that interval, but the loops currently bypass peek only for exactly zero. Treat other nonzero values as an open defect, not supported configuration.

## Key Types

```python
@dataclass
class FitResult:
    success: bool
    coefficients: jnp.ndarray
    predictions: jnp.ndarray
    cost: float
    n_residuals: int
    n_params: int
    message: str = ""
    nfev: int = 0

@dataclass
class EvalResult:
    fit: FitResult
    score: float
    r2: float
    evar: float
    aic: float
    bic: float
    chisqr: float
    redchi: float
    mae: float
    rmae: float
    predictions: jnp.ndarray
```

`nfev` currently counts optimizer iterations rather than every residual or Jacobian evaluation.

## Implementation Gotchas

### JAX precision

JAX commonly defaults to float32. Requesting float64 without enabling x64 produces warnings and truncates values.

```python
import jax

jax.config.update("jax_enable_x64", True)
```

Alternatively, set `JAX_ENABLE_X64=1`. Tests use approximately `5e-3` absolute or relative tolerances for float32 compatibility. Do not tighten these tolerances without enabling x64 and reviewing platform behavior.

### Expression conversion

- Use `import jax.numpy as jnp` for arrays.
- `sympy.lambdify(..., modules="jax")` covers common expressions but can fall back to NumPy.
- Coefficients are separate arguments in the lambdified function; `JAXModel.jax_fun` unpacks its coefficient array.
- Constant expressions need special handling because they have no coefficient or variable arguments.
- `SearchModel.expr` is lazy, but `JAXModel` expands its input during construction.
- Instrumentation that calls `size()` or `pretty_expr()` can force expansion and size calculations.

### Coefficient symbols

Inference recognizes symbols beginning with `C_` or `C[`. Bare `C` is an input variable unless rewritten first.

`SearchModel.rewrite_coeff_only(expr)` assigns `C_0`, `C_1`, and so on during recursive traversal. `rewrite_coeff()` also updates model fields and immediately constructs the JAX wrapper. Grown children use the pure rewrite and defer JAX construction; initial expressions and algebraic variants use the combined method.

### Optimizer behavior

Levenberg-Marquardt checks initial cost, relative cost change, coefficient change, and damping blowup. Its Python loop calls JAX operations each iteration. Zero initialization can be unsuitable for some nonlinear forms. Fit failures remove candidates from `_eval_models()` results and set `errored=True`.

### Selection compatibility

- `selNSGA2` and `assignCrowdingDist` read `individual.values`.
- `sortLogNondominated` reads `individual.wvalues`.
- `SearchModel.values` returns `fitness_values`; `wvalues` is a stored tuple.
- Empty selections return empty lists, and callers must handle exhausted populations.

## Adding Code

### Metric

1. Implement it in `pge_jax/metrics.py`.
2. Export it in `pge_jax/__init__.py` if it is public at package level.
3. Add it to `EvalResult` and `evaluate()` if it belongs in complete evaluation.
4. Add tests in `tests/test_metrics.py` and `tests/test_evaluate.py` as appropriate.

### Optimizer

1. Implement it in `pge_jax/optimize.py` and return `FitResult`.
2. Export it in `pge_jax/__init__.py`.
3. Add a supported method in `fit_model()`.
4. Add convergence and failure-path tests in `tests/test_optimize.py` and `tests/test_evaluate.py`.

### Model feature

1. Update the appropriate model wrapper, not both by default.
2. Update JAX construction if evaluation is affected.
3. Test symbolic state, numerical results, and failure behavior separately.

### Search feature

1. Keep `grow()` and `grow_with_stats()` aligned.
2. Update filtering, memoization, instrumentation, and reporting only where required.
3. Export first-class public types in `pge_jax/__init__.py`.
4. Add focused unit tests and full-search integration coverage.
5. Update `STATUS.md` or `.design/` when an active plan changes.

### Filter

1. Add a callable `(modl: SearchModel, expr) -> bool`, where `True` means reject, in `pge_jax/filters.py`.
2. Add it to `default_filters` or `PGE._get_default_filters()` when it should be active by default.
3. Test accepted and rejected expressions plus filter statistics.
