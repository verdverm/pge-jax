# pge-jax — Agent Instructions

## Project Structure

```
pge-jax/
├── pge_jax/              # Package source
│   ├── __init__.py       # Public API exports
│   ├── model.py          # JAXModel: sympy → JAX evaluation wrapper
│   ├── optimize.py       # LM and BFGS optimizers
│   ├── metrics.py        # JAX-native regression metrics
│   ├── evaluate.py       # High-level fit/predict/evaluate pipeline
│   ├── search_model.py   # SearchModel: state, size, fitness
│   ├── filters.py        # Expression validity filters
│   ├── algebra.py        # Symbolic expand/factor/simplify
│   ├── memoize.py        # Hash-based expression deduplication
│   ├── fitness_funcs.py  # Multi-objective fitness construction
│   ├── selection.py      # NSGA-II, SPEA-II, log-ND sort
│   ├── expand.py         # Grower: grammar-based expression enumeration
│   └── search.py         # PGE: main search loop orchestration
├── tests/                # pytest test suite
├── pyproject.toml        # Build config, deps, tool settings
├── README.md             # User-facing documentation
└── AGENTS.md             # This file
```

## Running the Code

```bash
# Enter the venv (created during setup)
source .venv/bin/activate

# Run all tests
python -m pytest tests/ -v

# Run a single test file
python -m pytest tests/test_search.py -v

# Run a single test
python -m pytest tests/test_search.py::TestIntegration::test_pge_single_var -v

# Lint
ruff check pge_jax/ tests/

# Format
ruff format pge_jax/ tests/

# Type check
mypy pge_jax/
```

## Coding Conventions

- **Python 3.11+** — use modern syntax: `X | Y` unions, `match`/`case`, `typing.Self`
- **120-char line limit** — configured in `pyproject.toml`
- **Docstrings** — Google style (parameters, returns, raises sections)
- **Imports** — standard library first, then third-party, then local (`pge_jax.*`), alphabetized within groups
- **No bare `import jax`** — always `import jax.numpy as jnp` for array ops
- **Type hints** — preferred on all public functions and class attributes
- **No print statements** in library code — use logging or return values

## Core Architecture

### Two Abstraction Layers

**Layer 1: Evaluation** (`JAXModel` + `optimize.py` + `metrics.py`)
- Wraps a sympy expression, provides JAX-compatible predict/jacobian
- Fits coefficients via Levenberg-Marquardt (JAX-native)
- Computes regression metrics (RMSE, R², AIC, BIC, etc.)

**Layer 2: Search** (`SearchModel` + `PGE` + selection + expansion)
- `SearchModel`: wraps expression with lifecycle state flags, size metrics, fitness values
- `PGE`: orchestrates the full search loop (generate → filter → memoize → algebra → evaluate → select → expand)
- `Grower`: grammar-based expression enumeration with 5 operators (var_sub, add_extend, mul_extend, add_xtop, shrink)

### Evaluation Pipeline

1. `JAXModel.__init__()` wraps sympy Expr, compiles `jax_fun` + `jac_fun`
2. `fit_levenberg_marquardt()` optimizes coefficients → `FitResult`
3. `evaluate()` computes all regression metrics → `EvalResult`

### Search Pipeline

1. `Grower.first_exprs()` — seed expressions from grammar
2. `filter_models()` — reject invalid expressions
3. `Memoizer` — skip already-seen expressions
4. `manip_model()` — symbolic expand/factor/simplify
5. `filter_models()` + `Memoizer` — dedup algebraic variants
6. `PGE._eval_models()` — peek evaluation on `peek_npts` subset
7. `PGE._peek_pop()` — NSGA-II selection to keep promising candidates
8. `PGE._eval_models()` — full evaluation on all training data
9. `PGE._final_push()` — accumulate into final Pareto front

### Key Types

```python
@dataclass
class FitResult:
    success: bool
    coefficients: jnp.ndarray   # optimized coefficients
    predictions: jnp.ndarray    # model predictions at optimized coeffs
    cost: float                 # sum of squared residuals
    n_residuals: int
    n_params: int
    message: str = ""
    nfev: int = 0

@dataclass
class EvalResult:
    fit: FitResult
    score: float        # RMSE
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

## Gotchas

### JAX float64

JAX defaults to float32 on most platforms. To use float64 you need:

```python
jax.config.update("jax_enable_x64", True)
```

Or set the env var: `JAX_ENABLE_X64=1`. The code requests float64 in `jnp.asarray(..., dtype=jnp.float64)` which triggers warnings when x64 is disabled. Tests pass with float32, but precision-sensitive work (e.g., symbolic regression convergence) may need x64.

### sympy.lambdify + JAX

- `sympy.lambdify` with `modules="jax"` works for most expressions but falls back to numpy on failure.
- Each coefficient symbol becomes a separate argument. The wrapper in `JAXModel.jax_fun` unpacks the coefficient array to match.
- If the expression has no variables (constant), `jax_fun` receives no input arrays — handle this case.

### LM Optimizer Convergence

The LM implementation has multiple convergence criteria:
1. Cost below tolerance (`cost < tol`)
2. Relative cost change below tolerance
3. Coefficient change below tolerance
4. Damping blowup (`mu > 1e12`) — failure

The default `tol=1e-4` works for most cases. The float32 precision means you won't get better than ~1e-6 accuracy regardless.

### Coefficient Symbol Convention

Coefficients are identified by name prefix: `C_` or `C[`. The `SearchModel.rewrite_coeff()` method converts bare `sympy.Symbol('C')` leaves into `C_0`, `C_1`, etc. If you create expressions with bare `C` symbols, they won't be extracted as coefficients unless `rewrite_coeff()` is called.

### Test Precision

Tests use `atol=5e-3` or `rel=5e-3` tolerances because float32 limits precision. Don't tighten these without first ensuring x64 is enabled.

### Selection API Compatibility

The `selection.py` functions expect specific attributes on model objects:
- `selNSGA2` / `assignCrowdingDist` — expects `individual.values` (tuple of raw fitness)
- `sortLogNondominated` — expects `individual.wvalues` (tuple of weighted fitness)
- `SearchModel` provides both: `values` property returns `fitness_values`, `wvalues` is a direct attribute

### Empty Population Edge Cases

`selNSGA2` and `sortLogNondominated` return early (empty list) when given an empty individuals list. The PGE search loop may pass empty lists when the population is exhausted — callers should handle this.

## Adding New Code

### New Metric

1. Add function to `pge_jax/metrics.py` with Google-style docstring
2. Add import and export in `pge_jax/__init__.py`
3. Add to `EvalResult` in `pge_jax/evaluate.py` if it should be part of the full evaluation
4. Add tests in `tests/test_metrics.py`

### New Optimizer

1. Add function to `pge_jax/optimize.py` — return `FitResult`
2. Add import in `pge_jax/__init__.py`
3. Wire into `fit_model()` in `pge_jax/evaluate.py` if it should be a first-class option
4. Add tests in `tests/test_optimize.py`

### New Model Feature

1. Modify `JAXModel` in `pge_jax/model.py`
2. Update `_build_jax_functions()` if the change affects evaluation
3. Add tests in `tests/test_model.py`

### New Search Module

1. Add to `pge_jax/` with Google-style docstring
2. Add exports in `pge_jax/__init__.py`
3. Add tests in `tests/test_search.py`
4. Wire into `PGE` class in `pge_jax/search.py` if it should be a first-class option

### New Filter

1. Add function to `pge_jax/filters.py` — takes `(modl: SearchModel, expr)` returns `True` if rejected
2. Add to `default_filters` list
3. Add tests in `tests/test_search.py::TestFilters`
