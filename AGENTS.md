# pge-jax — Agent Instructions

## Project Structure

```
pge-jax/
├── pge_jax/          # Package source
│   ├── __init__.py   # Public API — add new exports here
│   ├── model.py      # JAXModel: sympy → JAX evaluation wrapper
│   ├── optimize.py   # LM and BFGS optimizers
│   ├── metrics.py    # JAX-native regression metrics
│   └── evaluate.py   # High-level fit/predict/evaluate pipeline
├── tests/            # pytest test suite
├── pyproject.toml    # Build config, deps, tool settings
├── README.md         # User-facing documentation
└── AGENTS.md         # This file
```

## Running the Code

```bash
# Enter the venv (created during setup)
source .venv/bin/activate

# Run all tests
python -m pytest tests/ -v

# Run a single test file
python -m pytest tests/test_model.py -v

# Run a single test
python -m pytest tests/test_model.py::TestJAXModel::test_linear_model_basic -v

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

### The Evaluation Pipeline

The key abstraction is `JAXModel` (model.py). It wraps a sympy expression and provides JAX-compatible evaluation:

1. **Symbol extraction** — `_extract_coeffs_and_vars()` splits `expr.free_symbols` into coefficients (`C_*`) and variables. Sorted by string name for determinism.

2. **JAX function construction** — `sympy.lambdify(all_syms, expr, modules="jax")` creates a callable. The wrapper unpacks the coefficient array into individual args matching the lambdify signature.

3. **Jacobian** — computed via `jax.jacfwd` + `jax.vmap`. Each sample's Jacobian is computed independently and batched.

4. **Optimization** — `fit_levenberg_marquardt()` in optimize.py solves the damped normal equations. The Jacobian from step 3 feeds directly into the LM step.

### Data Flow

```
sympy.Expr → JAXModel.__init__() → jax_fun, jac_fun
                                → fit_levenberg_marquardt()
                                → FitResult(coefficients, predictions, cost, ...)
```

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

Or set the env var: `JAX_ENABLE_X64=1`. The current code requests float64 in `jnp.asarray(..., dtype=jnp.float64)` which triggers warnings when x64 is disabled. Tests pass with float32, but precision-sensitive work (e.g., symbolic regression convergence) may need x64.

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

Coefficients are identified by name prefix: `C_` or `C[`. The original pypge used bare `C` with `rewrite_coeff()` to generate `C_0`, `C_1`, etc. The JAX port assumes this convention is already applied. If you create expressions with bare `C` symbols, they won't be extracted as coefficients.

### Test Precision

Tests use `atol=5e-3` or `rel=5e-3` tolerances because float32 limits precision. Don't tighten these without first ensuring x64 is enabled.

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

## PGE Integration (Future)

The `pge_jax` module is the evaluation back-end. The PGE search loop (selection, expansion, prioritization) will be built on top of it. When implementing:

- Use `JAXModel` for all expression evaluation — never call `sympy.lambdify` directly
- Use `fit_model()` for the full fit+evaluate pipeline
- Use metrics from `pge_jax.metrics` for fitness computation
- The `EvalResult` dataclass is the fitness evaluation output — it will feed into NSGA-II selection

## Reference: pypge

The original implementation lives at https://github.com/verdverm/pypge. Key modules to reference:

- `pypge/model.py` → `pge_jax/model.py` (JAXModel replaces Model)
- `pypge/evaluate.py` → `pge_jax/evaluate.py` + `optimize.py` (lmfit → JAX LM)
- `pypge/fitness_funcs.py` → `pge_jax/metrics.py` (sklearn metrics → JAX-native)

The PGE search logic (expand.py, search.py, selection.py) is not ported yet — it will be rewritten to use the JAX evaluation pipeline.
