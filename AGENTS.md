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

## Current Task: Port the Full PGE Search Loop

The evaluation back-end (`JAXModel`, `fit_model`, `evaluate`, metrics) is complete and tested. The next phase is porting the PGE search loop from `pypge/pypge/` to `pge_jax/`.

### Porting Strategy

- **No DEAP dependency** — fitness values stored as tuples on model objects
- **No multiprocessing** — JAX/XLA handles parallelism; skip `parallel.py` initially
- **No remote evaluation** — all computation is local via JAX
- **No `lmfit` / `sklearn`** — already replaced by JAX-native equivalents
- **Model class** — needs a search-loop-aware model with state flags and size metrics, separate from `JAXModel` (which is the pure evaluation wrapper)

### Porting Order

1. **`filters.py`** — Expression validity filters (no deps)
2. **`algebra.py`** — Symbolic expand/factor/simplify (only sympy dep)
3. **`memoize.py`** — Hash-based expression deduplication
4. **`fitness_funcs.py`** — Multi-objective fitness construction (no DEAP)
5. **`selection.py`** — NSGA-II / SPEA-II / log-ND sort (standalone)
6. **`expand.py`** — `Grower` class (depends on filters for `_uniquify`)
7. **Search-loop `Model` class** — State tracking, size metrics, parent/child tracking
8. **`search.py`** — `PGE` class orchestrating the full loop

### Key Adaptations from pypge

- `pypge/Model.params` (lmfit.Parameters) → `jnp.ndarray` coefficients
- `pypge/Model.score` (from sklearn) → `EvalResult.score` (from JAX metrics)
- `pypge/fit()` (lmfit LM) → `fit_model()` (JAX-native LM via `fit_levenberg_marquardt`)
- Fitness objects: replace `FitnessCalculator` DEAP class with simple tuple storage on model objects
- `sortLogNondominated` expects `fitness.wvalues` — adapt to use a `wvalues` attribute or property on model objects
- `assignCrowdingDist` expects `fitness.crowding_dist` — store as direct attribute on model objects

## Reference: pypge (Local)

The original implementation is at `pypge/pypge/`. Key modules and their port targets:

| pypge module | pge_jax target | Status |
|---|---|---|
| `pypge/model.py` (Model class) | Already replaced by `JAXModel` | Done |
| `pypge/evaluate.py` (Fit/Eval/Score) | `pge_jax/evaluate.py` + `optimize.py` | Done |
| `pypge/fitness_funcs.py` (multi-obj fitness) | `pge_jax/fitness_funcs.py` | **Not ported** |
| `pypge/filters.py` (expression filters) | `pge_jax/filters.py` | **Not ported** |
| `pypge/algebra.py` (expand/factor/simplify) | `pge_jax/algebra.py` | **Not ported** |
| `pypge/memoize.py` (deduplication) | `pge_jax/memoize.py` | **Not ported** |
| `pypge/expand.py` (Grower class) | `pge_jax/expand.py` | **Not ported** |
| `pypge/selection.py` (NSGA-II, SPEA-II) | `pge_jax/selection.py` | **Not ported** |
| `pypge/search.py` (PGE class, main loop) | `pge_jax/search.py` | **Not ported** |
| `pypge/parallel.py` (multiprocessing) | `pge_jax/parallel.py` | **Not ported** |
| `pypge/base.py` / `creator.py` (DEAP fitness) | `pge_jax/base.py` or inline | **Not ported** |

---

## PGE Search Loop — Architecture Reference

### The Full Search Pipeline

```
                    ┌─────────────────────────────────────────────┐
                    │              PGE.fit(X, Y)                   │
                    │  ┌───────────────────────────────────────┐  │
                    │  │           PRELOOP                      │  │
                    │  │ 1. Generate first_exprs()              │  │
                    │  │ 2. Filter (remove invalid)             │  │
                    │  │ 3. Memoize (deduplicate)               │  │
                    │  │ 4. Algebra (expand/factor)             │  │
                    │  │ 5. Filter + Memoize again              │  │
                    │  │ 6. Peek-evaluate OR full-evaluate      │  │
                    │  │ 7. Push to nsga2_list (population)     │  │
                    │  │ 8. Push to final (fully evaluated)     │  │
                    │  └───────────────────────────────────────┘  │
                    │                                              │
                    │  ┌───────────────────────────────────────┐  │
                    │  │           MAIN LOOP (N iterations)     │  │
                    │  │                                       │  │
                    │  │  For each expander:                   │  │
                    │  │    1. heap_pop() — NSGA-II select     │  │
                    │  │    2. grower.grow() — expand models   │  │
                    │  │                                       │  │
                    │  │ 3. Filter expanded models             │  │
                    │  │ 4. Memoize (deduplicate)              │  │
                    │  │ 5. Algebra (expand/factor)            │  │
                    │  │ 6. Filter + Memoize again             │  │
                    │  │                                       │  │
                    │  │ 7. Peek-evaluate unique models        │  │
                    │  │ 8. Push to nsga2_peek heap            │  │
                    │  │ 9. heap_pop(nsga2_peek)               │  │
                    │  │ 10. Full-evaluate selected models     │  │
                    │  │ 11. Push to final                     │  │
                    │  │ 12. Push to nsga2_list                │  │
                    │  └───────────────────────────────────────┘  │
                    │                                              │
                    │  ┌───────────────────────────────────────┐  │
                    │  │          FINALIZE                      │  │
                    │  │ 1. Combine final + nsga2_list         │  │
                    │  │ 2. sortLogNondominated()              │  │
                    │  │ 3. Print Pareto fronts                │  │
                    │  │ 4. Stop workers                       │  │
                    │  └───────────────────────────────────────┘  │
                    └─────────────────────────────────────────────┘
```

### Component Breakdown

#### 1. Expression Model (`pypge/model.py`)

The `Model` class wraps a sympy expression and manages its lifecycle state. Key attributes:

- **Identification**: `id`, `iter_id`, `parent_id`, `gen_relation` (e.g., "first_gen", "var_xpnd", "add_xpnd", "mul_xpnd", "shrunk")
- **Expression**: `orig` (original before coeff rewriting), `expr` (expanded after `rewrite_coeff()`), `pretty` (cached string)
- **Variables/Coefficients**: `xs` (input variables), `cs` (coefficient symbols `C_0, C_1, ...`), `params` (fitted values)
- **Size metrics**: `sz` (tree size), `psz` (penalized size, +2 per function node), `jsz`/`jpsz` (Jacobian sizes), `ncs` (num coefficients)
- **Fitness**: `peek_*` (partial fitness on subset), full `score`, `r2`, `evar`, `aic`, `bic`, `chisqr`, `redchi`, `mae`, `rmae`
- **Improvement over parent**: `improve_score`, `improve_r2`, etc.
- **State flags**: `inited`, `memoized`, `algebrad`, `peeked`, `peek_queued`, `peek_popped`, `evaluated`, `popped`, `expanded`, `finalized`, `errored`

Key method: `rewrite_coeff()` — converts bare `sympy.Symbol('C')` leaves into `C_0, C_1, ...` and creates `lmfit.Parameters`. The JAX port's `JAXModel._extract_coeffs_and_vars()` already handles this.

**Size calculation** (`calc_tree_size`): walks `sympy.preorder_traversal`, counts nodes (+2 penalty per `is_Function`), sums integer absolute values as size.

#### 2. Expression Expansion / Grower (`pypge/expand.py`)

The `Grower` class implements grammar-based expression enumeration. Pre-computed expression pools built in `__init__`:

- **Variable powers**: `xs_pow1` through `xs_pow4` — `x^n`, `x^-n` for n=1..4
- **Variable products**: `wout_c_xs1_muls` through `wout_c_xs4_muls` — products of 1-4 variables (with replacement)
- **Coefficient-scaled products**: `with_c_xs1_muls` through `with_c_xs4_muls` — `C * variable_product`
- **Function expressions**: `wout_c_linear_funcs`, `wout_c_nonlin_funcs`, `with_c_linear_funcs`, `with_c_nonlin_funcs`
  - Linear: `f(var_product)`, `f(var_product)` with inverse
  - Nonlinear: `f(C*var_product + C)`, with inverse
- **Expansion term pools**: `var_sub_terms`, `add_extend_terms`, `mul_extend_terms` — vary by complexity level (low/med/high)

**Key methods:**

- **`first_exprs()`**: Generate initial generation. Combines variable products + function expressions, adds `+C` variants, uniquifies via `evalf()` hashing, wraps in `Model` objects with `gen_relation="first_gen"`
- **`grow(M)`**: Expand a single model via 5 operators:
  1. **`_var_sub()`**: Variable substitution — replaces input variable occurrences with more complex expressions (recursive, depth-limited)
  2. **`_add_extend()`**: Addition extension — adds terms to `Add` nodes (recursive)
  3. **`_mul_extend()`**: Multiplication extension — adds factors to `Mul` nodes (recursive)
  4. **`_add_extend_top_level()`**: Top-level addition extension with more complex multiplication (non-recursive)
  5. **`_shrinker()`**: Term removal from `Add` nodes (recursive)
- **`_toggle_plus_C(expr)`**: Adds or removes a bare `C` from an addition
- **`_uniquify(exprs)`**: Removes duplicates via `p.evalf()` hashing

**Policy levels** (all configurable via kwargs):
- `func_level`: "linear" or "nonlin" — whether functions use affine transforms `f(C*x+C)` or just `f(x)`
- `init_level`: "low", "med", "high" — complexity of first-generation models
- `grow_level` / `subs_level` / `adds_level` / `muls_level`: expansion complexity levels
- `add_xtop`: Whether to extend at the top level of additions
- `shrinker`: Whether to try removing terms from additions
- `limiting_depth`: Depth limit for variable substitution (default 4)
- `grow_filter`: Whether to avoid duplicate terms in additions

#### 3. Multi-Objective Selection (`pypge/selection.py`)

Taken from DEAP. Key functions:

- **`selNSGA2(individuals, k, nd='standard')`**: NSGA-II selection. Sorts into Pareto fronts, assigns crowding distances, fills selection from best fronts, breaking ties by crowding distance.
  - `nd='standard'`: Uses `sortNondominated()` (O(M*N^2))
  - `nd='log'`: Uses `sortLogNondominated()` — **this is what PGE uses**
- **`sortLogNondominated(individuals, k)`**: Generalized Reduced Run-Time Complexity Non-Dominated Sorting (Fortin et al. 2013). Uses recursive divide-and-conquer with sweep procedures (`sweepA`, `sweepB`) for O(N log^(M-1) N) complexity.
- **`assignCrowdingDist(individuals)`**: Assigns crowding distance to each individual for diversity maintenance.
- **`selSPEA2(individuals, k)`**: Strength Pareto EA II selection.
- **`isDominated(wvalues1, wvalues2)`**: Checks if wvalues1 is dominated by wvalues2 (minimization semantics: lower is better).

Fitness objects need `values` attribute (tuple) for `selNSGA2` and `wvalues` attribute for `sortLogNondominated`.

#### 4. Fitness Calculation (`pypge/fitness_funcs.py`)

Dynamically constructs multi-objective fitness from parameter list like `["normalize", "-(1)jpsz", "-score", "+bic"]`:

- Weight parsing: `"-(1)jpsz"` → weight = -1.0 (minimize jpsz), `"+bic"` → weight = +1.0 (maximize bic)
- `"normalize"` prefix: normalizes each objective across the population by L2 norm before setting fitness values
- `build_value_extractor(params)`: creates a function that extracts specified attributes from each model
- Fitness values are set via `modl.fitness.setValues(vals)` on DEAP-style fitness objects

For JAX port, fitness values can be stored directly as tuples on model objects — no DEAP dependency needed.

#### 5. Expression Filters (`pypge/filters.py`)

`default_filters` list applied to every model:

- `filter_too_big`: Removes models with `size() > 64`
- `filter_has_int_coeff`: Removes expressions with hardcoded integer coefficients (not using `C` symbols)
- `filter_has_big_pow`: Removes expressions with power exponents > 6
- `filter_just_C`: Removes expressions that are just the coefficient symbol `C` or a number
- `filter_no_C`: Removes expressions with no coefficients (constant expressions)
- `filter_has_coeff_pow`: Removes expressions like `C^2` (coefficient raised to a power)

Filter API: each filter takes `(modl, expr)` and returns `True` if the expression should be **rejected**. Walks `sympy.preorder_traversal(expr)` and checks each node.

#### 6. Algebraic Manipulation (`pypge/algebra.py`)

Applies symbolic simplification to expressions:

- **`manip_model(modl, method)`**: Applies method to model. Returns `(new_model, None)` if expression changed, `(None, "same")` if unchanged, `(None, error)` on failure.
- **`do_simp(expr, method)`**: Supports `"simplify"`, `"expand"`, `"factor"`
- Methods: `sympy.simplify()`, `sympy.expand()`, `sympy.factor()`

#### 7. Memoization / Deduplication (`pypge/memoize.py`)

- **`Memoizer`**: Indexes models by expression hash. `insert(model)` returns `True` if new, `False` if duplicate. `lookup(model)` returns `(found, model)`.
- **`Mapper`**: Maps sympy node types to integer codes for serialization (for remote evaluation). Not needed in JAX port — use `hash(expr)` or `expr.__hash__()` directly.
- In the JAX port, deduplication can use a simple `dict[expr_hash: Expr] -> Model` mapping.

#### 8. The PGE Search Class (`pypge/search.py`)

The `PGE` class orchestrates everything. Key data structures:

- **`self.models`**: All models ever created (indexed by `id`)
- **`self.hmap`**: Dict mapping `expr -> Model` for deduplication
- **`self.final`**: All fully evaluated, finalized models
- **`self.nsga2_list`**: Models waiting to be expanded (the "population" heap)
- **`self.nsga2_peek`**: Models that have been peek-evaluated, waiting for full evaluation
- **`self.multi_expanders[i]`**: Per-expander config with `pop_count`, `nsga2_list`, and `grower`

**Main loop per iteration:**

1. **Multi-expand**: For each expander, pop `pop_count` models via NSGA-II selection, call `grower.grow()`, push remaining back
2. **Filter**: Remove invalid models
3. **Memoize**: Deduplicate via expression hash
4. **Algebra**: Apply expand/factor to unique models
5. **Filter + Memoize** algebraic results again
6. **Peek-evaluate** unique models on subset of data
7. **Push** peek'd models to `nsga2_peek` heap
8. **Pop** best models from peek heap for full evaluation (via NSGA-II selection)
9. **Full-evaluate** selected models on full training data
10. **Push** to `self.final` and `nsga2_list`

**Progressive evaluation**: Uses `peek_npts` (default 16) data points for fast partial evaluation, then selects best models from peek heap for full evaluation on all data.

### Data Flow for Each Model

```
1. Creation: Grower.first_exprs() or Grower.grow() → sympy.Expr
2. Model wrapping: Model(expr) → rewrite_coeff(), compute Jacobian
3. Filtering: filters.filter_models() → remove invalid expressions
4. Deduplication: memoize_models() → check self.hmap[expr]
5. Algebra: algebra.manip_model() → expand/factor
6. Deduplication again: after algebra
7. Peek evaluation: eval_model() on subset of data (fast)
8. Selection: selNSGA2() or sortLogNondominated() picks best peek'd models
9. Full evaluation: eval_model() on full data (slow, with lmfit)
10. Fitness assignment: fitness_funcs.build_fitness_calc() sets model.fitness.values
11. Population update: push to nsga2_list for future expansion or final for output
```

### JAX Port Design Decisions

Key differences from pypge for the JAX port:

1. **No DEAP dependency** — fitness values stored directly as tuples on model objects
2. **No multiprocessing by default** — JAX handles parallelism via XLA; can add later
3. **No remote evaluation** — all computation is local via JAX
4. **No lmfit** — replaced by custom JAX-native LM optimizer
5. **No sklearn** — replaced by JAX-native metrics
6. **No `lmfit.Parameters`** — coefficients stored as `jnp.ndarray`
7. **Model class** — can be simplified; state flags are important for the search loop but the DEAP-style fitness wrapper is not needed

### Dependencies

Current `pyproject.toml` dependencies:
- `jax`, `jaxlib` — array computation + autodiff
- `numpy` — numeric operations
- `scipy` — for `jax.scipy.optimize.minimize`
- `sympy` — symbolic expression representation
- `networkx` — graph storage (for relationship tracking)
- `tqdm` — progress bars

Optional deps (not needed for core PGE loop):
- `scikit-learn`, `lmfit`, `pandas` — benchmark/compatibility only

### Porting Order (Recommended)

1. **`filters.py`** — Simple, no dependencies on other new modules
2. **`algebra.py`** — Simple, only depends on sympy
3. **`memoize.py`** — Simple hash-based dedup
4. **`fitness_funcs.py`** — Multi-objective fitness construction (no DEAP needed)
5. **`selection.py`** — NSGA-II / SPEA-II (standalone, no other new deps)
6. **`expand.py`** — Grower class (depends on filters for `_uniquify`, creates Model objects)
7. **`model.py`** — Simplified Model class for the search loop (state tracking, size metrics)
8. **`search.py`** — PGE class (depends on all above)
9. **`parallel.py`** — Multiprocessing (optional, can skip initially)
