# pge-jax Technical Review

## Scope

This review covers the current implementation in `pge_jax/`, its tests, and package documentation. Completed work and historical change logs are intentionally omitted. `STATUS.md` owns the implementation queue; `.design/` contains detailed designs.

## Verification baseline

| Check | Result |
|-------|--------|
| `python -m pytest tests/ -q` | 323 passed, 90 skipped, 55 x64 warnings |
| `ruff check pge_jax/` | Clean |
| `ruff check tests/` | 14 errors, concentrated around skipped tests and import ordering |
| `mypy pge_jax/` | Clean |

The working suite establishes a regression baseline, not coverage of the planned pipeline or coefficient extensions.

## Highest-priority findings

### 1. Progressive-evaluation guards are inconsistent

`_set_data()` uses the full training arrays when `peek_fraction <= 0` or `peek_fraction >= 1`. However, `_preloop()` and `_loop()` bypass peek selection only when `peek_fraction == 0`. Negative values and values at or above 1 therefore fit and select with the full arrays rather than following the documented disable behavior.

Define one validity predicate and use it in data setup and both loops. Add tests for negative, zero, fractional, one, and greater-than-one values.

### 2. The 90 skipped tests are not an active specification

`tests/test_pipeline_stages.py` and `tests/test_memoization.py` are skipped at module level. They contain stale identity assumptions, duplicate coverage, lint violations, and at least one malformed expression. They cannot remain the implicit acceptance suite for the pipeline work without curation.

For each test, choose one action:

- Enable it against current behavior.
- Rewrite it for an approved pipeline contract.
- Remove it because active tests cover the same behavior.

### 3. Multivariate `JAXModel.jac_fun` does not pass all inputs

Prediction accepts one array per input variable. The Jacobian wrapper differentiates and vector-maps only `inputs[0]`, so expressions with multiple variables are not handled correctly. Search fitting uses `jax.jacfwd` around a different prediction closure and does not exercise this method.

Make the direct Jacobian API support every declared input variable and add multivariate shape and numerical-equivalence tests.

### 4. Configured search options do not all affect behavior

`PGE.min_size` and `PGE.err_method` are stored and printed but are not used by filtering or metric selection. The `score` produced by `evaluate()` is always RMSE. `Grower.grow_level` is also stored but does not select grow pools; the effective controls are `subs_level`, `adds_level`, and `muls_level`. Implement the advertised options with tests or remove them from public configuration and examples.

### 5. Instrumented growth performs extra symbolic work

`Grower.grow_with_stats()` computes average child sizes with `size()`, forcing expression expansion and Jacobian-size calculation for all generated children. This can erase savings expected from lazy expressions and can distort the timings being measured.

Use inexpensive counts by default, make size summaries optional, or move them after filtering. Benchmark instrumented and uninstrumented runs against equivalent candidate streams.

## Maintainability and performance

### Grow implementation duplication

`grow()` and `grow_with_stats()` contain separate copies of child rewriting, raw/expanded construction, and model assembly. Extract a shared implementation before adding pruning or moving deduplication.

### Two memoization APIs

`PGE` maintains `models` and `hmap` directly while the package also exports `Memoizer` with separate model storage. Their shared hash convention should be explicit. Consolidation is useful only if it preserves model-ID and parent-lookup behavior.

### Search module size and output coupling

`search.py` combines loop control, data handling, evaluation, selection, statistics, and console formatting. Pipeline work should establish stage boundaries first; reporting can then be separated without combining two large behavioral changes.

### Levenberg-Marquardt dispatch

The LM loop is Python-driven around JAX operations. `jax.lax.while_loop` or another compiled control-flow strategy is a performance candidate, but conversion should be benchmark-driven and preserve termination semantics. Zero initialization may also cause slow or failed fits for some nonlinear expressions; restarts require an explicit design rather than a silent default change.

### Expression expansion in evaluation

`SearchModel` defers expansion, while `JAXModel.__init__` expands again during evaluation. This may be acceptable for correctness, but evaluation benchmarks should determine whether a pre-expanded trusted path is worthwhile.

### Filter-statistics labels

PGE wraps its size and power filters in lambdas. `filter_models_with_stats()` labels these entries `filter_0` and `filter_2` rather than their underlying filter names. Use stable configured names so rejection reports remain interpretable.

### Fit-result accounting

`FitResult.nfev` counts optimizer iterations, not every model or Jacobian evaluation. Reporting calls these counts function evaluations and derives point-evaluation totals from them. Rename the metric to iterations or implement complete evaluation counting, then adjust summaries and tests.

### Per-model evaluation totals

`SearchModel.total_fev` is initialized but never updated. Set it from peek and full fitting counts, or remove it in favor of a derived property.

### BFGS bounds are penalties

`fit_least_squares(..., bounds=...)` applies a quadratic penalty before unconstrained BFGS. It does not enforce hard parameter bounds. Document this behavior or replace it with a constrained method.

## Test and tooling work

### Immediate

- [ ] Fix progressive-evaluation boundary behavior and add boundary tests
- [ ] Fix direct multivariate Jacobian behavior
- [ ] Curate skipped tests and repair their lint errors
- [ ] Resolve `min_size` and `err_method` semantics
- [ ] Add a multivariate `JAXModel.jacobian()` regression test

### Next

- [ ] Add repeatable grow and evaluation benchmarks before optimization changes
- [ ] Assert grow candidate equivalence between instrumented and uninstrumented paths
- [ ] Add Pareto-front invariant tests for selection
- [ ] Test memoization hash collisions or switch to collision-checked keys
- [ ] Reconcile evaluation-count terminology and reporting
- [ ] Decide whether `chisqr`, `redchi`, and `rmae` should be package-level exports

### Test environment

Explicit float64 requests generate 55 warnings under the default test environment. Choose and document one policy: enable x64 for the suite, or retain float32 coverage and treat the warnings as intentional. Precision-sensitive numerical tests should declare their required mode.

## Active design work

1. Complete grow benchmarks, deduplication changes, and branching controls.
2. Introduce the work-filter stage abstraction without changing search semantics.
3. Implement opt-in named and physical coefficients after grow behavior is stable.

Details are in `STATUS.md` and `.design/`.
