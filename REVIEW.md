# REVIEW — pge-jax

Date of review: working tree at commit `d6a5cf7` ("Update README.md").

Scope: full verification of all markdown documentation against the code (treating the
code as the source of truth), a module-by-module code review, a test-suite review,
tooling results, and prioritized suggestions.

---

## 1. Executive Summary

pge-jax is a working, tested JAX-native reimplementation of Prioritized Grammar
Enumeration (PGE) symbolic regression. The two-layer architecture (evaluation layer:
`JAXModel` + LM optimizer + metrics; search layer: `SearchModel` + `PGE` + selection +
expansion) is real, coherent, and matches the design described in the docs.

Current state of the test suite:

```
413 tests collected
323 passed
 90 skipped  (reason: "paired with stashed search.py code")
```

The project is mid-refactor on three planned workstreams:

| Workstream | State |
|---|---|
| Peek refactor (`peek_npts` → `peek_fraction`) | **Complete** |
| Grow phase optimization (Phases 1–7) | **Phases 1–5 done**, 6–7 pending |
| Core loop work-filter stage rework (Goal 3) | **Not started**; one attempt was reverted and its git stash is lost |
| Coefficient expansion (C/N/P kinds) | **Not started**, gated on grow phase completion |

Overall assessment: the codebase is in good shape for a research-stage project. The main
risks are (a) the lost/reverted pipeline-stage work with 90 orphaned skipped tests,
(b) a small amount of dead or misleading code (`ExpanderConfig`), and (c) performance
headroom left on the table in the optimizer and grow loop. None of these block
correctness; all tests that run, pass.

---

## 2. Methodology

Every factual claim in `AGENTS.md`, `README.md`, and `STATUS.md` was checked against the
source code. This included:

- Running the full test suite (`python -m pytest tests/ -q`)
- Running the linter (`ruff check pge_jax/ tests/`) and type checker (`mypy pge_jax/`)
- Verifying every name in README's "Public API" and "Individual Components" sections
  against `pge_jax/__init__.py` exports
- Verifying every code path, line reference, and "current state" claim in STATUS.md
  against the actual modules
- Verifying AGENTS.md's project structure, key types (`FitResult`, `EvalResult`),
  gotchas, and conventions against the code
- Checking git history and `git stash` for the provenance of the skipped tests

---

## 3. Documentation Review

### 3.1 README.md — two errors found and fixed

**Error 1: Nonexistent `expanders` parameter in the PGE example.**

The README showed:

```python
expanders=[
    ExpanderConfig(pop_count=3, max_size=32),
    ExpanderConfig(pop_count=2, max_size=64),
],
```

This was wrong on three levels:

1. The parameter is named `multi_expander_params`, not `expanders`
   (`pge_jax/search.py:132`).
2. `PGE` consumes it as a list of **dicts** (`p.get("pop_count")`,
   `p.get("grow_params")` at `search.py:312-321`), not `ExpanderConfig` objects.
   Passing an `ExpanderConfig` would crash on the `.get()` call.
3. `ExpanderConfig` (`search.py:52`) has fields `pop_count`, `usable_funcs`,
   `grow_params` — there is no `max_size` field, and `Grower` does not accept a
   `max_size` kwarg either (its policy keys are `func_level`, `init_level`,
   `grow_level`, `subs_level`, `adds_level`, `muls_level`, `add_xtop`, `shrinker`,
   `limiting_depth`, `grow_filter`).

Fixed to show the real dict-based API:

```python
multi_expander_params=[
    {"pop_count": 3, "grow_params": {"grow_level": "low"}},
    {"pop_count": 2, "grow_params": {"grow_level": "high"}},
],
```

**Error 2: Unexported metrics listed as package-level imports.**

The "Individual Components" section listed `chisqr, redchi, rmae` in a
`from pge_jax import (...)` block. These functions exist in `pge_jax/metrics.py` and are
used internally by `evaluate.py`, but are **not** re-exported in `pge_jax/__init__.py`.
The import as written would raise `ImportError`. Fixed to list only the exported
metrics (`rmse, mae, mse, r2, explained_variance, aic, bic`) with a note that the
remaining three are available via `pge_jax.metrics`.

**Verified accurate:** quick start, progressive evaluation diagram, `SearchModel` and
`JAXModel` examples, `peek_fraction` semantics (default 0.0 = off), installation extras
(`dev`, `benchmark`, `notebook` — all present in `pyproject.toml`), design decisions,
prior art links.

### 3.2 STATUS.md — four inaccuracies found and fixed

**Inaccuracy 1: Stale test counts.** Claimed "116 passing tests (92 original + 24 new
phase validation tests)" in the overview, phase 1 verification, and the verification
checklist. Reality: 413 collected, 323 passing, 90 skipped. (The 24 phase-validation
tests do exist in `tests/test_search.py::TestSearchModel` and still pass — the total
simply grew.) Updated all three locations.

**Inaccuracy 2: Grow phase "Current State" said "Nothing implemented."** This was stale.
Phases 1–5 are implemented and verified in code:

| Phase | Evidence |
|---|---|
| 1. Lazy `expr` | `search_model.py:366-440` — `_raw_expr` stored, `_expr = None`, `expr` is a property calling `sympy.expand` on first access, with a setter |
| 2. Defer JAX compilation | `search_model.py:572` `rewrite_coeff_only()` (static, pure symbol rewriting) split from `build_jax_model()` (line 625) |
| 3. Fix bare C | `expand.py:350,430` — `grow()` calls `rewrite_coeff_only()` and passes `cs` to the `SearchModel` constructor |
| 4. Raw + expanded children | `expand.py:345-358` — `_make_models()` creates Model A (lazy raw) and Model B (cached expanded) per child |
| 5. Lazy JAX in `_eval_models()` | `search.py:715` — `modl.build_jax_model()` called in try/except only for models selected for evaluation |

The section was reworded to make clear the listed code paths describe the
*pre-optimization* state.

**Inaccuracy 3: Goal 3 (pipeline stages) lacked history.** The section said only "Not
implemented", omitting important context: commits `9e58b90` ("some work on series of
process-filter stages in a loop") and `6cc1841` ("...tests from a failed refactor")
attempted this work; it was partially reverted. 90 tests remain skipped with reason
"paired with stashed search.py code" (30 in `tests/test_pipeline_stages.py`, 60 in
`tests/test_memoization.py`), and `git stash list` is **empty** — the stashed
implementation no longer exists. The skipped tests are the only surviving record of that
design. Added a "History" paragraph documenting this.

**Inaccuracy 4 (minor):** Phase 1's verification note referenced the stale 116 count.

**Verified accurate:** peek refactor section (constructor signature, `_set_data()`
computation `max(1, int(peek_fraction * eval_npts))`, `finalize()` prints), the Phase
6/7 TODOs (both confirmed absent from code), the coefficient expansion design (all
"not started" claims confirmed — no `coeff_registry.py`, no `system_model.py`, no
`N_`/`P_` handling anywhere), the architecture pipeline description, quick start, and
test commands.

### 3.3 AGENTS.md — verified accurate, no changes

Everything checked out:

- Project structure matches the actual `pge_jax/` module list exactly
- Commands (pytest, ruff, mypy) all work as written
- `FitResult` and `EvalResult` dataclass shapes match `optimize.py:10` and
  `evaluate.py:26`
- Gotchas are correct: float64/x64 behavior, `sympy.lambdify` fallback, LM convergence
  criteria (the four criteria match `optimize.py:140-168`), `C_`/`C[` prefix convention
  (`model.py:28`), `rewrite_coeff()` behavior, float32 test precision
- Selection API compatibility notes match: `SearchModel.values` property
  (`search_model.py:443`), `wvalues` attribute (line 413)
- `default_filters` list exists in `filters.py:166`
- The one soft spot: `optimize.py` is described as "LM and BFGS optimizers".
  `fit_least_squares` wraps `jax.scipy.optimize.minimize`, which is indeed BFGS under
  the hood, so this is accurate if slightly indirect. Left as-is.

---

## 4. Implementation Status — Done vs TODO

### 4.1 Implemented and tested

**Evaluation layer (complete):**
- `JAXModel` — sympy → JAX wrapper with `jax_fun`/`jac_fun` via `sympy.lambdify`,
  handles the constant-expression edge case
- `fit_levenberg_marquardt` — full JAX-native LM with adaptive damping and four
  convergence criteria
- `fit_least_squares` — BFGS-based alternative via `jax.scipy.optimize.minimize`,
  with optional bounds
- `metrics.py` — 10 metrics: rmse, mae, mse, r2, explained_variance, aic, bic, chisqr,
  redchi, rmae
- `evaluate.py` — `fit_model()`, `predict()`, `evaluate()` producing `FitResult` and
  `EvalResult`

**Search layer (complete, pre-refactor architecture):**
- `PGE` — full loop: `_preloop()`, `_loop()`, `finalize()`, sklearn-style `fit()`,
  `get_best_model()`, `get_final_paretos()`
- Multi-expander support via `multi_expander_params` (list of dicts)
- `Grower` — 5 expansion operators (var_sub, add_extend, mul_extend, add_xtop, shrink)
  with complexity-level policies
- Filters — 6 filters in `default_filters`, plus `filter_models_with_stats` for
  per-filter rejection breakdowns
- `Memoizer` — hash-based dedup
- Selection — NSGA-II, SPEA-II, log-nondominated sort, crowding distance, tournament
- `fitness_funcs.py` — configurable multi-objective fitness (raw and normalized)
- Algebra — expand/factor/simplify via `manip_model()`/`do_simp()`

**Peek refactor (complete):** `peek_fraction: float = 0.0` replaces `peek_npts`;
opt-in; `_set_data()` derives point counts; `finalize()` prints preserved.

**Grow phase optimization Phases 1–5 (complete):** lazy `expr` property, split
`rewrite_coeff_only()`/`build_jax_model()`, bare-C bug fixed (children now carry
populated `cs`), dual raw+expanded children per grow operator match, JAX compilation
deferred to `_eval_models()`.

**Instrumentation (complete, undocumented until now):** `StageTimings`,
`IterationStageTimes`, `IterationSubTimes`, `IterationProgress`,
`grow_with_stats()`/`GrowOperatorStats`, `filter_models_with_stats`, per-iteration
progress records, `print_timing()`/`print_summary()`.

### 4.2 Still TODO

**Grow phase optimization:**
- [ ] Phase 6: move `_uniquify()` after filtering. `grow()` still uniquifies
      expression lists and constructs two `SearchModel`s per surviving expression
      before filtering (`expand.py:294,325-326,341`), paying `SearchModel` creation
      cost for models that filters will reject.
- [ ] Phase 7: `max_grow_children` config and size-based pruning in `grow()`. No trace
      in the code.
- [ ] Benchmarks quantifying the grow-phase improvements.

**Goal 3 — work-filter stage pipeline (not started):**
- [ ] `pge_jax/pipeline.py` with a `Stage` abstraction
- [ ] Stage factories (grow, filter, memoize, algebra, eval, selection)
- [ ] `_loop()` and `_preloop()` refactored to stage lists
- [ ] Absorb grow Phases 6–7 as pipeline stages (`DedupStage`, `PruneStage`)
- [ ] Un-skip or delete the 90 orphaned tests (see §5.4)

**Coefficient expansion (last priority, design complete):**
- [ ] `pge_jax/coeff_registry.py` — global registry for named (`N_i`) and physical
      (`P_i`) coefficients
- [ ] `pge_jax/system_model.py` — multi-equation wrapper
- [ ] `rewrite_coefficients()` handling C/N/P; kind-aware extraction in `model.py`;
      `P_i` substitution before JAX compilation
- [ ] Triple term pools and cross-kind combinatorial expansion in `expand.py`
- [ ] Kind-aware memoization hash, filter updates, size-penalty weights
      (free=+1, named=+1/0, physical=+0)
- [ ] PGE constructor params (`named_count`, `physical_constants`)

---

## 5. Code Review by Module

### 5.1 `model.py` (211 lines) — JAXModel

Solid. `_extract_coeffs_and_vars()` is deterministic (sorted by name), the
constant-expression edge case is handled, and docstrings are thorough.

Observations:
- `JAXModel.__init__` still calls `sympy.expand(expr)` eagerly (line 85). This is fine
  for the evaluation layer but worth remembering: the laziness introduced in Phase 1
  lives in `SearchModel`, not here. A `SearchModel` that defers expansion but then
  builds a `JAXModel` re-expands. Not a bug — `sympy.expand` is idempotent-ish — but
  the expanded expression is computed twice on separate paths.
- Docstring says bare `C` symbols are "treated as optimisable coefficients", but
  `_extract_coeffs_and_vars()` only recognizes `C_`/`C[` prefixes — bare `C` lands in
  `vars_`. The class docstring contradicts the helper's docstring and the code. This is
  the same bare-C trap STATUS documents; the `JAXModel` docstring should be corrected.
- Each `JAXModel` triggers a fresh `sympy.lambdify` + XLA compile. There is no compile
  cache keyed by expression structure. With Phase 5's deferral this only bites for
  selected models, but structurally similar children still each compile.

### 5.2 `optimize.py` (229 lines) — LM + least squares

Correct and readable. The LM implementation has adaptive damping, four convergence
criteria, and a damping-blowup failure path.

Observations:
- **The LM loop is a Python `for` loop, not `jax.lax.scan` or `lax.while_loop`.** Every
  LM iteration round-trips through the Python interpreter and re-dispatches the
  Jacobian and residual computations. For the search loop's thousands of fits on small
  models this is likely the single largest performance lever available. Converting the
  loop body (`_step` plus the convergence state machine) to `jax.lax.while_loop` with
  a carry of `(c, mu, prev_cost, i, done)` would let XLA fuse the whole fit. The
  convergence messages are Python strings, so they would need to become integer exit
  codes mapped back to messages after the loop.
- `x0` defaults to zeros (line 78). Zero is a degenerate starting point for many
  symbolic regression forms (e.g. `C*sin(C*x)` is flat at `C=0`). pypge historically
  used small non-zero or randomized restarts. Worth considering a small constant or
  multi-start.
- `fit_least_squares` imports `jax.scipy.optimize.minimize` inside the function body —
  fine for cold-path use, slightly unusual.
- `nfev` counts LM iterations, not function evaluations (each iteration evaluates the
  residual at least twice plus the Jacobian). The name is a mild lie; it matters only
  because `PGE` prints point-eval counts derived from it.

### 5.3 `metrics.py` (86 lines) and `evaluate.py` (138 lines)

Clean and minimal. `evaluate()` assembles all metrics into `EvalResult`.

Observations:
- `chisqr`/`redchi` assume unit variance or take a `y_err` — check the signature
  before relying on absolute values in model comparison.
- Both files request `jnp.float64` explicitly, which triggers the x64 warning storm
  seen in the test output (55 warnings) unless `jax_enable_x64` is set. This is
  documented in AGENTS.md; consider setting the config in `pge_jax/__init__.py` or a
  `conftest.py` to silence the noise, or accept the warnings as a deliberate signal.

### 5.4 `search_model.py` (664 lines) — SearchModel + instrumentation dataclasses

The lazy-expr refactor (Phase 1) is cleanly done: `_raw_expr` + cached `_expr`
property with a setter, `orig` preserved.

Observations:
- **mypy error:** `memoized: int = 0` is declared twice in the instrumentation
  dataclass (lines 253-254, `no-redef`). Trivial fix.
- The class docstring repeats `JAXModel`'s incorrect claim that bare `C` is treated as
  a coefficient (line ~262). Same fix needed.
- `SearchModel` is a large state machine: 13 boolean lifecycle flags (`inited`,
  `memoized`, `algebrad`, `peeked`, `peek_queued`, `peek_popped`, `evaluated`,
  `queued`, `popped`, `expanded`, `finalized`, `errored`, ...). These are only weakly
  enforced — nothing prevents illegal transitions. The Goal 3 pipeline rework is the
  natural place to either enforce transitions or delete flags that no stage reads.
- `rewrite_coeff()` (line 617) operates on `self.orig` (the raw expression), so the
  rewritten symbols land in `_raw_expr` and the lazy property expands the rewritten
  form. Consistent, but subtle — worth a comment.

### 5.5 `search.py` (1384 lines) — PGE

The largest module and the least modular. The flat pipeline in `_preloop()`/`_loop()`
is exactly what Goal 3 targets.

Observations:
- **`ExpanderConfig` is dead code.** Defined at line 52, exported in `__init__.py`,
  referenced in the `PGE` docstring (line 104) — and never actually usable, because
  `_preloop()` treats `multi_expander_params` entries as dicts (`p.get(...)`).
  Either wire `ExpanderConfig` into `_preloop()` (accept both, normalize with
  `dataclasses.asdict` or `isinstance` checks) or delete the class and fix the
  docstring. As it stands it actively misleads users (it misled the README).
- **`search.py:377`: `to_eval = self._peek_pop() + self._peek_pop()`** — a double pop
  with the comment "double pop first time". This works but is surprising; a named
  method (`_peek_pop_double()` or a `count` parameter) would document intent.
- **Print-based reporting throughout** (`print_best`, `print_results`,
  `print_summary`, `print_timing`, plus config echo in `fit()`). AGENTS.md says "no
  print statements in library code — use logging or return values", and this module
  violates that convention extensively. Either grandfather it explicitly in AGENTS.md
  (the prints are clearly a deliberate console UX) or route through `logging`.
- Roughly 400 of the 1384 lines are reporting/instrumentation. Extracting a
  `reporting.py` would slim the core loop ahead of the Goal 3 refactor.
- The fitness defaults (`DEFAULT_FITNESS_PARAMS = ["-(1)jpsz", "-score", "+r2",
  "+bic"]`) are sensible and match the pypge lineage.

### 5.6 `expand.py` (691 lines) — Grower

The heart of the search. The Phase 3/4 integration (rewrite coefficients, emit raw +
expanded pairs) is implemented in a local `_make_models()` helper inside `grow()` and
duplicated in `grow_with_stats()` — the two copies will drift. Extract a shared helper.

Observations:
- **ruff F811:** `C` is redefined at line 20 (first defined line 17). Cosmetic but
  trivially fixable.
- Term pools are built eagerly in `__init__` across all levels — fine, they're
  one-time costs.
- The bare-`C` term pools (`wout_c_xs*_muls` etc. use `C = sympy.Symbol("C")`) remain
  correct *only because* `grow()` now rewrites before model creation. Any future code
  path that constructs a `SearchModel` or `JAXModel` directly from pool expressions
  without `rewrite_coeff_only()` reintroduces the bug. A comment at the pool
  definitions warning of this would be cheap insurance.
- Phase 6/7 (post-filter uniquify, branching cap) will touch this file; the current
  structure (build exprs → uniquify → make models → return) is amenable to both.

### 5.7 `filters.py`, `memoize.py`, `algebra.py`, `fitness_funcs.py`, `selection.py`

- `filters.py` (173 lines): six filters, `filter_models_with_stats` for rejection
  accounting. Note `filter_no_C` rejects models with empty `cs` — which is why the
  Phase 3 fix (populating `cs` in `grow()`) matters so much: before it, every child
  was rejected here.
- `memoize.py` (83 lines): `Memoizer.insert()` hashes `model.orig` — the **raw**
  expression. Two algebraically equivalent expressions with different raw forms get
  different hashes. That is by design (raw + expanded children are deliberately both
  kept), but the class docstring should say the dedup key is the raw form. Note that
  `PGE` also maintains its own `self.hmap` dict separate from `Memoizer.hmap` —
  two sources of truth for dedup state. Worth consolidating.
- `algebra.py` (72 lines): thin, fine.
- `fitness_funcs.py` (172 lines): the string-based fitness spec parser is flexible;
  raw vs normalized calc both present.
- `selection.py` (632 lines): faithful ports of NSGA-II/SPEA-II/log-ND sort including
  the Kung et al. divide-and-conquer helpers. No DEAP dependency, matching the design
  goal. This is the module most likely to harbor subtle porting bugs and has the
  thinnest test coverage relative to complexity — property-based tests (e.g. "no
  member of front A dominates a member of an earlier front") would be high value.

### 5.8 `__init__.py` (50 lines)

Exports are curated and mostly consistent. Gaps:
- `chisqr`, `redchi`, `rmae` not exported (README now reflects this; decide whether
  the code or the docs should move).
- `ExpanderConfig` exported but dead (see §5.5).
- Instrumentation dataclasses (`StageTimings`, `IterationStageTimes`) are exported but
  `IterationSubTimes` and `IterationProgress` are not. Either export all or none.

---

## 6. Test Suite Review

**Counts:** 413 collected, 323 passing, 90 skipped, 0 failures.

**Coverage by file:**

| File | Tests (funcs) | Focus |
|---|---|---|
| `test_search.py` | 77 | SearchModel lifecycle, phases 1–5 validation, filters, algebra, memoizer, selection, fitness, expand, integration, instrumentation |
| `test_pipeline_stages.py` | 30 | **Entirely skipped** — paired with the lost pipeline refactor |
| `test_memoization.py` | 14 | Partially skipped (60 parametrized cases skipped) — same reason |
| `test_coefficient_rewriting.py` | 22 | `rewrite_coeff_only` behavior |
| `test_model.py` | 13 | JAXModel construction, eval, jacobian |
| `test_metrics.py` | 11 | All metrics |
| `test_optimize.py` | 9 | LM + least squares convergence |
| `test_evaluate.py` | 8 | fit/predict/evaluate pipeline |

**The 90 skipped tests are the elephant in the room.** They are skipped at module or
class level with reason "paired with stashed search.py code". The stash is gone
(`git stash list` is empty). Three options:

1. **Reimplement the pipeline work** (Goal 3) against these tests as the spec.
2. **Curate them**: un-skip the ones that pass against current code (some may —
   `TestRewriteCoeffOnly` overlaps with `test_coefficient_rewriting.py`), fix or
   delete the rest.
3. **Delete them** and treat STATUS.md Goal 3 as the spec.

Leaving them skipped indefinitely is the worst option: they rot, inflate the skip
count, and obscure the real signal of the suite. Recommendation: option 2 as an
immediate hygiene pass, option 1 as the planned work.

**Other test observations:**
- The phase-validation tests (`test_phase1..5_*`, 24 tests) are genuinely good — they
  pin the refactor's semantics, not just its outcomes.
- Integration tests exercise full PGE runs on synthetic data with float32 tolerances
  (5e-3), consistent with AGENTS.md guidance.
- No benchmarks exist. The grow-phase TODO explicitly calls for them; `pytest-benchmark`
  or a simple `hack/` script would do.
- No property-based tests (hypothesis) anywhere. `selection.py` and `memoize.py` are
  the best candidates.

---

## 7. Tooling

**ruff:** 16 errors.

- `pge_jax/expand.py:20` — F811 redefinition of `C`
- `pge_jax/search.py:1325` — E501 line too long (128 > 120)
- `tests/test_memoization.py`, `tests/test_pipeline_stages.py` — E402/I001 import
  ordering (deliberate: skips are declared before imports), F401 unused imports
- 6 of 16 auto-fixable with `--fix`

**mypy:** 1 error — `search_model.py:254` duplicate `memoized` field. Several
`annotation-unchecked` notes for untyped function bodies (memoize.py, expand.py).

Neither tool is clean. Since the project has pre-commit configured, wiring both into it
(or CI) would prevent drift.

---

## 8. Issues Found (Code-Level)

Ordered by impact:

1. **Lost pipeline-stage implementation; 90 orphaned skipped tests.** The skipped tests
   reference stashed code that no longer exists. Decide their fate (§6).
2. **`ExpanderConfig` is dead and misleading.** Exported, documented in the `PGE`
   docstring, unusable in practice (§5.5). Fix or delete.
3. **LM loop is Python-interpreted.** Biggest known performance lever; convert to
   `lax.while_loop` (§5.2). Matters more once grow-phase work increases the number of
   surviving candidates per iteration.
4. **Docstrings repeat the bare-C falsehood.** `JAXModel` and `SearchModel` docstrings
   claim bare `C` is treated as a coefficient; the extractor only recognizes `C_`/`C[`
   (§5.1, §5.4). This is the exact misconception that caused the Phase 3 bug.
5. **Grow operator duplication.** `_make_models()` is copy-pasted between `grow()` and
   `grow_with_stats()` (§5.6).
6. **Two memoization sources of truth.** `Memoizer.hmap` and `PGE.hmap` (§5.7).
7. **mypy duplicate field** (`search_model.py:253-254`).
8. **ruff: F811 in expand.py, E501 in search.py**, import-order noise in the two
   skipped test files.
9. **Print-based reporting** contradicts AGENTS.md's no-print convention (§5.5).
10. **Metrics export gap** (`chisqr`, `redchi`, `rmae`) — docs now match code, but the
    code is arguably the thing that's wrong (§5.8).
11. **`nfev` misnomer** — counts iterations, not function evaluations (§5.2).
12. **No x64 config management** — 55 warnings per test run (§5.3).

---

## 9. Suggestions (Prioritized)

**Immediate hygiene (small, independent):**

1. Fix the mypy duplicate `memoized` field and the two ruff errors in library code
   (`expand.py` F811, `search.py` E501). Five minutes each.
2. Resolve `ExpanderConfig`: either support it in `_preloop()` (normalize dataclass →
   dict at entry) or remove it and correct the `PGE` docstring. Supporting it is the
   better API — typed configs beat stringly-typed dicts.
3. Fix the bare-C claims in the `JAXModel` and `SearchModel` docstrings.
4. Decide on the metric exports: add `chisqr`, `redchi`, `rmae` to `__init__.py`
   (consistent with `EvalResult` carrying them) or leave as-is now that the README is
   accurate.
5. Curate the 90 skipped tests: un-skip what passes, delete what is superseded, keep
   skipped only what genuinely specifies the Goal 3 design.

**Short term (aligns with existing TODOs):**

6. Grow Phase 6 + 7. Phase 6's "Option B" (keep `_uniquify()` in `grow()`, defer
   `SearchModel` creation past filtering) is the right call for the reasons STATUS
   gives. Phase 7's branching cap will matter more after Phase 4 doubled the child
   count.
7. Add a benchmark script under `hack/` measuring children/parent, filter rejection
   rate, JAX compiles per iteration, and wall time per iteration. Without this, Phases
   6–7 are flying blind.
8. Extract the shared `_make_models()` helper in `expand.py` before touching it for
   Phase 6.
9. Consolidate `Memoizer.hmap` and `PGE.hmap`.

**Medium term (the planned big rocks):**

10. Goal 3 pipeline stages. The design in STATUS.md is sound. Two additions worth
    considering: (a) make `Stage` carry its own timing so `_loop_phase_times` falls
    out naturally; (b) use the skipped `test_pipeline_stages.py` as the acceptance
    spec — that is what it was written for.
11. Convert the LM loop to `jax.lax.while_loop` with integer exit codes. Benchmark
    before/after on a realistic PGE run; expected to be the largest single speedup in
    the evaluation path.
12. Property-based tests for `selection.py` (Pareto-front invariants) and
    `memoize.py` (insert/lookup idempotence).

**Long term:**

13. Coefficient expansion (C/N/P kinds) — only after the above, per the stated
    priority order.
14. Consider a `logging`-based reporting layer (or a `verbose: int` flag) to replace
    bare prints, and reconcile AGENTS.md's no-print rule with reality.
15. Consider enabling `jax_enable_x64` in package init or documenting a one-line
    user-side setup more prominently; the current warning noise trains users to ignore
    warnings.

---

## 10. Bottom Line

The documentation now matches the code. The code is correct (all running tests pass),
well-structured at the module level, and mid-way through a well-specified optimization
plan. The project's real debt is concentrated in three places: the orphaned pipeline
tests and their lost implementation, the dead `ExpanderConfig` API, and the
Python-interpreted LM loop. Addressing the first two is hygiene; addressing the third
is where the performance is.
