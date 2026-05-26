# STATUS — PGE JAX

## Overview

Prioritized Grammar Enumeration (PGE) for symbolic regression, implemented in JAX. The core search loop is complete with 116 passing tests (92 original + 24 new phase validation tests).

**Priority order:** (1) Grow phase optimization → (2) Coefficient expansion (LAST).

| Upgrade | Priority | Scope | Impact | Status |
|---|---|---|---|---|
| **Grow phase optimization** | 1 | Expression enumeration performance | Bare C bug fix, lazy JAX | Partial |
| **Coefficient expansion** | 3 (last) | Coefficient system + expression generation | New coefficient kinds (named, physical), expression-level dedup | Not started |

Peek refactor is complete and not listed above.

---

## Peek Refactor

### Brief

Replace `peek_npts: int = 16` with `peek_fraction: float = 0.0` in the PGE constructor. The old default (16) enabled peek by default. The new default (0.0) disables peek by default — users must opt in with a positive fraction. The fraction scales with dataset size (e.g. `0.2` = 20% of training data).

### Status

**Complete.** All code changes, tests, and documentation updates are done.

- PGE constructor accepts `peek_fraction` instead of `peek_npts`
- `_set_data()` computes `self.peek_npts = max(1, int(peek_fraction * eval_npts))` and stores it for `finalize()` prints
- `_preloop()` and `_loop()` check `peek_fraction` instead of `peek_npts`
- `finalize()` prints preserved via stored `self.peek_npts`
- Tests updated: `test_pge_multi_var` → `peek_fraction=0.125`
- New tests: `test_peek_default_off`, `test_peek_fraction_opt_in`

### Important Context

- `peek_fraction >= 1` or `<= 0` → skip peek (same as old `peek_npts == 0`)
- When peek is off, `X_peek`/`Y_peek` are set to full training data (existing behavior)
- The `_eval_models()` logic is unchanged — it uses `self.X_peek`/`self.Y_peek` without knowing how they were computed
- `peek_npts` is still stored as an instance attribute so `finalize()` prints report accurate point-eval counts

---

## 1. Grow Phase Optimization

### Brief

The grow phase dominates search time. Three architectural issues:

1. `sympy.expand()` is called eagerly in `SearchModel.__init__` for every child, but `SearchModel` is a generic wrapper that shouldn't care about expression normalization
2. Raw and expanded forms are structurally different trees that produce different children when `grow()` is applied — currently only the expanded form exists (hidden in `__init__`)
3. Children always fail `filter_no_C` because `cs` is never populated, wasting all downstream filter work

### Current State

Nothing implemented. The following code paths are confirmed slow/broken:

- `search_model.py:133` — `self.expr = sympy.expand(expr)` runs eagerly in every `SearchModel.__init__`
- `expand.py:337-341` — `grow()` creates `SearchModel(e, p_id=M.id, reln="...")` with no `cs` arg, so `self.cs = []`
- `expand.py:149-152` — term pools use bare `C = sympy.Symbol("C")`
- `model.py:26` — `_extract_coeffs_and_vars()` only recognizes `C_` and `C[` prefixes, so bare `C` ends up in `vars_` not `cs_`
- `search_model.py:338` — `rewrite_coeff()` calls `_JAXModel()` immediately, triggering JAX compilation for every child

### Phase 1: Lazy `expr` in `SearchModel`

**Files:** `search_model.py`

- Store raw expression in `__init__`: `self._raw_expr = expr`, set `self._expr = None`
- Convert `expr` to a property: compute `sympy.expand(self._raw_expr)` on first access
- `orig` stays as the unexpanded input (also set to same value as `_raw_expr` in `__init__` — redundant but needed by `filter_just_C`)
- `size()`, `calc_tree_size()`, `calc_jac_size()` all access `self.expr` — property is fine, first access triggers expand, subsequent accesses are cached
- `pretty_expr()` also accesses `self.expr` — no change needed
- `SearchModel.__hash__` uses `self.id`, not expression — no change needed
- Added `expr` setter for direct cache assignment (used by Phase 4)

**Verification:** ✅ All 116 tests pass. First access defers to lazy computation. No behavioral change.

**Tests added:** `test_phase1_lazy_expr_not_computed_on_init`, `test_phase1_lazy_expr_computed_on_access`, `test_phase1_lazy_expr_cached`, `test_phase1_lazy_expr_setter`, `test_phase1_orig_unchanged`

### Phase 2: Defer JAX compilation

**Files:** `search_model.py`, `model.py`

- Split `rewrite_coeff()` into two steps:
  1. `rewrite_coeff_only(expr)` — static method, symbol rewriting only, returns `(rewritten_expr, cs_list)`. No JAX, no side effects.
  2. `build_jax_model(expr, cs, xs)` — new method on `SearchModel` that creates `JAXModel(expr, cs=cs, xs=xs)`
- `rewrite_coeff()` (instance method) calls both steps (keeps current behavior for `first_exprs()` and `manip_model()`)
- `grow()` calls only `rewrite_coeff_only()` to get `cs` list, passes it to `SearchModel` constructor
- `SearchModel.__init__` accepts optional `cs` and stores it, sets `self.jax_model = None`
- `build_jax_model()` is called lazily in `_eval_models()` when a model is selected for evaluation

**Why this matters:** `rewrite_coeff()` currently builds `JAXModel` which calls `_build_jax_functions()` → `sympy.lambdify()` → XLA compilation. This happens eagerly for every child in `first_exprs()` and `grow()`. Deferring compilation means only models that survive filtering and memoization get compiled.

**Tests added:** `test_phase2_rewrite_coeff_only_is_static`, `test_phase2_rewrite_coeff_only_returns_rewritten_expr`, `test_phase2_rewrite_coeff_only_returns_cs_list`, `test_phase2_rewrite_coeff_only_no_side_effects`, `test_phase2_rewrite_coeff_builds_jax`

### Phase 3: Fix bare C — populate `cs` in `grow()`

**Files:** `expand.py`, `search_model.py`

- `grow()` calls `rewrite_coeff_only()` on each generated expression to get `(rewritten_expr, cs_list)`
- Pass `cs=cs_list` to `SearchModel` constructor
- Children now have `cs = [C_0, C_1, ...]` instead of `cs = []`
- `filter_no_C` now passes for children (they have coefficients)
- `jax_model` stays `None` until Phase 5 calls `build_jax_model()`

**Bare C bug fix:** Previously, children had bare `C` in their expressions and `cs = []`. When `jax_model` was eventually built, `_extract_coeffs_and_vars()` put bare `C` into `vars_` because it only recognises `C_` and `C[` prefixes. Now `rewrite_coeff()` replaces bare `C` with `C_0, C_1, ...` before the expression leaves `grow()`.

**Tests added:** `test_phase3_searchmodel_accepts_cs`, `test_phase3_searchmodel_cs_empty_by_default`, `test_phase3_filter_no_C_passes_when_cs_populated`, `test_phase3_filter_no_C_fails_when_cs_empty`

### Phase 4: Two `SearchModel` per child (raw + expanded)

**Files:** `expand.py`

- After rewriting in `grow()`, create two `SearchModel` instances per child:
  - Model A: `SearchModel(rewritten, xs=self.xs, cs=cs, p_id=M.id, reln=reln)` — raw form, `_expr = None` (lazy)
  - Model B: `SearchModel(expanded, xs=self.xs, cs=cs, p_id=M.id, reln=reln)` with `m_expanded._expr = expanded` — expanded form, `_expr` cached
- Both have the same `cs` and `xs` (rewriting is the same for both)
- Both have `jax_model = None` (deferred to Phase 5)
- Returns ~2× children but explores different operator match surfaces

**Rationale:** The raw form `(C*x0 + C) * (C*x1 + C)` has a `Mul` at the top with two `Add` children. The expanded form `C**2*x0*x1 + C**2*x0 + C**2*x1 + C**2` has an `Add` at the top with four `Mul` children. Different tree structures → different grow operator matches → different children.

**Tests added:** `test_phase4_grow_returns_raw_and_expanded`, `test_phase4_raw_model_has_lazy_expr`, `test_phase4_expanded_model_has_cached_expr`, `test_phase4_both_models_have_same_cs`, `test_phase4_double_children_count`, `test_phase4_raw_and_expanded_different_trees`

### Phase 5: Build JAX lazily in `_eval_models()`

**Files:** `search.py`

- `_eval_models()` now returns `List[SearchModel]` (only successfully evaluated models)
- Before calling `fit_model()`, check if `modl.jax_model is None`
- If `None`, call `modl.build_jax_model()` in a try/except block
- Models that fail JAX compilation are marked `errored = True` and skipped
- This ensures JAX compilation only happens for models that survive filtering + memoization + peek selection

**Tests added:** `test_phase5_build_jax_model_creates_jax_wrapper`, `test_phase5_build_jax_model_with_no_coeffs`, `test_phase5_build_jax_model_idempotent`, `test_phase5_searchmodel_jax_model_none_by_default`

### Phase 6: Move `_uniquify()` after filtering

**Files:** `expand.py`

- Current: `grow()` calls `_uniquify()` on expression lists → creates `SearchModel` for each → returns → filtered in `search.py:360`
- New: `grow()` returns raw expression lists (no `_uniquify()`, no `SearchModel` creation)
- Filtering happens on raw expressions in `search.py` (same filter logic, works on sympy Expr)
- After filtering, call `_uniquify()` on the filtered list before memoization
- Saves `SearchModel` creation + `__hash__` + `size()` computation for rejected expressions

**Note:** Some filters in `filters.py` access `modl.size()` and `modl.cs`. After this change, filtered models won't have `SearchModel` instances yet. Two options:
  - Option A: Move `_uniquify()` + `SearchModel` creation to after filtering, but make filters work on raw expressions (remove `modl` parameter, change filters to pure `expr → bool`)
  - Option B: Keep `SearchModel` creation in `grow()` but skip `_uniquify()` and `size()` computation until after filtering. `size()` is already lazy (computed on first access when `sz == 0`), so this is cheap.
  - **Recommendation: Option B.** Minimal filter changes needed. `filter_no_C` checks `len(modl.cs)` which is now populated (Phase 3). `filter_too_big` accesses `modl.size()` which is lazy.

### Phase 7: Reduce branching factor

**Files:** `expand.py`

- Add `max_grow_children` config to `Grower.__init__` (default: unlimited, opt-in)
- Size-based pruning in `grow()`: skip operators that would produce children exceeding `max_size`
- Pool sizes are already bounded by `subs_level`, `adds_level`, `muls_level` params. Consider reducing defaults for `high` levels.

### Verification

- All 116 tests pass after each phase (92 original + 24 new phase validation tests)
- Add benchmark comparing grow time per iteration before/after (Phase 7)
- Track: children per parent, filter rejection rate, time per child, JAX compilations per iteration

### TODO

- [ ] Phase 6: Move `_uniquify()` after filtering
- [ ] Phase 7: Reduce branching factor
- [ ] Add benchmarks

---

## 2. Coefficient Expansion (LAST)

### Brief

Extend the coefficient system from a single kind (`C_i`, free optimisable) to three kinds:

| Kind | Symbol | Optimisable? | Lifetime | Use Case |
|---|---|---|---|---|
| **Free** | `C_i` | Yes | Per-expression | Standard symbolic regression |
| **Named** | `N_i` | Yes | Cross-expression | Systems of equations, shared parameters |
| **Physical** | `P_i` | No | Global constants | `g`, `c`, `k_B`, `h`, `R` |

Key design decisions from user:
- All three kinds must be grown (combinatorial expansion)
- Named coefficients support both count-based and name-based indexing
- Physical constants are user-provided with a built-in registry of well-known constants
- `SystemModel` wrapper for multi-equation representation (no system fitness yet)
- LM optimizer treats free + named identically; physical constants are substituted before optimization
- `P_i` stays readable as named constants in output but is substituted away before JAX compilation
- New kinds are opt-in via PGE constructor parameters

### Status

**Not started.** Design is complete. No code changes yet. **Must be done AFTER grow phase optimization.**

### Prerequisite

Grow phase optimization must be complete first. The bare C fix (Phase 3) and deferred JAX compilation (Phase 2) are prerequisites for safely integrating coefficient kinds into `grow()`. Without deferred compilation, adding three coefficient kinds would multiply the JAX compilation cost by 3× (or more with cross-kind combos).

### TODO

- [ ] Create `pge_jax/coeff_registry.py` — global `CoefficientRegistry` for N_i and P_i
- [ ] Create `pge_jax/system_model.py` — `SystemModel` wrapper for multi-equation representation
- [ ] Rename `rewrite_coeff()` → `rewrite_coefficients()` in `SearchModel` — handle C/N/P
- [ ] Update `model.py` — kind-aware `_extract_coeffs_and_vars()`, P_i substitution before JAX compilation
- [ ] Update `expand.py` — triple term pools + cross-kind combinations (combinatorial)
- [ ] Update `memoize.py` — kind-aware hash (same structure + same coefficient indices)
- [ ] Update `search.py` — PGE accepts `named_count`, `physical_constants`, passes to Grower
- [ ] Update `filters.py` — `filter_has_int_coeff` allows physical constant values
- [ ] Update `fitness_funcs.py` — size penalty: free=+1, named=+1 if new/0 if shared, physical=+0
- [ ] Update `__init__.py` — export new types
- [ ] Add tests for all new functionality

### Important Context

**Bare C bug** (fixed by grow phase optimization, Phase 3): `grow()` creates child models from bare-`C` term pools without calling `rewrite_coeff()`. Children have `jax_model = None`, `cs = []`, and bare `C` in their expressions. When `jax_model` is eventually built lazily, `_extract_coeffs_and_vars()` puts bare `C` into `vars_` instead of `cs_` because it only recognises `C_` and `C[` prefixes. This bug is fixed by Phase 3 of grow phase optimization.

---

## Current Architecture

1. `Grower.first_exprs()` — seed expressions from grammar
2. `filter_models()` — reject invalid expressions
3. `Memoizer` — skip already-seen expressions
4. `manip_model()` — symbolic expand/factor/simplify
5. `filter_models()` + `Memoizer` — dedup algebraic variants
6. `PGE._eval_models()` — peek evaluation on subset (when `peek_fraction > 0`)
7. `PGE._peek_pop()` — NSGA-II selection to keep promising candidates
8. `PGE._eval_models()` — full evaluation on all training data
9. `PGE._final_push()` — accumulate into final Pareto front

### Key Design Decisions

1. **No DEAP dependency** — fitness values stored as tuples on model objects (`fitness_values`, `wvalues`)
2. **No multiprocessing** — JAX/XLA handles parallelism
3. **No remote evaluation** — all computation is local via JAX
4. **No lmfit/sklearn** — replaced by JAX-native LM optimizer + JAX metrics
5. **Two model classes**: `JAXModel` (pure JAX evaluation) + `SearchModel` (search loop state)
6. **`sortLogNondominated`** expects `wvalues` as plain tuples (not objects)

---

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
    peek_fraction=0.16,  # 16% of training data
)
pge.fit(X, Y)

# Get results
paretos = pge.get_final_paretos()
best = pge.get_best_model()
print(best.pretty_expr())
```

---

## Test Commands

```bash
# All tests
python -m pytest tests/ -v

# Search loop tests only
python -m pytest tests/test_search.py -v

# Lint
ruff check pge_jax/ tests/

# Format
ruff format pge_jax/ tests/
```
