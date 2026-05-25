# STATUS — PGE JAX

## Overview

Prioritized Grammar Enumeration (PGE) for symbolic regression, implemented in JAX. The core search loop is complete with 92 passing tests. Three major upgrades are planned:

| Upgrade | Scope | Impact |
|---|---|---|
| **Peek refactor** | Search loop parameter interface | Behavioral shift: peek off by default, fraction-based |
| **Coefficient expansion** | Coefficient system + expression generation | New coefficient kinds (named, physical), expression-level dedup |
| **Grow phase optimization** | Expression enumeration performance | 2–5× speedup, two-model per child architecture |

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

### Todos

None. All done.

### Important Context

- `peek_fraction >= 1` or `<= 0` → skip peek (same as old `peek_npts == 0`)
- When peek is off, `X_peek`/`Y_peek` are set to full training data (existing behavior)
- The `_eval_models()` logic is unchanged — it uses `self.X_peek`/`self.Y_peek` without knowing how they were computed
- `peek_npts` is still stored as an instance attribute so `finalize()` prints report accurate point-eval counts

---

## Coefficient Expansion

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

**Not started.** Design is complete. No code changes yet.

### Todos

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

**Bare C bug** (pre-existing): `grow()` creates child models from bare-`C` term pools without calling `rewrite_coeff()`. Children have `jax_model = None`, `cs = []`, and bare `C` in their expressions. When `jax_model` is eventually built lazily, `_extract_coeffs_and_vars()` puts bare `C` into `vars_` instead of `cs_` because it only recognises `C_` and `C[` prefixes. This is a bug that affects the current single-kind system and will need to be fixed as part of this work.

**Interaction with grow phase optimization** (see below): The coefficient expansion work intersects with the grow phase performance analysis. Specifically, fixing the bare `C` bug requires calling `rewrite_coeff()` in `grow()` before creating children, which currently triggers JAX compilation eagerly. The grow phase optimization proposes deferring JAX compilation, which would make this fix viable without the compilation cost. These two upgrades should be coordinated.

---

## Grow Phase Optimization

### Brief

The grow phase dominates search time at 96% of `search_loop` (~18.8s per iteration). Three architectural issues are identified:

1. `sympy.expand()` is called eagerly in `SearchModel.__init__` for every child, but `SearchModel` is a generic wrapper that shouldn't care about expression normalization
2. Raw and expanded forms are structurally different trees that produce different children when `grow()` is applied — currently only the expanded form exists (hidden in `__init__`)
3. Children always fail `filter_no_C` because `cs` is never populated, wasting all downstream filter work

### Status

**Not started.** Performance analysis is documented. No code changes yet.

### Todos

- [ ] Move `sympy.expand()` from `SearchModel.__init__` into `grow()` — it's a grow-level operation, not a wrapper concern
- [ ] Create two `SearchModel` instances per child: one from raw expression, one from expanded form
- [ ] Make `expr` lazy in `SearchModel` — only compute on first access (for raw models)
- [ ] Fix `cs` population for children — call `rewrite_coeff()` in `grow()` before creating child models
- [ ] Defer JAX compilation — separate `rewrite_coeff()` (symbol rewriting) from `build_jax_model()` (JAX compilation)
- [ ] Move `_uniquify()` after filtering — deduplicating rejected expressions is wasteful
- [ ] Reduce branching factor in `grow()` — reduce pool sizes or add size-based pruning during growth
- [ ] Add benchmarks to measure improvement

### Important Context

**Cost breakdown per iteration (20 parents, ~5,200 children):**

| Phase | Time | % of search_loop |
|---|---|---|
| `grow()` — tree traversal + expression construction | ~17.6 s | 96.0% |
| `_uniquify()` — `evalf()` dedup | ~0.3 s | 1.6% |
| `SearchModel.__init__` — `sympy.expand()` + object creation | ~0.2 s | 1.1% |
| `filter_models()` — filter tree walks | ~0.1 s | 0.5% |
| `peek_eval()` — JAX fitting on subset | ~0.05 s | 0.3% |
| `full_eval()` — JAX fitting on full data | ~0.5 s | 2.7% |

**Per parent, ~261 children are produced:**

| Operator | Children | What it does |
|---|---|---|
| `var_xpnd` | ~3,100/iter | Variable substitution — replace variables with complex terms |
| `add_xpnd` | ~2,400/iter | Addition extension — add terms to Add nodes |
| `mul_xpnd` | ~4,800/iter | Multiplication extension — multiply by new factors |

**Two-model per child rationale:** The raw form `(C*x0 + C) * (C*x1 + C)` has a `Mul` at the top with two `Add` children. The expanded form `C**2*x0*x1 + C**2*x0 + C**2*x1 + C**2` has an `Add` at the top with four `Mul` children. Different tree structures → different grow operator matches → different children. Growing from both doubles the exploration surface.

**Interaction with coefficient expansion:** Fixing the bare `C` bug requires calling `rewrite_coeff()` in `grow()`. Currently this builds `jax_model` eagerly, triggering JAX compilation for every child (~52,000 compilations per 10-iteration run). The deferred JAX compilation proposed here makes the bare `C` fix viable. These two upgrades should be coordinated — the deferred compilation work benefits both.

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
