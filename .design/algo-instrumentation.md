# PGE Algorithm Instrumentation Audit & Plan

## Current State

### Per-Model Fields on `SearchModel` (`search_model.py:88-202`)

| Field | Type | Set? | Where Set |
|---|---|---|---|
| `id` | int | **BUG** | Never — all models have `id=-2` |
| `iter_id` | int | Yes | `_assign_iter_id()` |
| `parent_id` | int | Yes | `__init__` (default -2), `_algebra_models` (algebra children) |
| `gen_relation` | str | Yes | `__init__`, `_algebra_models` |
| `children_count` | int | Yes | `search.py:332` (set on **parent**, not child) |
| `rejected` | bool | **NEVER** | Declared but dead |
| `rejection_reason` | str | **NEVER** | Declared but dead |
| `peek_nfev` | int | Yes | `_eval_models` line 535 |
| `eval_nfev` | int | Yes | `_eval_models` line 526 |
| `total_fev` | int | **NEVER** | Declared but dead |
| `grow_time` | float | Yes | `search.py:333` (set on **parent**) |
| `filter_time` | float | **NEVER** | Declared but dead |
| `memoize_time` | float | **NEVER** | Declared but dead |
| `algebra_time` | float | **NEVER** | Declared but dead |
| `jax_build_time` | float | Yes | `_eval_models` line 478 |
| `fit_time` | float | Yes | `_eval_models` line 496 |
| `evaluate_time` | float | Yes | `_eval_models` line 514 |
| `total_time` | float | **Wrong** | `search.py:548` — currently `time.time() - _eval_start` (only covers eval window, misses grow/filter/memoize/algebra) |
| `_eval_start` | float | Yes | `search.py:466` — used only by the broken `total_time` calculation. Can be removed once `total_time` is fixed to a sum. |

### Aggregate Fields on `PGE` (`search.py:181-193`)

| Field | Set? | Notes |
|---|---|---|
| `curr_iter` | Yes | Current iteration number |
| `peekd_models` | Yes | Count of peek-evaluated models |
| `evald_models` | Yes | Count of fully evaluated models |
| `peek_nfev` | Yes | Cumulative peek function evals (count, not time) |
| `eval_nfev` | Yes | Cumulative full eval function evals (count, not time) |
| `_phase_times` | Yes | **Wall-clock** phase timing — should NOT sum per-model values |
| `_loop_phase_times` | Yes | Wall-clock per-iteration phase timing |

### Timing Model

**Rule: aggregate timing is wall-clock, not summation.**

Aggregate times (full PGE run, per-iteration totals, per-stage totals) must be measured as `t_end - t_start` wall-clock deltas. Never sum per-model timings to produce aggregate totals — that double-counts because each model's `fit_time` is already the time for that single model, and summing 98 of them gives 98× the actual wall time.

Per-model timing fields are per-model. If you need a total across all models, sum lazily at report time (e.g. `sum(m.fit_time for m in pge.final)`).

**Correct pattern:**
```python
# Aggregate (wall-clock) — in _loop() or _preloop()
t0 = time.time()
# ... do work ...
self._phase_times["grow"] = time.time() - t0
```

**Incorrect pattern (current bug):**
```python
# Per-model — in _eval_models()
modl.fit_time = time.time() - fit_t0  # correct: per-model
self._phase_times["fit"] += modl.fit_time  # WRONG: sums per-model into aggregate
```

### Phase Timing (`finalize()` output)

**Working — wall-clock aggregate timing:**
- `data_setup`, `preloop`, `search_loop` — top-level phases (measured via `t0`/`time.time()` in `fit()`)
- `grow`, `filter`, `algebra`, `peek_eval`, `full_eval` — search loop breakdown (measured via `t0`/`time.time()` in `_loop()`)

**Broken — per-model sums masquerading as aggregates:**
- `jax_build`, `fit`, `evaluate` — currently computed as `self._phase_times["fit"] += modl.fit_time` in `_eval_models()` (line 550). This sums 98 per-model values into a single key, producing ~2.1s when the actual wall-clock is ~0.7s.
- `finalize` shows **-2.4s (-93.5%)** — the calculation at line 879:
  ```python
  self._phase_times["finalize"] = runtime - sum(self._phase_times.values())
  ```
  This subtracts the inflated per-model sums (`jax_build + fit + evaluate`) plus `preloop + search_loop` from `runtime`. Since `fit` alone is 3× the actual wall time, the result is negative.

## Critical Bugs

### 1. Model ID Assignment Dead Code

`_memoize_models()` (line 671) assigns `m.id = len(self.models)` and appends to `self.models`, but it is **never called** for the main pipeline. The `_run_loop_pipeline()` method uses `Stage` objects with inline lambdas (`capture_unique`, `capture_algebra`) that only record counts to a dict — they never call `_memoize_models`.

**Evidence:** After a full run, `pge.models` is always empty (`len=0`), `pge.hmap` is empty (`len=0`), and every model has `id=-2`.

**Impact:** `_compute_improvements()` (line 560) does `self.models[modl.parent_id]` to find the parent — but since `self.models` is empty, parent lookups silently fail for all models. The improvement fields are always set to the "first generation" negative values.

### 2. `finalize` Time Calculation Negative

Line 879 computes `finalize` as `runtime - sum(all_phase_times)`. The `fit` and `evaluate` phase times are **per-model sums** (each model's `fit_time` is added to the aggregate), so `fit` alone is ~2.1s while actual wall-clock is ~0.7s. Result: negative finalize time.

### 3. `filter_time`, `memoize_time`, `algebra_time` Always Zero

These fields exist on `SearchModel` but are never set. The `Stage` class (`pipeline.py:54-58`) tracks wall-clock timing to `pge._loop_phase_times[self.name]` for the aggregate, but:
- The `_run_loop_pipeline()` inline lambdas bypass `Stage` for the first filter/memoize
- Even when `Stage` is used, it only accumulates to the global dict, never propagates to individual models
- The `_memoize_models` method (which would be the natural place) is never called

### 4. `rejected` / `rejection_reason` Never Set

`filter_models()` in `filters.py` silently drops rejected models. There is no record of **why** a model was rejected, or even **how many** were rejected per filter.

### 5. `total_fev` Never Set

`peek_nfev` and `eval_nfev` are set individually but never summed into `total_fev`.

## Pipeline Flow Analysis

```
_run_loop_pipeline(expanded)
  ├─ Stage("filter", lambda: filter_models(...))   → timing → _loop_phase_times["filter"]
  ├─ Stage("memoize", capture_unique)              → timing → _loop_phase_times["memoize"]
  │   (capture_unique only records to dict, never calls _memoize_models!)
  ├─ Stage("algebra", lambda: _algebra_models(...)) → timing → _loop_phase_times["algebra"]
  ├─ Stage("filter", lambda: filter_models(...))   → timing → _loop_phase_times["filter"]
  └─ Stage("memoize", capture_algebra)             → timing → _loop_phase_times["memoize"]
      (_memoize_models called separately for algebrad models at line 436)
  └─ returns algebrad_unique + unique
```

The `capture_unique` and `capture_algebra` callbacks are **no-ops for memoization** — they only store references to dicts. The actual `_memoize_models` (which assigns IDs and populates `self.models`) is only called for algebra children, not for the main candidate flow. This is why `pge.models` stays empty and all IDs are -2.

## What's Actually Useful in Output

The `finalize()` printout is mostly useless for debugging because:
1. All model IDs are `-2` — cannot distinguish models
2. Parent IDs are `-1` or `-2` — no lineage information
3. `grow_time` is set on **parents** (the popped models), not on the children that were actually generated
4. `children_count` is on parents, showing how many children each parent produced
5. No rejection tracking — don't know how many models were filtered out

## Plan

### Phase 1: Fix Critical Bugs

#### 1.1 Wire `_memoize_models` into the pipeline

Replace the `capture_unique`/`capture_algebra` dict-capture pattern with actual calls to `_memoize_models`. The pipeline should:
- Call `_memoize_models()` after the first filter stage (assigns IDs, populates `self.models`)
- Call `_memoize_models()` after algebra stages for algebra children

This fixes:
- Model IDs being assigned
- `self.models` list being populated
- `_compute_improvements()` parent lookups working
- `parent_id` lineage being correct

#### 1.2 Fix aggregate timing — stop summing per-model into globals

**Remove per-model summation from `_eval_models()`** (lines 549-551):
```python
# DELETE these lines:
self._phase_times["jax_build"] = self._phase_times.get("jax_build", 0) + modl.jax_build_time
self._phase_times["fit"] = self._phase_times.get("fit", 0) + modl.fit_time
self._phase_times["evaluate"] = self._phase_times.get("evaluate", 0) + modl.evaluate_time
```

**Add wall-clock timing in `_eval_models()`** instead:
```python
def _eval_models(self, models, peek=False):
    t0 = time.time()
    # ... existing eval loop ...
    elapsed = time.time() - t0
    key = "peek_eval" if peek else "full_eval"
    self._phase_times[key] = self._phase_times.get(key, 0) + elapsed
```

Wait — `peek_eval` and `full_eval` are already tracked via wall-clock in `_loop()` (lines 359, 367). The `jax_build`, `fit`, `evaluate` sub-timing should be measured as wall-clock deltas in a single wrapper around `_eval_models`, not per-model sums.

**Correct approach:** Measure `jax_build`, `fit`, `evaluate` as wall-clock in `_eval_models()`:
```python
def _eval_models(self, models, peek=False):
    jax_t0 = time.time()
    # ... build jax models ...
    fit_t0 = time.time()
    # ... fit ...
    eval_t0 = time.time()
    # ... evaluate ...
    self._phase_times["jax_build"] = self._phase_times.get("jax_build", 0) + (jax_build_end - jax_t0)
    self._phase_times["fit"] = self._phase_times.get("fit", 0) + (fit_end - fit_t0)
    self._phase_times["evaluate"] = self._phase_times.get("evaluate", 0) + (eval_end - eval_t0)
```

Or simpler: measure the three sub-phases as wall-clock blocks using `t0`/`time.time()` at the function level, not per-model accumulation.

#### 1.3 Fix `finalize` time calculation

Remove the broken `runtime - sum(...)` calculation. Measure `finalize` as wall-clock:
```python
def finalize(self, ...):
    t0 = time.time()
    # ... existing finalize logic ...
    self._phase_times["finalize"] = time.time() - t0
```

### Phase 2: Per-Model Timing (correct model)

Per-model timing fields are **per-model** — they measure time for that single model's lifecycle. They are NOT summed into aggregates.

#### 2.1 Set per-model stage timing

Each model should track how long it spent in each pipeline stage:
- **filter_time**: Time spent in `filter_models()` for this model. Set in `filter_models()` or via a per-model callback in the Stage wrapper.
- **memoize_time**: Time spent in `_memoize_models()` for this model (first insertion). Set in `_memoize_models()` when `m.memoized = True`.
- **algebra_time**: Time spent in `manip_model()` for algebra-generated children. Set in `_algebra_models()`.

These are per-model values. To get total filter time across all models: `sum(m.filter_time for m in pge.final)`.

#### 2.2 Set `total_fev` on models

In `_eval_models`, after setting `peek_nfev` and `eval_nfev`:
```python
modl.total_fev = modl.peek_nfev + modl.eval_nfev
```

#### 2.3 Per-model total_time — SUM of all stage times (NOT wall-clock)

Currently `modl.total_time = time.time() - modl._eval_start` (line 548) is **wrong**. It only covers the eval window (jax_build + fit + evaluate) and completely misses grow, filter, memoize, and algebra times.

**Fix: compute `total_time` as the sum of all per-model stage times:**
```python
modl.total_time = (modl.grow_time + modl.filter_time + modl.memoize_time
                   + modl.algebra_time + modl.jax_build_time
                   + modl.fit_time + modl.evaluate_time)
```

This is computed lazily at report time, not during the eval loop. The individual stage fields accumulate during the pipeline, and `total_time` is derived from them.

To get total time across all models: `sum(m.total_time for m in pge.final)`.

### Phase 3: Rejection Tracking

#### 3.1 Track which filter rejected each model

Modify `filter_model()` in `filters.py` to return the filter name that rejected, or modify `filter_models()` to return `(accepted, rejected)` pairs. Set `rejected=True` and `rejection_reason` on rejected models.

#### 3.2 Track rejection counts per filter

Add per-iteration rejection counters to PGE:
```python
self.rejection_counts: Dict[str, int] = {}
```

### Phase 4: Per-Iteration Statistics

#### 4.1 Track per-iteration model counts

In `_loop()`, after each pipeline stage, record:
- Models grown
- Models after filter 1
- Models after memoize 1
- Models after algebra
- Models after filter 2
- Models after memoize 2
- Models peek-evaluated
- Models fully evaluated

Store as `self.iter_stats: List[Dict[str, int]]`.

### Phase 5: Parent Lineage

#### 5.1 Ensure parent_id is set on all models

After fixing 1.1, parent_id will be correctly set. Verify that:
- Initial population models have `parent_id=-2`
- Grown children have `parent_id` set to the parent's assigned ID
- Algebra children have `parent_id` set to the source model's ID

#### 5.2 Add lineage traversal helper

Add a method to `SearchModel` or `PGE` to traverse parent lineage:
```python
def lineage(self) -> List[SearchModel]:
    """Return the chain of parents from root to this model."""
```

### Phase 6: Output Improvements

#### 6.1 Print per-model timing in results

Add timing columns to the final results table:
- `grow_t`, `filter_t`, `memo_t`, `alg_t`, `jax_t`, `fit_t`, `eval_t`, `total_t`

#### 6.2 Print rejection summary

After the results table, print:
```
Rejection Summary:
  filter_too_big:    12
  filter_has_int_coeff: 5
  ...
  total rejected:    17
```

#### 6.3 Print per-iteration summary

After the results table, print a compact per-iteration breakdown.

## Priority Order

1. **1.1** — Fix memoization wiring (breaks everything else)
2. **1.2** — Fix finalize time calculation (negative numbers are confusing)
3. **2.2** — Set `total_fev` (one line)
4. **2.1** — Per-model stage timing (depends on 1.1 for IDs to be meaningful)
5. **3.1** — Rejection tracking (useful for understanding search behavior)
6. **4.1** — Per-iteration stats (nice to have for analysis)
7. **5.1** — Parent lineage (follows from 1.1)
8. **6.x** — Output improvements (cosmetic, depends on data from above)
