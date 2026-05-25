# Design: Refactor peek_npts → peek_fraction

## 1. Goals

Two changes, both in the PGE constructor interface:

1. **Parameter interface**: `peek_npts: int = 16` → `peek_fraction: float = 0.0`. The subset size scales with dataset size instead of being a fixed count.
2. **Default behavior**: The old default was `peek_npts=16` (peek **on by default**). The new default is `peek_fraction=0.0` (peek **off by default**). Peek only runs when the user explicitly sets a positive value.

This is a behavioral shift: existing code that doesn't set `peek_npts` will get different behavior after the refactor.

## 2. Understanding: How peek works

### The progressive evaluation strategy

PGE generates candidate mathematical expressions and needs to evaluate them against training data to score them. Full evaluation is expensive because it involves:
1. Fitting coefficients via Levenberg-Marquardt optimization (up to 200 iterations)
2. Computing predictions on all training data
3. Computing regression metrics (RMSE, R², AIC, BIC, etc.)

To avoid this cost on bad candidates, PGE uses a two-stage evaluation:

**Stage 1 — Peek (fast partial evaluation):**
- Evaluate candidates on a small random subset of training data
- Score them quickly
- Use NSGA-II multi-objective selection to keep only the best `peek_count` models

**Stage 2 — Full evaluation:**
- Only the selected `peek_count` models from Stage 1 are fully evaluated on all training data
- These models are then pushed to the final Pareto front and added to the search population

### The parameter: how many data points for peek?

**Current implementation (`peek_npts: int = 16`):**
- An absolute count, default 16 (peek is **on by default**)
- When `peek_npts == 0`: skip peek entirely; all models go straight to full evaluation
- When `peek_npts > 0`: randomly sample exactly `peek_npts` points from training data, use those for peek evaluation
- The sampled points are stored in `self.X_peek` and `self.Y_peek`

**Problem with absolute counts:**
- `peek_npts=16` means very different things for a dataset of 50 points vs 10,000 points
- Users have to guess what count gives a good trade-off between speed and quality
- No way to express "use 25% of my data" — they have to calculate it themselves

### The new parameter: `peek_fraction: float = 0.0`

- A fraction of training data (0 to 1), default 0.0 (peek is **off by default**)
- When `peek_fraction == 0`: skip peek entirely; all models go straight to full evaluation
- When `peek_fraction > 0`: compute `peek_npts = max(1, int(peek_fraction * eval_npts))`, sample that many points
- When `peek_fraction >= 1`: skip peek (same as 0) — no benefit to running peek on all data

### Behavioral shift

| | Old default | New default |
|---|---|---|
| Parameter | `peek_npts=16` | `peek_fraction=0.0` |
| Peek runs by default? | Yes | No |
| User must set explicitly? | No (it's on) | Yes (must set > 0) |

The "opt-in via 0" model: peek is off by default, user opts in by setting a positive fraction. There is no separate enable/disable toggle — the parameter value controls everything.

## 3. Current code structure

### PGE class attributes (search.py)

```python
self.X_train: np.ndarray          # full training data
self.Y_train: np.ndarray          # full training targets
self.X_peek: np.ndarray           # peek subset (or full data if peek off)
self.Y_peek: np.ndarray           # peek targets
self.eval_npts: int               # len(Y_train)
self.peek_npts: int               # CURRENT: absolute count (16)
self.peek_fraction: float         # NEW: fraction (0.0)
self.peek_count: int              # how many models to pop from peek heap (6)
self.peekd_models: int            # count of models that were peek-evaluated
self.peek_nfev: int               # total function evaluations during peek
```

### Data setup (`_set_data`)

Called during `fit()`. Sets up training data and the peek subset.

Current logic:
```python
self.eval_npts = len(self.Y_train)
if self.peek_npts > 0 and self.peek_npts < self.eval_npts:
    pos = np.random.choice(self.eval_npts, self.peek_npts, replace=False)
    self.X_peek = self.X_train[pos, :]
    self.Y_peek = self.Y_train[pos]
else:
    self.X_peek = self.X_train
    self.Y_peek = self.Y_train
```

Key insight: when peek is disabled (`peek_npts == 0`), `X_peek` and `Y_peek` are set to the full training data. This is because `_eval_models` checks `if self.X_peek is None` — it never checks for "peek off." The conditional skip happens at the call site (`_preloop` and `_loop`), not inside `_eval_models`.

### Evaluation (`_eval_models`)

This function is called with `peek=True` or `peek=False`:

```python
def _eval_models(self, models, peek=False):
    for modl in models:
        # Uses self.X_peek and self.Y_peek for fitting
        xs_inputs = [jnp.asarray(self.X_peek[:, i]) for i in range(self.X_peek.shape[1])]
        y_true = jnp.asarray(self.Y_peek)
        fit_result = fit_model(modl.jax_model, y_true, *xs_inputs, max_iter=200)
        modl.jax_model.c_values = fit_result.coefficients

        # Always evaluates predictions on FULL training data for score
        full_xs = [jnp.asarray(self.X_train[:, i]) for i in range(self.X_train.shape[1])]
        full_y = jnp.asarray(self.Y_train)
        y_pred = modl.jax_model.jax_fun(fit_result.coefficients, *full_xs)
        eval_result = evaluate(modl.jax_model, full_y, y_pred)

        if peek:
            # Store peek-specific metrics
            modl.peek_score = eval_result.score
            modl.peek_r2 = eval_result.r2
            # ... etc
            self.peek_nfev += modl.peek_nfev
            self.peekd_models += 1
            modl.peeked = True
        else:
            self.eval_nfev += modl.eval_nfev
            self.evald_models += 1
            modl.evaluated = True
```

Important: peek evaluation uses the peek subset for fitting coefficients, but always computes the final score on full training data. The peek metrics (`peek_score`, `peek_r2`, etc.) are stored for NSGA-II selection during peek phase.

### Preloop (`_preloop`)

Initial population generation. Flow:
1. Generate seed expressions via `Grower.first_exprs()`
2. Filter invalid expressions
3. Memoize (dedup)
4. Algebraic manipulation (expand, factor)
5. Filter + memoize algebra results
6. Combine algebra and non-algebra models into `to_peek`
7. **Peek conditional:**
   - If `peek_npts == 0`: skip peek, `to_eval = to_peek`
   - Else: `_eval_models(to_peek, peek=True)`, push to peek heap, pop best via `_peek_pop()` (double pop first time)
8. Full evaluate: `_eval_models(to_eval)`
9. Push to final list

### Main loop (`_loop`)

Same pattern as preloop but for each search iteration:
1. Grow candidates from population
2. Filter, memoize, algebra
3. Combine into `to_peek`
4. **Peek conditional:**
   - If `peek_npts == 0`: `to_eval = to_peek`
   - Else: `_eval_models(to_peek, peek=True)`, push to peek heap, pop best
5. Full evaluate and push to final

### Finalization (`finalize`)

Prints statistics including:
```python
print(f"num peek evals:    {self.peek_nfev} ({self.peek_nfev * self.peek_npts} point-evals)")
print(f"num eval evals:    {self.eval_nfev} ({self.eval_nfev * self.eval_npts} point-evals)")
print(f"num total evals:   {self.peek_nfev * self.peek_npts + self.eval_nfev * self.eval_npts}")
```

These compute total point-evaluations by multiplying eval count by data points used. Note: `self.peek_npts` in the print refers to the **constructor argument** (an absolute count), not the computed subset size. With `peek_fraction`, the subset size varies per run (depends on dataset size), so we need to store the computed value as an instance attribute to preserve this output.

## 4. Changes needed

### Change 1: Constructor parameter

**Before:**
```python
peek_npts: int = 16,
...
self.peek_npts: int = peek_npts
```

**After:**
```python
peek_fraction: float = 0.0,
...
self.peek_fraction: float = peek_fraction
```

**Rationale:** Rename the parameter and attribute. Default changes from 16 (peek on) to 0.0 (peek off).

### Change 2: `_set_data()` — how peek data is sampled

**Before:**
```python
if self.peek_npts > 0 and self.peek_npts < self.eval_npts:
    pos = np.random.choice(self.eval_npts, self.peek_npts, replace=False)
    self.X_peek = self.X_train[pos, :]
    self.Y_peek = self.Y_train[pos]
else:
    self.X_peek = self.X_train
    self.Y_peek = self.Y_train
```

**After:**
```python
if self.peek_fraction > 0 and self.peek_fraction < 1:
    self.peek_npts = max(1, int(self.peek_fraction * self.eval_npts))
    pos = np.random.choice(self.eval_npts, self.peek_npts, replace=False)
    self.X_peek = self.X_train[pos, :]
    self.Y_peek = self.Y_train[pos]
else:
    self.peek_npts = self.eval_npts
    self.X_peek = self.X_train
    self.Y_peek = self.Y_train
```

**Rationale:** `peek_fraction` is a fraction (0 to 1). Convert to an absolute count before sampling. When `peek_fraction == 0` or `>= 1`, skip peek. The computed `peek_npts` is stored as an instance attribute so `finalize()` prints can still report point-evals accurately.

**Why store `self.peek_npts`:** The `finalize()` prints use `self.peek_npts` to compute point-evals. With `peek_fraction`, the subset size is computed at runtime (depends on dataset size), so we need to preserve it as an attribute. This is cleaner than dropping the point-evals output or computing it from the fraction (which would lose precision from the `int()` truncation).

### Change 3: `_preloop()` — conditional skip

**Before:**
```python
if self.peek_npts == 0:
    to_eval = to_peek
else:
    self._eval_models(to_peek, peek=True)
    self._peek_push_models(to_peek)
    to_eval = self._peek_pop() + self._peek_pop()
```

**After:**
```python
if self.peek_fraction == 0:
    to_eval = to_peek
else:
    self._eval_models(to_peek, peek=True)
    self._peek_push_models(to_peek)
    to_eval = self._peek_pop() + self._peek_pop()
```

**Rationale:** Check `peek_fraction` instead of `peek_npts`. When 0, skip peek and evaluate all candidates directly.

### Change 4: `_loop()` — conditional skip (same pattern)

**Before:**
```python
if self.peek_npts == 0:
    to_eval = to_peek
else:
    self._eval_models(to_peek, peek=True)
    self._peek_push_models(to_peek)
    to_eval = self._peek_pop()
```

**After:**
```python
if self.peek_fraction == 0:
    to_eval = to_peek
else:
    self._eval_models(to_peek, peek=True)
    self._peek_push_models(to_peek)
    to_eval = self._peek_pop()
```

**Rationale:** Same as `_preloop`, just the main loop version.

### Change 5: `finalize()` — print statements

**No change needed.** The prints already use `self.peek_npts`, which is now set in `_set_data()` (either the computed subset size or `self.eval_npts` when peek is off). The output is preserved exactly:

```python
print(f"num peek evals:    {self.peek_nfev} ({self.peek_nfev * self.peek_npts} point-evals)")
print(f"num eval evals:    {self.eval_nfev} ({self.eval_nfev * self.eval_npts} point-evals)")
print(f"num total evals:   {self.peek_nfev * self.peek_npts + self.eval_nfev * self.eval_npts}")
```

**Rationale:** By storing the computed `peek_npts` as an instance attribute in `_set_data()`, the existing prints work without modification. This is preferable to dropping the point-evals output — it's useful diagnostic information.

### Change 6: Tests

**`test_pge_single_var`:**
- Before: `peek_npts=8` (8 out of 50 training points)
- After: `peek_fraction=0.16` (8/50 = 0.16)
- This test exercises the peek code path and should continue to do so.

**`test_pge_multi_var`:**
- Before: `peek_npts=10` (10 out of 80 training points)
- After: `peek_fraction=0.125` (10/80 = 0.125)

**New tests:**
- `test_peek_default_off`: verify `peek_fraction=0.0` (default) skips peek entirely
- `test_peek_fraction_opt_in`: verify setting `peek_fraction=0.2` on 100-point data samples 20 points and runs peek

### Change 7: README

- All `peek_npts=16` become `peek_fraction=0.25`
- Descriptions change from "number of data points" to "fraction of training data"
- Default value changes from 16 to 0.0 in any documentation of defaults
- Note that peek is off by default and must be explicitly enabled

### Change 8: Other references

The following files also reference `peek_npts` and should be updated:
- `AGENTS.md:89` — architecture reference
- `STATUS.md:62,109` — internal status doc
- `hack/quickstart.py:21` — example script
- `hack/benchmark.py:34,56,59` — benchmark configs

## 5. What does NOT change

- `_eval_models()` logic — it already uses `self.X_peek`/`self.Y_peek` without knowing how they were computed
- `SearchModel` — peek attributes (`peek_score`, `peek_r2`, etc.) are unchanged
- Selection logic (`selNSGA2`, `_peek_pop`) — works on whatever models are in the peek heap
- Full evaluation always runs on all training data regardless of peek

## 6. Risks and edge cases

- **`peek_fraction >= 1`**: treated the same as 0 (skip peek). No benefit to running NSGA-II selection on all data.
- **`peek_fraction <= 0`**: treated the same as 0 (skip peek).
- **Very small datasets**: `max(1, int(peek_fraction * eval_npts))` ensures at least 1 point is sampled.
- **Backward compatibility**: existing code using `peek_npts=16` will get different behavior (peek off) after the refactor. Users must explicitly set `peek_fraction` to re-enable peek.
