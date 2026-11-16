# Progressive Evaluation Design

## Interface

`PGE` exposes `peek_fraction: float = 0.0`. Progressive evaluation is intended to be disabled by default and enabled with a fraction strictly between 0 and 1.

For enabled runs, `_set_data()` computes:

```python
peek_npts = max(1, int(peek_fraction * eval_npts))
```

It samples that many distinct training rows without replacement into `X_peek` and `Y_peek`. `peek_npts` remains available for point-count reporting. `peek_count` independently controls how many models selection advances from the peek population.

## Evaluation semantics

The normal full path fits coefficients on all training data and evaluates predictions on all training data.

The peek path currently:

1. Fits coefficients on `X_peek` and `Y_peek`.
2. Generates predictions on all training data with those coefficients.
3. Computes metrics against all training targets.
4. Stores the same metric values in both full and `peek_*` model fields.
5. Uses NSGA-II to advance up to `peek_count` candidates.
6. Fits and evaluates those candidates again through the normal full path.

This means peek currently reduces the number of rows used for the first coefficient fit, not the rows used for its model metrics. Documentation and benchmarks must not describe the first metrics pass as subset-only.

The preloop performs two `_peek_pop()` calls when progressive evaluation is enabled; the main loop performs one. This distinction is existing search behavior and needs explicit coverage before refactoring.

## Boundary behavior and open defect

The intended rule is to disable peek outside `0 < peek_fraction < 1`. Current loop guards bypass peek only for exactly `0`, while `_set_data()` uses full arrays for every out-of-range value. Negative values and values at or above 1 therefore still run peek fitting and selection using the full arrays.

Use a shared predicate for data setup, preloop, and main-loop dispatch:

```python
peek_enabled = 0 < peek_fraction < 1
```

Decide whether invalid values should disable peek or raise `ValueError`; silent disabling is the smallest behavioral change consistent with the existing interface description. Add boundary tests before changing it.

## Reporting

`print_summary()` derives point-evaluation totals by multiplying optimizer iteration counts by peek or full dataset sizes. `FitResult.nfev` counts iterations, so the resulting labels are approximations rather than complete function-call counts. See `REVIEW.md` for the accounting issue.

When peek runs, a model may contribute both peek and full fitting counts. `SearchModel.total_fev` should reflect that total if the field is retained.

## Refactor constraints

- Preserve random sampling and `random_seed` reproducibility.
- Preserve at least one peek row for very small positive fractions.
- Do not reuse the peek subset as the full training data object when peek is enabled.
- Handle empty peek selections without attempting invalid full evaluation.
- Test both preloop and main-loop populations.
- Keep stage timing semantics explicit as peek moves into any future pipeline abstraction.
- Revisit full-data metric calculation separately if subset-only metrics are desired; that would change selection semantics.

## Tests

Coverage should verify:

- Default construction bypasses peek
- Fractional values compute the expected sample size without replacement
- Negative, zero, one, and greater-than-one values follow the selected boundary policy
- Peek coefficients and full-fit coefficients use their respective datasets
- Metrics are calculated on the documented dataset
- Selection advances the configured number of models
- Empty and failed peek evaluations do not enter final populations
- Point and iteration reporting use accurate labels
