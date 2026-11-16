# Grow-Phase Performance Design

## Purpose

Define the remaining grow optimizations without changing candidate semantics by default. Current implementation details belong here; task status and sequencing are tracked in `STATUS.md`.

## Current execution model

For each selected parent, `Grower.grow_with_stats()`:

1. Applies variable substitution, addition extension, and multiplication extension recursively to `parent.orig`.
2. Adds or removes a bare coefficient with `_toggle_plus_C` for the primary growth results.
3. Optionally applies top-level addition extension and shrinking.
4. Runs `_uniquify()` on operator result lists using `expr.evalf()` values as set members.
5. Rewrites each bare `C` to numbered `C_i` symbols.
6. Creates one model for the rewritten raw expression and one for its expanded form.
7. Returns children plus operator counts and timings.
8. Computes average child sizes for instrumentation, forcing `size()` on generated models.

The search loop later filters, memoizes, creates algebraic variants, and evaluates candidates. `Grower.grow()` provides the same candidate transformation without statistics, but duplicates the child-construction code.

## Why raw and expanded models both exist

Raw and expanded expressions are distinct sympy trees:

```text
Raw:      (C*x0 + C) * (C*x1 + C)
Expanded: C**2*x0*x1 + C**2*x0 + C**2*x1 + C**2
```

The raw root is a multiplication; the expanded root is an addition with multiple multiplication children. Growth operators therefore expose different transformation sites. Treating these as separate candidates is search behavior, not only an optimization technique.

Both variants receive the same `xs` and `cs`. Bare coefficients must be rewritten before model construction so `filter_no_C` and JAX coefficient inference operate correctly.

## Lazy work and its limits

`SearchModel.expr` expands `_raw_expr` on first access. The expanded variant caches its expression directly. JAX construction for grown models occurs in `_eval_models()` after candidate selection.

Laziness does not guarantee less symbolic work:

- `filter_too_big` calls `size()`.
- `size()` computes tree and Jacobian sizes and accesses `expr`.
- Grow statistics call `size()` for every generated model.
- Reports and result formatting also access expression and size fields.

Optimization decisions must account for all callers, especially instrumentation.

## Performance levers

### Shared child construction

Extract one implementation for coefficient rewriting, raw/expanded expansion, model construction, and relation assignment. `grow_with_stats()` should add measurements around the same path rather than maintain a second implementation.

Acceptance requirement: for the same parent and grower configuration, both entry points return equivalent candidate expressions and metadata.

### Deduplication placement

Current deduplication happens before cross-stage filtering. Moving or reducing it can avoid `evalf()` for rejected candidates, but also changes how many models reach filters and how counts are reported.

Preferred approach:

1. Preserve `SearchModel`-based filters.
2. Separate cheap per-operator duplicate suppression from expensive pre-filter normalization where useful.
3. Deduplicate surviving candidates before persistent memoization.
4. Avoid forcing `size()` solely for optional reporting.

A raw-expression filter pipeline remains possible, but requires a separate design because filters depend on `SearchModel.orig`, `cs`, and `size()`.

### Branching controls

Add an optional `max_grow_children` limit. Define:

- Whether the cap applies before or after raw/expanded pairing
- Whether it is per operator, per parent, or per expander
- How candidates are ordered or sampled under a fixed seed
- How raw and expanded partner retention is handled
- How capped counts appear in instrumentation

The default must remain unlimited. Size-aware pruning may reject candidates earlier only when it applies the same size rule as normal filtering.

### Term-pool policies

Growth cost grows with matching-node count and term-pool size. Benchmark policy changes separately from refactoring because smaller pools change the search space. Existing levels and `limiting_depth` should remain the baseline configuration.

### Instrumentation cost

Full child-size summaries mix observability with search cost. Prefer:

- Counts and timings on every run
- Average sizes after filtering for normal reporting
- Optional detailed size metrics for dedicated profiling runs
- Equivalent instrumentation on any replacement pipeline stage

## Benchmark plan

Use fixed seeds, representative variable counts, and several grow policies. Compare candidates as well as wall time; a faster run that silently drops expressions is not equivalent.

Record:

- Raw expressions and unique expressions per operator
- Children per parent before and after raw/expanded pairing
- Filter acceptances and rejection reasons
- Memoization duplicates
- JAX wrapper builds and successful fits
- Time in operators, uniquification, model construction, filtering, algebra, and evaluation
- Peak candidate counts
- Final Pareto-front equivalence for semantic no-change refactors

Run at least one instrumented and one uninstrumented configuration to measure observer overhead.

## Remaining implementation checklist

- [ ] Add deterministic benchmark fixtures
- [ ] Unify `grow()` and `grow_with_stats()` candidate construction
- [ ] Reduce unnecessary size evaluation in normal instrumentation
- [ ] Implement and validate deduplication placement
- [ ] Add opt-in branching controls
- [ ] Compare benchmark results and document any semantic changes separately
