# pge-jax Status

## Current capabilities

The package provides a working PGE search loop and JAX-based expression evaluation. Current behavior includes:

- Fraction-based progressive evaluation with `peek_fraction=0.0` by default
- Lazy `SearchModel.expr` expansion
- Bare `C` rewriting before grown models are created
- Separate coefficient rewriting and JAX wrapper construction
- Raw and expanded models for each generated child
- JAX wrapper construction during evaluation for grown models
- Multiple expanders configured with dictionaries or `ExpanderConfig`
- Model IDs, parent lookups, filter counts, memoization counts, and stage timings

The latest verification collected 413 tests: 323 passed and 90 were skipped. The skipped tests require review before they can serve as specifications. Library code passes mypy. Ruff passes for `pge_jax/`; 14 existing errors remain in test files.

## Work queue

| Priority | Work | State |
|----------|------|-------|
| 1 | Grow-phase performance | Remaining deduplication, pruning, and benchmark work |
| 2 | Work-filter stage pipeline | Not implemented |
| 3 | Named and physical coefficients | Design only; implement after grow work |
| Ongoing | Review findings | See `REVIEW.md` |

## 1. Grow-phase performance

### Current constraints

`Grower.grow()` and `Grower.grow_with_stats()` still construct raw and expanded `SearchModel` pairs before filtering. Each performs local `_uniquify()` calls, coefficient rewriting, expansion, and model construction. The two entry points duplicate this assembly logic.

`_uniquify()` calls `evalf()` on candidate expressions. The grow operators also recursively traverse expression trees and expand every matching node across their term pools, so candidate count remains a central performance constraint.

Instrumentation computes average sizes for grown models, filters, and algebra variants. These calculations can force lazy expression expansion and size computation even when a later stage rejects the model.

### Remaining design

#### Deduplication and model construction

Reduce work performed on candidates that filtering will reject while preserving the existing filter interface.

The preferred direction is Option B:

1. Keep filters operating on `SearchModel` instances.
2. Avoid unnecessary pre-filter uniquification and size calculation where possible.
3. Preserve raw and expanded candidates with the same coefficient and variable lists.
4. Deduplicate before persistent memoization and model-ID assignment.
5. Measure candidate counts and output equivalence against the current implementation.

Returning raw expressions and rewriting every filter to accept them is a broader alternative, not the default approach.

#### Branching control

Add opt-in controls without changing default search semantics:

- `max_grow_children` on `Grower`
- Size-aware pruning during growth where a child can be rejected before further work
- Smaller or policy-specific term pools where benchmarks support the change
- A pruning stage when the pipeline abstraction is available

The ordering and selection rule for `max_grow_children` must be deterministic under `random_seed`.

#### Shared grow implementation

Extract one child-construction path for `grow()` and `grow_with_stats()` so semantics and optimizations remain aligned. Instrumented growth should avoid introducing size calculations that uninstrumented growth would not perform.

### TODO

- [ ] Add reproducible grow benchmarks and record baseline results
- [ ] Extract shared child construction for both grow entry points
- [ ] Reduce pre-filter deduplication, expansion, and size work without changing candidate semantics
- [ ] Add `max_grow_children` as an opt-in limit
- [ ] Benchmark children per parent, rejection rate, deduplication rate, model builds, and wall time

Detailed behavior and design notes: `.design/grow-phase-performance.md`.

## 2. Work-filter stage pipeline

`PGE._preloop()` and `PGE._loop()` currently call each stage directly. There is no `pge_jax.pipeline` module. The goal is composability and clearer stage boundaries, not a change to search results.

### Proposed contract

```python
Stage = Callable[[list[SearchModel], PGE], list[SearchModel]]
```

A named stage object should support uniform timing and error handling:

```python
@dataclass
class Stage:
    name: str
    func: Callable[[list[SearchModel], PGE], list[SearchModel]]
```

Candidate stages:

- Work: initial expressions, grow, algebra
- Filters: expression filters, memoization, deduplication, pruning
- Evaluation: peek evaluation, full evaluation
- Selection: NSGA-II population selection

A representative main pipeline:

```python
pipeline = [
    Stage("grow", grow_stage),
    Stage("filter", filter_stage(default_filters)),
    Stage("memoize", memoize_stage),
    Stage("algebra", algebra_stage),
    Stage("filter", filter_stage(default_filters)),
    Stage("memoize", memoize_stage),
    Stage("peek_eval", peek_eval_stage),
    Stage("selection", selection_stage(peek_count)),
    Stage("full_eval", full_eval_stage),
]
```

The exact order must match the behavior selected by current configuration, including the no-algebra and peek-disabled paths.

### Implementation sequence

- [ ] Define `Stage` and a stage runner in `pge_jax/pipeline.py`
- [ ] Extract grow, filter, memoize, algebra, evaluation, and selection factories
- [ ] Convert `_loop()` while preserving population updates and reporting
- [ ] Convert `_preloop()` with the same abstraction
- [ ] Incorporate grow deduplication and pruning as explicit stages where useful
- [ ] Preserve timing names or define and test explicit replacements

### Acceptance criteria

- Empty candidates terminate the relevant iteration without invalid stage calls.
- Multi-expander behavior remains intact.
- Model IDs, parent relationships, populations, and final Pareto fronts are equivalent to direct stage calls.
- Peek-off and algebra-off configurations do not perform hidden stage work.
- Timing and progress reporting remain usable.

The 90 skipped tests must be curated as part of this work. Enable tests that describe current or newly implemented behavior, rewrite tests that need a different contract, and remove tests that specify neither.

## 3. Named and physical coefficients

The existing implementation supports free `C_i` coefficients only. `N_i`, `P_i`, a coefficient registry, and `SystemModel` are not implemented.

The proposed extension adds:

| Kind | Symbol | Optimization | Scope |
|------|--------|--------------|-------|
| Free | `C_i` | Fitted per expression | Expression-local |
| Named | `N_i` | Fitted | Shared across expressions |
| Physical | `P_i` | Fixed | Registry-backed constant |

Requirements:

- Named and physical kinds are opt-in.
- The grower enumerates each kind and supported mixed-kind combinations.
- Named coefficients support count-based and name-based configuration.
- Physical constants have user-provided values and a built-in reference catalog.
- Free and named coefficients share one optimization vector.
- Physical constants remain symbolic for display and are substituted before JAX construction.
- `SystemModel` represents multiple equations; system-level fitness remains out of scope initially.
- Existing `C_i` behavior remains unchanged when new kinds are disabled.

### TODO

- [ ] Add `CoefficientRegistry` in `pge_jax/coeff_registry.py`
- [ ] Add `SystemModel` in `pge_jax/system_model.py`
- [ ] Extend coefficient rewriting and extraction for C/N/P symbols
- [ ] Substitute physical constants before JAX construction
- [ ] Extend term pools and growth for enabled coefficient kinds
- [ ] Define and test kind-aware memoization and expression-size rules
- [ ] Add PGE configuration for named coefficients and physical constants
- [ ] Update filters and public exports
- [ ] Add focused and integration tests

Full design: `.design/coefficient.md`.

## Verification commands

```bash
source .venv/bin/activate
python -m pytest tests/ -q
ruff check pge_jax/
ruff check tests/
mypy pge_jax/
```

Documentation changes should report test and lint results without treating unrelated existing failures as regressions.
