# PGE Instrumentation Design

## Measurement principles

Use wall-clock deltas for phase and iteration durations. A phase timed around a batch is not the sum of deliberately overlapping or enclosed timers. Keep per-model measurements separate from wall-clock stage reporting.

Do not let normal instrumentation change candidate semantics or cause symbolic work that the search would otherwise avoid. Detailed profiling can be an explicit, measured option.

Model identity and parent lookup are prerequisites for meaningful lineage statistics.

## Current records

### Per-model data

`SearchModel` stores:

- `id`, `iter_id`, `parent_id`, and `gen_relation`
- Lifecycle flags for memoization, algebra, evaluation, selection, finalization, and errors
- `peek_nfev`, `eval_nfev`, and `total_fev`
- `timings: StageTimings` with build, fit, and evaluation durations

`PGE._memoize_models()` assigns IDs and appends accepted models to `PGE.models`. Grown and algebraic children carry parent IDs so `_compute_improvements()` can compare metrics when a valid parent exists.

`total_fev` remains unset. Build, fit, and evaluation timings describe those operations individually and do not form a complete model-lifecycle duration.

### Aggregate search data

`PGE` stores:

- `_phase_times`
- `_loop_phase_times`
- `_loop_stage_times: IterationStageTimes`
- `_loop_sub_times: IterationSubTimes`
- `_iteration_times`
- `_iteration_progress`
- Cumulative peek/full candidate and optimizer-iteration counts

`IterationSubTimes` includes grow expander details, grow operator statistics and timings, filter counts and rejection breakdowns, memoization records, and algebra-method records.

### Filtering and memoization

`filter_models_with_stats()` counts which filter first rejected each candidate. Filters short-circuit, so counts depend on filter order and should not be interpreted as the total number of invalid properties per candidate.

`MemoizeRecord` records input count, accepted unique count, and duplicates for each `_memoize_models()` call. Calls can occur multiple times per iteration, including around algebraic variants.

### Reporting

The search prints configuration, preloop and iteration summaries, final Pareto results, candidate counts, and detailed timing reports. `print_timing()` also prints operator, filter, algebra, memoization, and per-model timing details.

## Known accounting issues

### Optimizer iterations versus evaluations

`FitResult.nfev` counts optimizer iterations. Summary labels currently refer to evaluations and point-evaluations derived from these counts. One optimizer iteration can perform multiple predictions and a Jacobian computation.

Use accurate `optimizer_iterations` and `optimizer_point_iterations` terminology, or implement comprehensive counters. Do not combine partial counters and call them total evaluations.

### `total_fev`

Set `total_fev = peek_nfev + eval_nfev` after fitting if retaining the current fields, preferably through a derived property. Define its meaning consistently with the optimizer-iteration correction.

### Residual phase time

`print_timing()` derives `finalize` by subtracting stored phases from total runtime. This residual includes unmeasured work and can become misleading when phases overlap or are recorded through different paths. Prefer direct timing or label it unaccounted time.

### Observer overhead

Grow statistics compute average sizes for all generated models. `size()` accesses lazy expressions and computes Jacobian size. Normal progress collection also computes average grown, filtered, and algebra sizes. These costs must be included in instrumentation benchmarks.

## Proposed instrumentation layers

### Standard layer

Always lightweight:

- Candidate counts by stage
- Duplicate and errored counts
- Wall-clock stage and iteration durations
- Best score and population size
- Successful peek and full fits

### Detailed layer

Opt-in for profiling:

- Per-operator timings
- Average expression and Jacobian sizes
- Per-model build, fit, and evaluation timings
- Per-filter rejection details beyond simple counters
- Full memoization call logs

Expose configuration through one explicit verbosity or profiling option rather than adding independent booleans across modules.

## Pipeline integration

Each future `Stage` should return models and a stage record, or update a stage context owned by the run. Avoid timing through model mutation when the quantity belongs to a batch.

A stage record should include:

```python
@dataclass
class StageRecord:
    name: str
    iteration: int
    input_count: int
    output_count: int
    wall_time: float
    details: dict[str, object]
```

Repeated stage names require unique sequence numbers or structured details; aggregating two filter or memoize stages into one record loses useful information.

Records should remain available as Python data without parsing console output.

## Remaining TODO

- [ ] Correct evaluation-count terminology and model totals
- [ ] Separate standard and detailed instrumentation costs
- [ ] Test aggregate timing, empty iterations, and multiple same-name stages
- [ ] Preserve stage records as public or documented internal data during pipeline work
- [ ] Verify parent lineage and improvement metrics with focused tests
- [ ] Define stable reporting behavior before extracting console formatting
