# Type-Safety Notes

## Current baseline

`mypy pge_jax/` reports no issues in 13 source files. Sympy imports use targeted `# type: ignore[import-untyped]` comments because upstream type information is unavailable. Informational notes about unchecked bodies remain possible in partially annotated modules.

Library linting is clean. Test linting has separate existing failures; type-check success should not be reported as whole-repository lint success.

## Remaining risks

### Dynamic Sympy values

Sympy objects largely appear as unparameterized expressions, symbols, lists, and tuples. Public APIs should define accepted symbolic forms with type hints, but internal tree operations still need focused tests because static analysis cannot verify Sympy rewrite semantics.

### Broad `Any` use in selection

Selection functions accept DEAP-like objects through dynamic attributes. `SearchModel` satisfies this protocol with `values`, `wvalues`, and `crowding_dist`, but the contract is not encoded in a shared type.

Introduce a `Protocol` for selection candidates if it can be done without changing the public selection API. Keep generic algorithm coverage alongside concrete `SearchModel` integration tests.

### Partially configured search state

Training and peek arrays are optional before `fit()` and non-optional during evaluation. Assertions protect some paths, but type narrowing does not establish the larger lifecycle invariant.

Keep dataset conversion and initialization centralized. A stage refactor should avoid widening every stage to accept uninitialized `PGE` state.

### JAX versus NumPy results

Lambdified functions and JAX optimization results can contain dynamic values. Explicit conversions at result boundaries are preferable to broad `Any` suppression. Preserve scalar-versus-array and float64 behavior in numerical tests.

### Metric and optimizer result conventions

`FitResult.nfev` means iterations in current implementations, not function evaluations. A future rename or counter correction must update result construction, summaries, and tests together.

`fit_least_squares` accepts bounds but implements them as penalties. Types cannot communicate that semantic limitation; documentation and tests must do so.

## Configuration policy

Keep the existing direct mypy configuration in `pyproject.toml`. Prefer local, specific ignores with a reason over module-wide or global suppressions. Do not disable `warn_return_any` to hide conversion issues.

Run these checks independently:

```bash
source .venv/bin/activate
mypy pge_jax/
ruff check pge_jax/
ruff check tests/
python -m pytest tests/ -q
```

A clean type check does not replace numerical, symbolic, or integration tests.

## Change guidelines

When touching an incomplete module:

1. Annotate new public interfaces and result types.
2. Tighten local variable annotations only as needed for the change.
3. Replace dynamic behavior with a `Protocol` or typed dataclass where it clarifies an established contract.
4. Use targeted ignores only for verified third-party boundaries.
5. Add runtime tests for invariants mypy cannot prove.
6. Keep lint cleanup separate from algorithmic changes when practical.

## TODO

- [ ] Add a selection-candidate `Protocol` or document why the dynamic API must remain
- [ ] Annotate remaining unchecked utility functions when they are next modified
- [ ] Curate skipped tests and fix test lint violations
- [ ] Add multivariate Jacobian tests to cover behavior static typing misses
- [ ] Add CI or pre-commit enforcement for current clean library mypy and ruff checks
