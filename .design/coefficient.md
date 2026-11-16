# Named and Physical Coefficient Design

## Status

Design only. The implementation currently supports free coefficients named `C_0`, `C_1`, and so on. `N_i`, `P_i`, `CoefficientRegistry`, and `SystemModel` do not exist. Implement this work after grow behavior and its tests have stabilized.

## Goals

Add two opt-in coefficient kinds without changing default search behavior:

| Kind | Placeholder | Indexed symbol | Value | Scope |
|------|-------------|----------------|-------|-------|
| Free | `C` | `C_i` | Fitted independently | One expression |
| Named | `N` | `N_i` | Fitted | Shared across expressions |
| Physical | `P` | `P_i` | Fixed numeric constant | Registry-backed |

Free and named coefficients are optimized together. Physical constants remain readable in symbolic expressions and are substituted before JAX construction.

Non-goals for the first implementation:

- System-level fitness
- Joint fitting across separately searched equations
- Automatic discovery of physical constants
- Changes to existing free-coefficient defaults

## Current coefficient contract

The grower uses bare `C` placeholders. `SearchModel.rewrite_coeff_only()` recursively rewrites placeholders to numbered symbols and returns the coefficient list. Initial and algebraic models call `rewrite_coeff()`, which also builds the JAX wrapper. Grown models defer wrapper construction.

Automatic inference in `JAXModel` recognizes `C_` and `C[` prefixes. A mixed-kind implementation must preserve existing rewriting and inference tests before introducing new prefixes.

## Registry

Introduce `pge_jax/coeff_registry.py` with a registry owned by a search or system rather than process-global mutable state. Shared ownership is required; global module state would make independent searches interfere.

The registry must support:

- Named coefficients configured by count
- Named coefficients configured by an ordered list of user names
- A stable mapping between display names and `N_i` symbols
- Physical constants configured by user name and value
- A curated catalog of common physical constants
- User overrides with explicit precedence over catalog values
- Consistent iteration and lookup order for reproducibility

A suitable API should cover:

```python
class CoefficientRegistry:
    def named_symbols(self) -> list[sympy.Symbol]: ...
    def physical_symbols(self) -> list[sympy.Symbol]: ...
    def name_for(self, symbol: sympy.Symbol) -> str: ...
    def physical_values(self) -> dict[sympy.Symbol, float]: ...
```

Validate duplicate names, unknown constant names, non-finite values, and empty configurations.

## Symbolic representation

Keep registry-backed `N_i` and `P_i` symbols in `SearchModel.orig` and `SearchModel.expr`. Display methods should use configured names where available, while preserving stable indexed symbols for identity and serialization.

Extend rewriting so each enabled placeholder kind has a separate mapping and deterministic index assignment. Define how repeated placeholder occurrences are treated and test that behavior explicitly. Existing numbered symbols must not be silently renumbered.

A single coefficient list with kind lookup through the registry is preferred initially. It avoids duplicating coefficient state across `cs_free`, `cs_named`, and `cs_phys`. The optimizer vector contains only free and named symbols.

## JAX conversion and fitting

Before constructing `JAXModel`:

1. Substitute each `P_i` with its numeric registry value in the evaluation expression.
2. Extract or explicitly supply free and named coefficients in a stable order.
3. Construct one coefficient vector containing both fitted kinds.
4. Compute the Jacobian with respect to that complete vector.

Physical constants do not appear in the fitted vector or Jacobian. Preserve the original symbolic model for display instead of overwriting it with its substitution result.

Free and named coefficients have identical fitting mechanics. Sharing named values across equations is represented by shared identity; independent per-equation searches still fit independently until system optimization is designed.

Filtering and metric code must handle numeric substitutions correctly. Known physical constants are legitimate fixed values and should not be confused with hardcoded coefficients introduced by the grammar.

## Growth

The grower must enumerate all enabled kinds, not merely substitute named or physical symbols after free-only growth.

Construct term pools for:

- Free-only terms
- Named-only terms when enabled
- Physical-only terms when enabled
- Supported free-named, free-physical, and named-physical combinations
- Free-named-physical combinations when all are enabled

Combinatorial growth needs explicit control. Reuse existing policy levels and optional child limits rather than unconditionally producing every combination. Benchmark pool sizes and candidate counts before choosing defaults.

Named and physical kinds are disabled when not configured. Existing runs should produce the same candidate streams under that configuration.

## Identity and memoization

Define identity separately for each coefficient kind:

- Indexed symbols preserve their exact indices.
- Free coefficients are expression-local despite reusing indices in different expressions.
- Named indices are meaningful across expressions using the same registry.
- Physical identity includes its registry-backed value if models from different configurations can share a memoizer.

Hashing `model.orig` already distinguishes symbolic structures and indices, but a memoizer must not mix models with incompatible registries. Either scope the memoizer to one search or include registry configuration in the deduplication context.

## Size and fitness

Proposed initial size rules:

| Kind | Size contribution |
|------|-------------------|
| Free | +1 per free coefficient occurrence under existing size rules |
| Named | +1 for each distinct named coefficient |
| Physical | +0 fitted-parameter penalty; symbolic tree cost is defined separately |

The distinction between expression-tree size and fitted-parameter count must remain explicit. A cross-expression penalty such as zero for previously seen named coefficients requires search-global state and should be deferred until deterministic semantics and tests exist.

Start with deterministic per-expression metrics, then evaluate shared-complexity objectives separately.

## System representation

Add `pge_jax/system_model.py` with a `SystemModel` containing:

- An ordered collection of `SearchModel` equations
- A shared coefficient registry
- Optional equation names or output labels
- System-wide symbolic display helpers

The first version is a representation and coordination wrapper. It does not define a joint residual, system fitness, or joint optimizer API.

## Public configuration

Possible constructor shape:

```python
pge = PGE(
    usable_vars=["x"],
    named_coefficients=["mass", "spring", "damping"],
    physical_constants={"g": 9.81},
)
```

Count-based named configuration must also be supported, for example `named_count=3`. Final parameter names and mutual exclusivity rules should be settled before implementation; do not infer values from expression names.

No new coefficient kinds are active when both named and physical configuration are omitted.

## Implementation sequence

- [ ] Add registry types and configuration validation
- [ ] Extend rewriting, explicit extraction, and display for C/N/P
- [ ] Add physical substitution before JAX construction
- [ ] Add isolated free/named/physical growth pools and enabled mixed combinations
- [ ] Update filters, memoization scope, and size calculations
- [ ] Add `SystemModel` representation without system fitness
- [ ] Add PGE setup and public exports
- [ ] Add unit, integration, backward-compatibility, and nonconvergence tests

## Acceptance criteria

- Existing free-only tests pass unchanged.
- Named symbols retain identity across equations using one registry.
- Independent registries do not share mutable values.
- Physical constants are absent from the fitted vector and Jacobian.
- Symbolic display retains physical names.
- Every enabled kind appears in growth tests, including mixed-kind combinations.
- Memoization cannot conflate models from incompatible registries.
- Numerical tests include convergent and nonconvergent nonlinear fits.
