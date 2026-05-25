# Coefficient Management

## Current State

### Two Symbol Conventions

| Convention | Example | Where Used |
|---|---|---|
| Bare `C` | `sympy.Symbol("C")` | `expand.py` term pools (`with_c_xs1_muls`, `var_sub_terms`, etc.) |
| Numbered `C_i` | `sympy.Symbol("C_0")`, `C_1` | After `rewrite_coeff()` is called |

### Flow

1. **`Grower`** builds term pools using bare `C` (e.g. `C*x`, `C*sin(x)`)
2. **`first_exprs()`** creates `SearchModel` instances, calls `rewrite_coeff()` on each
3. **`grow()`** creates child `SearchModel` from term pools — **does NOT call `rewrite_coeff()`**
4. **`SearchModel.__init__`** does NOT call `rewrite_coeff()` — `jax_model` stays `None`
5. **`manip_model()`** (algebra) calls `rewrite_coeff()` after simplification

### `rewrite_coeff()`

- Walks `self.orig` (the expression before expansion)
- Replaces every bare `C` with `C_0`, `C_1`, etc. sequentially left-to-right
- Builds `self.cs = [C_0, C_1, ...]`
- Builds `self.jax_model` (JAXModel wrapper)
- Called in: `first_exprs()`, `manip_model()`

### `_extract_coeffs_and_vars()` (in `model.py`)

- Used by `JAXModel` when `cs=None`
- Convention: symbols whose name starts with `C_` or `C[` are coefficients
- **Bare `C` is NOT treated as a coefficient** — it falls through to `vars_`

## Problems

### 1. Bare `C` leaks into the search loop

`grow()` creates child models from bare-`C` term pools without calling `rewrite_coeff()`. These models have:
- `jax_model = None` (not built yet)
- `cs = []` (empty)
- `expr` contains bare `C`

When `jax_model` is eventually built (lazy), `_extract_coeffs_and_vars()` puts bare `C` into `vars_` instead of `cs_`.

### 2. Multiple copies of the same `C` get different indices

Before the recent fix, `C*x + C` became `C_0 + C_1*x` (two coefficients) instead of `C_0*x + C_0` (one). The fix uses a dict to track `C -> C_i` mapping.

### 3. Mixed expressions break indexing

When algebra produces `C_0*x + C_1`, both numbered symbols must be in `cs`. The current approach scans the final expression for all `C_i` symbols and builds `cs` from them.

## Design Decisions Needed

1. **When should `rewrite_coeff()` be called?**
   - Always in `SearchModel.__init__`? (simplifies everything downstream)
   - Or only at specific pipeline stages?

2. **Where do term pools get their coefficients?**
   - Should `Grower` produce `C_i`-based terms instead of bare `C`?
   - Or should `SearchModel.__init__` handle bare `C` immediately?

3. **What is the single source of truth for coefficient identity?**
   - Bare `C` means "a new coefficient slot"
   - `C_i` means "a specific coefficient slot"
   - The same bare `C` appearing multiple times in an expression must map to the same `C_i`

4. **Should bare `C` be forbidden after `__init__`?**
   - If `rewrite_coeff()` is always called in `__init__`, bare `C` never appears in a `SearchModel`
   - The grower term pools would need to be constructed differently (or `rewrite_coeff` called on them)

---

## Multi-Kind Coefficients

### Motivation

Current system treats all coefficients as a single flat pool of free-form optimisable parameters (`C_i`). This breaks down when:

- **Systems of equations** — the same coefficient must appear in multiple expressions (e.g. mass and damping in a spring-mass system: `m*a = -k*x - c*v`). The coefficient `k` has one value shared across equations.
- **Physical constants** — gravity `g`, speed of light `c`, Boltzmann constant `k_B` are known values that should not be optimised. They appear in expressions but are fixed.
- **Named parameters** — some coefficients carry semantic meaning and must be tracked by name across the search, not just by index.

### Three Coefficient Kinds

| Kind | Symbol Prefix | Optimisable? | Lifetime | Use Case |
|---|---|---|---|---|
| **Free** | `C_i` | Yes | Per-expression | Standard symbolic regression |
| **Named** | `N_i` | Yes | Cross-expression | Systems of equations, shared parameters |
| **Physical** | `P_i` | No | Global constants | `g`, `c`, `k_B`, `h`, `R` |

#### Free Coefficients (`C_i`)

- Current behavior: all bare `C` in an expression map to `C_0, C_1, ...`
- Optimised independently per expression
- No cross-expression identity
- Example: `C_0*x + C_1` — two independent optimisable parameters

#### Named Coefficients (`N_i`)

- A **global registry** of named coefficient slots
- Same `N_i` in two different expressions means the same optimisable parameter
- Shared across expressions in a system — optimising one equation optimises all shared `N_i`
- Must maintain consistent indexing: if expression A uses `N_0, N_1` and expression B uses `N_0, N_2`, then `N_0` is the same parameter in both
- Example (spring-mass system):
  - Eq 1: `N_0*a + N_1*x + N_2*v = 0` (mass, spring, damping)
  - Eq 2: `a = dv/dt` (kinematic constraint)
  - `N_0` (mass) has one value shared across both equations

#### Physical Constants (`P_i`)

- Fixed values known before search — never optimised
- Pre-defined mapping of `P_i` → numeric value
- Appear in expressions but are substituted at evaluation time
- Example: `P_0 = 9.81` (gravity), `P_1 = 3e8` (speed of light)
- Example expression: `P_0*x` — gravitational force model with fixed gravity

### Symbol Naming

| Kind | Symbol | Example |
|---|---|---|
| Free | `C_i` | `C_0`, `C_1`, `C_2` |
| Named | `N_i` | `N_0`, `N_1`, `N_2` |
| Physical | `P_i` | `P_0`, `P_1`, `P_2` |

Bare placeholder symbols for each kind:
- `C` → free coefficient placeholder
- `N` → named coefficient placeholder
- `P` → physical constant placeholder

### Architecture Changes

#### `SearchModel`

```python
class SearchModel:
    # Current
    cs: list[sympy.Symbol]  # All coefficients in one flat list

    # Proposed — split by kind
    cs_free: list[sympy.Symbol]   # C_i symbols
    cs_named: list[sympy.Symbol]  # N_i symbols
    cs_phys: list[sympy.Symbol]   # P_i symbols

    # Or keep flat but distinguish by prefix
    cs: list[sympy.Symbol]  # C_i, N_i, P_i all in one list
    coeff_kind: dict[sympy.Symbol, Literal["free", "named", "physical"]]
```

#### `rewrite_coeff()` — becomes `rewrite_coefficients()`

```python
def rewrite_coefficients(self) -> None:
    """Replace bare C, N, P symbols with indexed C_i, N_i, P_i."""
    # Three separate mappings:
    #   C -> C_0, C_1, ...   (per-expression, sequential)
    #   N -> N_0, N_1, ...   (global registry, shared across models)
    #   P -> P_0, P_1, ...   (global registry, fixed values)
```

#### Global Coefficient Registries

Named and physical coefficients need global state:

```python
class CoefficientRegistry:
    """Global registry for named and physical coefficients."""

    def get_or_create_named(self, index: int) -> sympy.Symbol:
        """Return N_i, creating it if it doesn't exist."""

    def get_or_create_physical(self, index: int) -> sympy.Symbol:
        """Return P_i, creating it if it doesn't exist."""

    def set_physical_value(self, index: int, value: float):
        """Set the fixed numeric value of a physical constant."""

    def get_physical_value(self, index: int) -> float:
        """Get the fixed numeric value of a physical constant."""

    def get_all_named(self) -> list[sympy.Symbol]:
        """Return all named coefficient symbols."""

    def get_all_physical(self) -> list[tuple[int, float]]:
        """Return all (index, value) pairs for physical constants."""
```

#### `JAXModel` — three coefficient arrays

```python
class JAXModel:
    # Three separate arrays for evaluation
    # coefs: [C_0_val, C_1_val, ...] — optimised by LM
    # named: [N_0_val, N_1_val, ...] — shared across equations, optimised jointly
    # phys:  [P_0_val, P_1_val, ...] — fixed, substituted at build time

    def predict(self, free_coefs, named_coefs, *inputs):
        ...

    # Jacobian only over free + named (physical are constant)
    def jacobian(self, free_coefs, named_coefs, *inputs):
        ...
```

#### `Grower` — separate term pools

```python
class Grower:
    # Current: one pool with bare C
    # with_c_xs1_muls = [C * m for m in xs_pow1]

    # Proposed: three pools
    with_free_xs1_muls = [C * m for m in xs_pow1]   # C_0*x, C_1*x, ...
    with_named_xs1_muls = [N * m for m in xs_pow1]   # N_0*x, N_1*x, ...
    with_phys_xs1_muls = [P * m for m in xs_pow1]    # P_0*x, P_1*x, ...
```

#### Memoization — kind-aware deduplication

Two expressions are the same if they have the same structure AND the same coefficient indices:
- `C_0*x + C_0` ≡ `C_0*x + C_0` (same free coeffs)
- `N_0*x + N_0` ≡ `N_0*x + N_0` (same named coeffs)
- `C_0*x + C_0` ≠ `C_1*x + C_1` (different free coeffs — different optimisation problem)
- `N_0*x + P_0` ≡ `N_0*x + P_0` (same named + physical)

#### Fitness — kind-aware objectives

```python
# Current: minimise size + error
fitness = [size, error]

# Proposed: size only counts free + named (physical don't add complexity)
# Or: penalise differently by kind
# - Free coeffs: +1 size each (standard)
# - Named coeffs: +0 size if already used elsewhere, +1 if new
# - Physical coeffs: +0 size (known, not discovered)
```

### Open Questions

1. **How does the grower know which kind of coefficient to use?**
   - Grammar rules? Policy config? Context from parent expression?
   - E.g., `add_extend` might use `C` (free), while system-level operators use `N` (named)

> we need to grow with all of them, more list comprehensions, probably some more complex ones; this is a highly combinatorial process

2. **How do named coefficients get their indices?**
   - Auto-increment from a global counter?
   - User-provided mapping (e.g. "mass = N_0, spring = N_1")?
   - Inferred from expression structure?

> we should support both a count or list of names

3. **How are physical constants defined?**
   - Hardcoded list (`P_0 = 9.81, P_1 = 3e8, ...`)?
   - User-provided dict (`{"g": 9.81, "c": 3e8}`)?
   - Auto-discovered from expression (e.g. `P("g")` → 9.81)?

> user provided, we should have a list of well known / widely used they can reference as well

4. **How does multi-equation search work?**
   - One `SearchModel` per equation, sharing `N_i` and `P_i`?
   - A new `SystemModel` wrapper that holds multiple `SearchModel`s?
   - How is fitness computed across the system?

> we haven't actually done this before, usually each equation is searched independently
> having a wrapper class sounds like a great idea, let's not worry about system fitness yet
> we just want to be able to represent and work on diff eqs for now

5. **How does the LM optimiser handle mixed kinds?**
   - Free + named: all optimised together in one Jacobian?
   - Physical: substituted before optimisation (removed from Jacobian)?
   - Or optimised in stages (free first, then named)?

> the optimizer probably shouldn't care about the difference between free and named, ya?
> physical should be substituted, these should not contribute to computational processing complexity
> we have to optimize as a whole

6. **Should `P_i` be substituted into the expression at build time?**
   - Replace `P_0*x` with `9.81*x` before JAX compilation?
   - Or keep `P_0` as a symbol and substitute at evaluation time?
   - Substituting early means `filter_has_int_coeff` might reject `9.81*x` — needs fix

> we want it readable for the user (P_i will be a name), but optimized away before we regress other coefficients

7. **Backward compatibility?**
   - All existing expressions use `C_i` — they should keep working unchanged
   - `N_i` and `P_i` are opt-in features

> Yes, make the new ones opt in by including them in the PGE setup from a user point of view

---

## Things to Work On

Noted, but not understood yet:

- need test cases to cover non-linear regressions including cases which do and do not converge to the right answer, eventually we will need a way to deal with local minima (knowing it is generally an unsolvable problem)


### Coefficient Kind Expansion

**Goal**: Extend coefficient system to support three kinds — `C_i` (free), `N_i` (named/shared), `P_i` (physical/fixed) — without breaking existing free-coefficient behavior.

**What needs to happen**:

#### 1. Term pool explosion in `Grower`

Current: one set of pools with bare `C`:
- `with_c_xs1_muls`, `with_c_xs2_muls`, `with_c_xs3_muls`, `with_c_xs4_muls`
- `with_c_linear_funcs`, `with_c_nonlin_funcs`
- `wout_c_linear_funcs`, `wout_c_nonlin_funcs`

Proposed: three pools per kind (3× the list comprehensions, plus cross-kind combinations):
- `with_free_xs*_muls`, `with_named_xs*_muls`, `with_phys_xs*_muls`
- `with_free_linear_funcs`, `with_named_linear_funcs`, `with_phys_linear_funcs`
- etc.

Plus cross-kind pools for mixed expressions:
- `with_free_named_funcs` — `C_0 * N_0 * x`
- `with_free_phys_funcs` — `C_0 * P_0 * x`
- `with_named_phys_funcs` — `N_0 * P_0 * x`
- `with_all_funcs` — `C_0 * N_0 * P_0 * x`

This is highly combinatorial. The grower needs to enumerate all valid combinations of coefficient kinds in each term.

> we need to grow with all of them, more list comprehensions, probably some more complex ones; this is a highly combinatorial process

#### 2. Named coefficient index assignment

Named coefficients need a global registry so the same `N_i` is shared across expressions. Two approaches:

- **Count-based**: user says "I need 5 named coefficients", indices auto-increment (`N_0` through `N_4`)
- **Name-based**: user provides a list like `["mass", "spring", "damping"]`, mapped to `N_0`, `N_1`, `N_2`

> we should support both a count or list of names

#### 3. Physical constant definitions

Physical constants are user-provided fixed values. Need:

- **Built-in registry**: a curated list of well-known constants (`g = 9.81`, `c = 3e8`, `h = 6.626e-34`, `k_B = 1.381e-23`, `R = 8.314`, `N_A = 6.022e23`, `e = 1.602e-19`, `epsilon_0 = 8.854e-12`, `mu_0 = 1.257e-6`, `sigma = 5.670e-8`)
- **User override**: user provides a dict `{"g": 9.81, "my_const": 42.0}`
- Constants get assigned `P_i` indices from the registry

> user provided, we should have a list of well known / widely used they can reference as well

#### 4. Differential equation support (no system fitness yet)

- Each equation is still searched independently — no joint optimisation across equations
- A `SystemModel` wrapper holds multiple `SearchModel`s (one per equation)
- `N_i` and `P_i` are shared across the system via the global registry
- The wrapper makes it possible to represent systems of ODEs/PDEs where coefficients are shared
- System-level fitness is deferred — for now just the ability to represent and work on diff eqs

> we haven't actually done this before, usually each equation is searched independently
> having a wrapper class sounds like a great idea, let's not worry about system fitness yet
> we just want to be able to represent and work on diff eqs for now

#### 5. Optimizer handling

The LM optimizer should not distinguish between free and named coefficients:
- Both `C_i` and `N_i` are optimised together in a single Jacobian
- Both contribute to the cost function identically
- Physical constants (`P_i`) are substituted into the expression before optimization — they become numeric literals, never appear in the Jacobian, contribute zero to computational complexity

> the optimizer probably shouldn't care about the difference between free and named, ya?
> physical should be substituted, these should not contribute to computational processing complexity
> we have to optimize as a whole

#### 6. Physical constant readability

`P_i` must remain readable as named constants (e.g. `g*x`) in output, but be substituted away before optimization:
- Expression stays as `P_0*x` in `SearchModel.expr` and `SearchModel.orig` (for display)
- Before JAX compilation, `P_i` → numeric value (e.g. `9.81*x`)
- `filter_has_int_coeff` needs to allow known physical constant values (don't reject `9.81*x`)

> we want it readable for the user (P_i will be a name), but optimized away before we regress other coefficients

#### 7. Opt-in by design

New coefficient kinds are opt-in from the user's perspective:
- Default PGE setup (bare `usable_vars`, `usable_funcs`) uses only `C_i` — existing behavior unchanged
- User must explicitly configure named/physical coefficients in PGE setup:
  ```python
  pge = PGE(
      usable_vars=["x"],
      named_count=3,           # opt-in: enables N_i
      physical_constants=["g"], # opt-in: enables P_i
  )
  ```
- If `named_count=0` and `physical_constants=None`, the system behaves exactly as it does today

> Yes, make the new ones opt in by including them in the PGE setup from a user point of view

#### Files that will need changes

| File | Change |
|---|---|
| `pge_jax/search_model.py` | `rewrite_coeff()` → `rewrite_coefficients()`, handle C/N/P |
| `pge_jax/model.py` | `_extract_coeffs_and_vars()` → kind-aware extraction, P_i substitution |
| `pge_jax/expand.py` | Triple term pools + cross-kind pools |
| `pge_jax/memoize.py` | Kind-aware hash (same structure + same coefficient indices) |
| `pge_jax/search.py` | PGE setup accepts `named_count`, `physical_constants`, passes to Grower |
| `pge_jax/filters.py` | `filter_has_int_coeff` allows physical constant values |
| `pge_jax/fitness_funcs.py` | Size penalty: free=+1, named=+1 if new/0 if shared, physical=+0 |
| `pge_jax/__init__.py` | Export new types |
| New: `pge_jax/coeff_registry.py` | Global `CoefficientRegistry` for N_i and P_i |
| New: `pge_jax/system_model.py` | `SystemModel` wrapper for multi-equation representation |


### Bare `C` Bug

**Where**: `expand.py:18` — `C = sympy.Symbol("C")`

**How**: `Grower` creates term pools using bare `C`:

| Pool | Lines | Examples |
|---|---|---|
| `with_c_xs1_muls` | 149 | `C*x`, `C/x` |
| `with_c_xs2_muls` | 150 | `C*x*y`, `C*x/x` |
| `with_c_xs3_muls` | 151 | `C*x*y*z` |
| `with_c_xs4_muls` | 152 | `C*x*y*z*w` |
| `wout_c_nonlin_funcs` | 157 | `sin(C*x + C)` |
| `with_c_linear_funcs` | 158 | `C*sin(x)` |
| `with_c_nonlin_funcs` | 159 | `C*sin(C*x + C)` |
| `add_terms` | 188 | `C*x + C` |
| `plus_C_exprs` | 288 | `expr + C` (in `first_exprs()`) |
| `_toggle_plus_C` | 552 | `expr + C` / `expr - C` |

**Why it's a bug**: `grow()` creates child `SearchModel` instances from these bare-`C` term pools but `SearchModel.__init__` does NOT call `rewrite_coeff()`. The child models have:

- `jax_model = None` — not built yet (lazy build happens later)
- `cs = []` — empty, no coefficients tracked
- `expr` contains bare `C`

When `jax_model` is eventually built lazily, `_extract_coeffs_and_vars()` (in `model.py:11-32`) puts bare `C` into `vars_` instead of `cs_` because it only recognises `C_` and `C[` prefixes:

```python
if name.startswith("C_") or name.startswith("C["):
    coeffs.append(sym)
else:
    vars_.append(sym)  # bare C falls through here!
```

This means the JAX model treats bare `C` as an input variable, not an optimisable coefficient. The optimiser will try to fit `C` as a variable alongside `x`, which produces wrong results.

**What should change**:

Option A: Call `rewrite_coeff()` in `SearchModel.__init__`
- Pros: One change, guarantees all models are clean
- Cons: Every `SearchModel` (even intermediate ones before evaluation) calls it

Option B: Call `rewrite_coeff()` in `Grower.grow()` before creating child models
- Pros: More surgical, only where bare `C` is introduced
- Cons: `SearchModel.__init__` still has the latent bug if someone creates a model with bare `C` directly

Option C: Fix `_extract_coeffs_and_vars()` to also recognise bare `C` as a coefficient
- Pros: Defensive — handles bare `C` wherever it appears
- Cons: Doesn't solve the root problem — bare `C` still leaks, and `_extract_coeffs_and_vars` doesn't have a counter to assign unique indices

**Recommended**: Option A + Option C. Call `rewrite_coeff()` in `__init__` as the primary fix, and make `_extract_coeffs_and_vars()` also treat bare `C` as a coefficient (defensive). This way bare `C` can never exist in a `SearchModel`, and even if it does, `JAXModel` won't misclassify it.

**Files to touch**:
- `pge_jax/search_model.py` — call `rewrite_coeff()` in `__init__`
- `pge_jax/model.py` — treat bare `C` as coefficient in `_extract_coeffs_and_vars()`
- `pge_jax/expand.py` — (optional) switch term pools to not use bare `C` at all, instead use placeholder expressions that `rewrite_coeff()` can handle
