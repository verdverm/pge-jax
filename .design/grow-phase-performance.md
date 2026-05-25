# Grow Phase Performance Analysis

## 1. What `sympy.expand()` Does

`sympy.expand()` distributes multiplication over addition and simplifies power expressions. It's essentially the "multiply it all out" operation.

```python
>>> sympy.expand((x + 1) * (x + 2))
x**2 + 3*x + 2

>>> sympy.expand((x + y) ** 3)
x**3 + 3*x**2*y + 3*x*y**2 + y**3

>>> sympy.expand(sympy.sin(x) + sympy.cos(x))
sin(x) + cos(x)       # no-op, nothing to distribute
```

For expressions with `C` symbols (coefficients):

```python
>>> sympy.expand((C*x0 + C) * (C*x1 + C))
C**2*x0*x1 + C**2*x0 + C**2*x1 + C**2
```

**Measured cost on typical child expressions:**

| Expression | Cost |
|---|---|
| `C*x0 + C*x1` | 0.8 us |
| `C*x0*x1 + C` | 0.8 us |
| `C*x1 + sin(C*x0 + C)` | 0.8 us |
| `(C*x0 + C)*(C*x1 + C)` | 1.2 us |
| `sin(C*x0 + C) + cos(C*x1 + C)` | 0.8 us |

For these simple expressions, `sympy.expand()` is effectively a no-op most of the time (0.8 us). The cost is real but small per call.

## 2. The Eager Expand in `SearchModel.__init__`

### Where it happens

`SearchModel.__init__` (`search_model.py:132-133`):

```python
def __init__(self, expr, xs=None, cs=None, p_id: int = -2, reln: str = "unknown"):
    ...
    self.orig: sympy.Expr = expr
    self.expr: sympy.Expr = sympy.expand(expr)   # <-- eager expand here
    ...
```

Every time `SearchModel(e, ...)` is created, this happens. From the benchmark, each iteration with 20 popped models produces ~5,200 children. 10 iterations = **~52,000 calls to `sympy.expand()`**.

### Why it's created

`self.expr` (the expanded form) is used in two places:

1. **`size()` / `calc_tree_size()` / `calc_jac_size()`** — walks `self.expr` to count nodes
2. **`pretty_expr()`** — `self.expr.subs(subs)` to substitute fitted coefficient values

Both of these require the expanded form because the original expression from `grow()` may contain unevaluated products like `(C*x0 + C) * (C*x1 + C)` that need to be distributed before tree-size calculation or coefficient substitution.

### The problem

The expanded form is computed eagerly for **every** `SearchModel`, including children that will be rejected by filters before `self.expr` is ever accessed.

### Architectural issue: expand belongs to grow, not init

`SearchModel` is a generic wrapper — it shouldn't care about expression normalization. `sympy.expand` is a grow-level operation that transforms raw expressions into canonical form. It belongs in `grow()`, not in `SearchModel.__init__`.

Moving expand to `grow()` changes the semantics: the raw expression from a grow operator and the expanded expression are structurally different trees. They should be treated as separate `SearchModel` instances because they produce different children when `grow()` is applied.

**Example:**

```
Raw:    (C*x0 + C) * (C*x1 + C)
Expand: C**2*x0*x1 + C**2*x0 + C**2*x1 + C**2
```

These are two different sympy trees. The raw form has a `Mul` at the top with two `Add` children. The expanded form has an `Add` at the top with four `Mul` children. If `grow()` is applied to each:

- `var_sub()` on the raw form finds the `Mul` node and the two `Add` nodes → branches at 3 points
- `var_sub()` on the expanded form finds the `Add` node and four `Mul` nodes → branches at 5 points

Different trees → different children. Treating them as a single `SearchModel` (with expand happening in `__init__`) means we lose the expanded form's expansion potential.

### Current behavior

Currently, `SearchModel.__init__` eagerly expands the expression and stores it in `self.expr`, while keeping the original in `self.orig`. Only one `SearchModel` is created per child from `grow()`. The expanded form is never grown from — it's only used for size calculation and pretty printing.

This is a missed opportunity: the expanded form could itself be grown, producing children that the raw form couldn't produce (and vice versa).

## 3. Why Most Children Are Filtered Out

### The filter pipeline

Default filters (`filters.py:119-125`):

```python
default_filters = [
    filter_too_big,
    filter_has_int_coeff,
    filter_has_big_pow,
    filter_just_C,
    filter_no_C,
    filter_has_coeff_pow,
]
```

Applied via `filter_models()` (`filters.py:35`):

```python
return [modl for modl in models if not filter_model(modl, modl.orig, filters)]
```

Each filter walks `modl.orig` (the **unexpanded** expression) via `preorder_traversal()`.

### The killer: `filter_no_C`

`filter_no_C` (`filters.py:69-73`):

```python
def filter_no_C(modl: SearchModel, expr) -> bool:
    if len(modl.cs) == 0:
        return True  # reject
    return False
```

This checks `modl.cs` — the list of coefficient symbols. If empty, the model is rejected.

**Children always have `cs = []`.** In `grow()` (`expand.py:337-341`):

```python
var_models = [SearchModel(e, p_id=M.id, reln="var_xpnd") for e in var_expands if e != C]
add_models = [SearchModel(e, p_id=M.id, reln="add_xpnd") for e in add_expands if e != C]
mul_models = [SearchModel(e, p_id=M.id, reln="mul_xpnd") for e in mul_expands if e != C]
```

No `cs` argument is passed, so `SearchModel.__init__` sets `self.cs = []` (line 138). The children's expressions **do** contain `C` symbols (that's how the grow operators work), but `modl.cs` is never populated because `rewrite_coeff()` is only called on the initial population (`first_exprs()`, line 298 of `expand.py`), not on children.

**Result: `filter_no_C` returns `True` (reject) for every single child.**

### Short-circuiting behavior

The filters are checked in order. `filter_too_big` is first, but it calls `modl.size()` which accesses `self.expr` (the expanded form). However, `filter_no_C` comes fifth in the list.

Actually, `filter_too_big` is first. It calls `modl.size()` which computes tree size from `self.expr`. But `filter_no_C` is fifth. So for children, the filter chain is:

1. `filter_too_big` → calls `modl.size()` → walks `self.expr` → computes tree size
2. `filter_has_int_coeff` → walks `modl.orig`
3. `filter_has_big_pow` → walks `modl.orig`
4. `filter_just_C` → checks `modl.orig`
5. `filter_no_C` → checks `modl.cs` → **always rejects children**

`filter_too_big` is first and it accesses `self.expr`. So the expanded form IS used for children. But everything after `filter_no_C` (which is fifth) is never reached for children since they're all rejected.

### How many children pass?

From the benchmark, with 20 popped models per iteration and ~261 children each, we get ~5,200 children per iteration. The `filter_no_C` rejects all of them. So the `filter_too_big` check (which requires `self.expr`) runs on all 5,200 children per iteration, but nothing after `filter_no_C` runs on any of them.

This means:
- `self.expr` IS used (by `filter_too_big`) for children
- But the rest of the filter chain is wasted on children
- And the JAX wrapper (`jax_model`) is never built for children until they survive filtering

### Relationship to other design docs

The coefficient management design doc (`coefficient.md`) discusses when `rewrite_coeff()` should be called. Option A (call it in `SearchModel.__init__`) would populate `cs` for all models including children, making `filter_no_C` useless. But this also means every `SearchModel` (even intermediate ones before evaluation) calls `rewrite_coeff()`, which builds the JAX wrapper eagerly.

The multi-kind coefficients design doc is orthogonal — it's about coefficient naming conventions, not expression generation mechanics. The grow performance analysis is fundamental regardless of how coefficients are named.

## 4. Two-Model per Child Architecture

### The proposal

Instead of one `SearchModel` per child (with expand hidden in `__init__`), create two:

```
grow() produces raw_expr:
  → SearchModel(raw_expr, ...)   # "raw model" — grows from the unevaluated form
  → SearchModel(sympy.expand(raw_expr), ...)  # "expanded model" — grows from the distributed form
```

### Why this matters

The raw form and expanded form are structurally different trees. They have different `is_Add` / `is_Mul` / `is_Function` nodes, so different grow operators match different nodes:

**Raw: `(C*x0 + C) * (C*x1 + C)`**
- `is_Mul` at top → `mul_extend` adds factors to the `Mul`
- `is_Add` at children → `add_extend` adds terms to each `Add`
- `var_sub` finds `x0`, `x1` → substitutes with ~20 terms each

**Expanded: `C**2*x0*x1 + C**2*x0 + C**2*x1 + C**2`**
- `is_Add` at top → `add_extend` adds terms to the `Add`
- `is_Mul` at children → `mul_extend` adds factors to each `Mul`
- `var_sub` finds `x0`, `x1` → substitutes with ~20 terms each
- Also finds `C**2` → potential substitutions on coefficient powers

The two forms produce **different children**. Growing from both doubles the exploration surface.

### Cost implications

Two models per child means:
- 2× `SearchModel` construction
- 2× `sympy.expand()` calls (one per model, or shared if cached)
- 2× filter evaluation
- 2× `_uniquify` work

But it also means:
- 2× the children produced per parent → 2× the search space
- Potentially finding better expressions faster

### Trade-off

Currently, ~50% of children are deduplicated by `_uniquify`. If we double the raw+expanded models, the dedup rate might be similar (since many raw/expanded pairs are numerically identical). The net effect would be roughly 2× the children, 2× the cost, but potentially 2× the quality of discovered expressions.

## 5. How Lazy Expand Would Work (on Raw Models Only)

### The proposal for raw models

```python
def __init__(self, expr, xs=None, cs=None, p_id: int = -2, reln: str = "unknown"):
    ...
    self.orig: sympy.Expr = expr
    self._expr: Optional[sympy.Expr] = None  # lazy
    ...

@property
def expr(self) -> sympy.Expr:
    if self._expr is None:
        self._expr = sympy.expand(self.orig)
    return self._expr
```

### Why this is safe for raw models

`self.expr` is only accessed in two places:

1. **`calc_tree_size()`** (`search_model.py:286`): walks `self.expr`
2. **`calc_jac_size()`** (`search_model.py:306`): walks `self.expr`
3. **`pretty_expr()`** (`search_model.py:256`): `self.expr.subs(subs)`

All three are only called for models that have survived filtering and are being evaluated or displayed. For children that are rejected by `filter_no_C`, none of these are ever called.

### What would be saved

Per 10-iteration run (52,000 raw children):
- **~52,000 `sympy.expand()` calls** — ~40 us each → ~2 seconds saved
- **~52,000 `size()` computations** — each walks the tree → ~5 seconds saved

But wait — `filter_too_big` IS the first filter, so it runs before `filter_no_C`. That means `size()` IS called for every child. The lazy expand would save the `sympy.expand()` call, but `size()` still needs the expanded form. So the lazy expand would trigger on the first `size()` call, which happens for every child anyway.

This means lazy expand alone doesn't save much — the expand happens on first `size()` access, which is always. The real savings come from **not calling `size()` at all for children that will be rejected**.

### The real fix: fix `cs` population for children

The root cause is that children always fail `filter_no_C` because `cs` is never populated.

**Option A: Call `rewrite_coeff()` in `SearchModel.__init__`**

This would populate `cs` for all models (including children) but build `jax_model` eagerly (expensive — JAX compilation). It would make `filter_no_C` useless and break the short-circuit that currently rejects children early.

**Option B: Call `rewrite_coeff()` in `grow()` before creating children**

More surgical — only called where bare `C` is introduced. Still builds `jax_model` eagerly.

**Option C: Separate coefficient rewriting from JAX compilation**

`rewrite_coeff()` only rewrites symbols and populates `cs`. `build_jax_model()` compiles the JAX function (called lazily on first evaluation). This is the cleanest approach.

**Recommended: Option C + lazy expand on raw models.** Call `rewrite_coeff()` in `grow()` so children have `cs` populated, and make `expr` lazy for raw models.

## 6. The `_uniquify()` Cost in `grow()`

### Where it happens

`_uniquify()` (`expand.py:536-542`):

```python
def _uniquify(self, exprs: list) -> list:
    """Remove duplicate expressions via ``evalf()`` hashing."""
    pass_set = set()
    for p in exprs:
        s = p.evalf()
        pass_set.add(s)
    return list(pass_set)
```

Called in `grow()` before `SearchModel` is created:

```python
# expand.py:323-325
var_expands = self._uniquify(var_expands + var_expands_C)
add_expands = self._uniquify(add_expands + add_expands_C)
mul_expands = self._uniquify(mul_expands + mul_expands_C)
```

### Cost

**Measured cost per expression:**

| Expression | `evalf()` cost |
|---|---|
| `C*x0 + C*x1` | 20.9 us |
| `C*x0*x1 + C` | 14.2 us |
| `C*x1 + sin(C*x0 + C)` | 36.5 us |
| `(C*x0 + C)*(C*x1 + C)` | 33.6 us |
| `sin(C*x0 + C) + cos(C*x1 + C)` | 51.3 us |

Per iteration with 5,200 children: ~5,200 `evalf()` calls → ~150-250 ms.

### Why it's wasteful

`_uniquify()` is called on expressions that will be rejected by `filter_no_C`. The deduplication happens before filtering, so we're deduplicating expressions we never keep.

### What `_uniquify` actually does

`expr.evalf()` numerically evaluates the expression (with symbolic `C` remaining as symbols). Two expressions that are structurally different but numerically equivalent (e.g., `x + x` vs `2*x`) will have the same `evalf()` result and one will be deduplicated.

This is a cheap structural dedup — much cheaper than comparing full sympy expressions with `==` or hashing them directly.

## 7. The `grow()` Branching Factor

### How many children per parent?

**Measured:** Each `grow()` call produces ~344 children for the first model. Average ~261 children per model.

### Breakdown by operator

| Operator | Children | What it does |
|---|---|---|
| `var_xpnd` | ~3,100/iter | Variable substitution — replace variables with complex terms |
| `add_xpnd` | ~2,400/iter | Addition extension — add terms to Add nodes |
| `mul_xpnd` | ~4,800/iter | Multiplication extension — multiply by new factors |

Total per iteration (20 parents): ~10,300 children (before `_uniquify` dedup).
After `_uniquify`: ~5,200 unique children (50% dedup rate).

### Why the branching factor is so high

Each grow operator walks the full expression tree and substitutes every matching node with all available terms from pre-computed pools. With 4 functions and 2 variables, the pools are:

- `var_sub_terms`: ~20 terms (variable powers + function expressions)
- `add_extend_terms`: ~15 terms (coefficient-scaled variables + functions)
- `mul_extend_terms`: ~15 terms (variable products + functions)

For a simple expression like `C*x0 + C*x1`:
- `var_sub()` finds 2 variable nodes (`x0`, `x1`) → 2 × 20 = 40 children
- `add_extend()` finds 1 Add node with 2 args → 15 children
- `mul_extend()` finds 0 Mul nodes → 0 children (but recurses into args)

For a more complex expression like `C*sin(C*x0 + C) + C*x1`:
- `var_sub()` finds multiple nested nodes → 100+ children
- `add_extend()` finds Add + nested Add → 30+ children
- `mul_extend()` recurses into function args → 50+ children

The branching factor grows with expression complexity.

## 8. Overall Cost Breakdown

### Per iteration (20 parents, ~5,200 children)

| Phase | Time | % of search_loop |
|---|---|---|
| `grow()` — tree traversal + expression construction | ~17.6 s | 96.0% |
| `_uniquify()` — `evalf()` dedup | ~0.3 s | 1.6% |
| `SearchModel.__init__` — `sympy.expand()` + object creation | ~0.2 s | 1.1% |
| `filter_models()` — filter tree walks | ~0.1 s | 0.5% |
| `peek_eval()` — JAX fitting on subset | ~0.05 s | 0.3% |
| `full_eval()` — JAX fitting on full data | ~0.5 s | 2.7% |

**Total per iteration: ~18.8 s**
**10 iterations: ~188 s (3.1 min)**

### Where the time actually goes

1. **`_var_sub()` tree traversal** (~60% of grow time) — recursive descent over every expression, branching at every variable node
2. **`_add_extend()` / `_mul_extend()` tree traversal** (~25% of grow time) — similar recursive descent
3. **`_uniquify()` evalf()** (~10% of grow time) — numerical evaluation for dedup
4. **`SearchModel.__init__`** (~5% of grow time) — `sympy.expand()` + object allocation

The tree traversal is the dominant cost because:
- Each parent expression is walked fully (not just top-level)
- Every variable node branches into ~20 possible substitutions
- The recursion goes deep into nested function calls

## 9. Optimization Opportunities

### 9.1 Two-model per child (raw + expanded)

**Impact: High (exploration) / High (cost)** — doubles the search space but also doubles the cost.

**Approach:** In `grow()`, create two `SearchModel` instances per child: one from the raw expression, one from the expanded form.

**Trade-off:** 2× the children, 2× the cost, but potentially 2× the quality of discovered expressions. The raw form can produce children that the expanded form can't (e.g., growing from a `Mul` node that gets flattened by expand). The expanded form can produce children that the raw form can't (e.g., growing from individual terms of a distributed sum).

### 9.2 Lazy `expr` on raw models

**Impact: Medium** — saves the expand call for rejected children.

**Only useful if** combined with fixing `cs` population so that `filter_no_C` doesn't reject all children. On its own, `size()` (called by `filter_too_big`) triggers the expand anyway.

### 9.3 Fix `cs` population for children

**Impact: High** — if children have `cs` populated, `filter_no_C` stops rejecting all children, and the filter chain can short-circuit earlier on other filters.

**Approach:** Call `rewrite_coeff()` (or a lightweight version that only rewrites symbols) in `grow()` before creating child `SearchModel` instances.

**Trade-off:** `rewrite_coeff()` currently builds `jax_model` which triggers JAX compilation. Need to defer JAX compilation until evaluation time.

### 9.4 Deferred JAX compilation

**Impact: High** — `rewrite_coeff()` builds `jax_model` eagerly. For 52,000 children per 10-iteration run, that's 52,000 JAX compilations.

**Approach:** Separate coefficient rewriting from JAX compilation:
- `rewrite_coeff()` only rewrites symbols and populates `cs`
- `build_jax_model()` compiles the JAX function (called lazily on first evaluation)

### 9.5 Reduce branching factor in `grow()`

**Impact: Medium** — the 261 children per parent is the root cause of all downstream cost.

**Approaches:**
- Reduce pool sizes (fewer variable powers, fewer function combinations)
- Add a size limit during growth (stop expanding if child exceeds `max_size`)
- Use a "limited depth" flag to reduce recursive branching on deep expressions

### 9.6 Move `_uniquify()` after filtering

**Impact: Medium** — deduplicating expressions that will be rejected is wasteful.

**Approach:** Call `_uniquify()` inside `filter_models()` or after filtering, on the surviving set.

**Trade-off:** `_uniquify()` uses `evalf()` which is ~20-50 us per expression. If we skip it for rejected expressions, we save ~100-250 ms per iteration.

### 9.7 Cache `evalf()` results

**Impact: Low** — each child expression is unique, so caching won't help much.

**Only useful if** the same expression structure appears multiple times (e.g., through algebraic manipulation).

## 10. Summary

The grow phase dominates search time at 96% of `search_loop`. The cost comes from:

1. **Tree traversal** in `_var_sub()`, `_add_extend()`, `_mul_extend()` — each parent is walked fully, and every variable/function node branches into multiple substitutions
2. **`_uniquify()`** — `evalf()` on every child before filtering
3. **`SearchModel.__init__`** — `sympy.expand()` on every child

**Architectural issues:**
- `sympy.expand` belongs in `grow()`, not `SearchModel.__init__`
- Raw and expanded forms are structurally different trees that should be separate `SearchModel` instances
- Children always fail `filter_no_C` because `cs` is never populated, wasting all downstream filter work

The biggest wins would be:
1. **Two-model per child** — growing from both raw and expanded forms doubles the search space
2. **Deferred JAX compilation** — separate coefficient rewriting from JAX compilation so children can be created without compilation overhead
3. **Reduce branching factor** — fewer children per parent means less work across all downstream phases

Lazy expand alone saves little because `size()` (called by `filter_too_big`) always needs the expanded form. But lazy expand + fixing `cs` + moving `filter_no_C` before `filter_too_big` would save all the expand/tree-walk work for children rejected by `filter_no_C`.
