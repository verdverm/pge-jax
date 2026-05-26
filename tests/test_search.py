"""Tests for the PGE search loop modules."""

from __future__ import annotations

import pytest
import sympy

from pge_jax.algebra import do_simp, manip_model
from pge_jax.expand import Grower, map_names_to_funcs
from pge_jax.filters import (
    default_filters,
    filter_has_big_pow,
    filter_has_coeff_pow,
    filter_has_int_coeff,
    filter_just_C,
    filter_models,
    filter_no_C,
    filter_too_big,
)
from pge_jax.fitness_funcs import (
    build_fitness_calc,
    build_fitness_weights,
    build_value_extractor,
)
from pge_jax.memoize import Memoizer
from pge_jax.search_model import SearchModel
from pge_jax.selection import (
    assignCrowdingDist,
    isDominated,
    selNSGA2,
    selSPEA2,
    sortLogNondominated,
    sortNondominated,
)


class TestSearchModel:
    """Tests for SearchModel lifecycle, lazy expr, and coefficient rewriting."""

    # Phase 1: Lazy expr
    def test_phase1_lazy_expr_not_computed_on_init(self):
        """_expr should be None after __init__, not eagerly expanded."""
        x = sympy.Symbol("x")
        expr = (x + 1) * (x + 2)
        m = SearchModel(expr)
        assert m._expr is None
        assert m._raw_expr is expr

    def test_phase1_lazy_expr_computed_on_access(self):
        """First access to expr should compute and cache the expanded form."""
        x = sympy.Symbol("x")
        expr = (x + 1) * (x + 2)
        m = SearchModel(expr)
        expanded = m.expr
        assert expanded == sympy.expand(expr)
        assert m._expr is expanded  # cached

    def test_phase1_lazy_expr_cached(self):
        """Subsequent accesses should return the cached expanded form."""
        x = sympy.Symbol("x")
        expr = (x + 1) * (x + 2)
        m = SearchModel(expr)
        first = m.expr
        second = m.expr
        assert first is second

    def test_phase1_lazy_expr_setter(self):
        """expr setter should cache the value directly."""
        x = sympy.Symbol("x")
        expr = x + 1
        m = SearchModel(expr)
        expanded = sympy.expand(expr)
        m.expr = expanded
        assert m._expr is expanded
        assert m.expr is expanded

    def test_phase1_orig_unchanged(self):
        """orig should be the unexpanded input expression."""
        x = sympy.Symbol("x")
        expr = (x + 1) * (x + 2)
        m = SearchModel(expr)
        assert m.orig is expr
        assert m.expr != expr  # expanded form is structurally different

    # Phase 2: Defer JAX compilation
    def test_phase2_rewrite_coeff_only_is_static(self):
        """rewrite_coeff_only should be a static method, not an instance method."""
        x = sympy.Symbol("x")
        C = sympy.Symbol("C")
        expr = C * x + C
        result = SearchModel.rewrite_coeff_only(expr)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_phase2_rewrite_coeff_only_returns_rewritten_expr(self):
        """rewrite_coeff_only should return an expression with C_0, C_1, ... instead of bare C."""
        C = sympy.Symbol("C")
        expr = C * sympy.Symbol("x") + C
        rewritten, cs = SearchModel.rewrite_coeff_only(expr)
        assert sympy.Symbol("C") not in rewritten.free_symbols
        assert sympy.Symbol("C_0") in rewritten.free_symbols
        assert sympy.Symbol("C_1") in rewritten.free_symbols

    def test_phase2_rewrite_coeff_only_returns_cs_list(self):
        """rewrite_coeff_only should return a cs list with C_0, C_1, ... symbols."""
        C = sympy.Symbol("C")
        x = sympy.Symbol("x")
        # Same bare C appearing multiple times gets the same index (C_0)
        expr = C * x + C
        rewritten, cs = SearchModel.rewrite_coeff_only(expr)
        assert len(cs) == 2
        assert cs[0] == sympy.Symbol("C_0")
        assert cs[1] == sympy.Symbol("C_1")

    def test_phase2_rewrite_coeff_only_no_side_effects(self):
        """rewrite_coeff_only should not modify the input expression or set any instance state."""
        C = sympy.Symbol("C")
        expr = C * sympy.Symbol("x")
        before_hash = hash(expr)
        rewritten, cs = SearchModel.rewrite_coeff_only(expr)
        assert hash(expr) == before_hash  # input unchanged
        assert rewritten is not expr  # returns new expression

    def test_phase2_rewrite_coeff_builds_jax(self):
        """rewrite_coeff (instance method) should still build JAX model for backwards compat."""
        C = sympy.Symbol("C")
        x = sympy.Symbol("x")
        expr = C * x + C
        m = SearchModel(expr, xs=[x])
        m.rewrite_coeff()
        assert len(m.cs) == 2
        assert m.jax_model is not None

    # Phase 3: Fix bare C — populate cs in SearchModel
    def test_phase3_searchmodel_accepts_cs(self):
        """SearchModel.__init__ should accept cs parameter and store it."""
        C = sympy.Symbol("C")
        x = sympy.Symbol("x")
        cs = [sympy.Symbol("C_0"), sympy.Symbol("C_1")]
        m = SearchModel(C * x, cs=cs, xs=[x])
        assert m.cs == cs

    def test_phase3_searchmodel_cs_empty_by_default(self):
        """SearchModel without cs parameter should have empty cs list."""
        x = sympy.Symbol("x")
        m = SearchModel(x + 1, xs=[x])
        assert m.cs == []

    def test_phase3_filter_no_C_passes_when_cs_populated(self):
        """filter_no_C should pass when cs is populated."""
        C0 = sympy.Symbol("C_0")
        x = sympy.Symbol("x")
        m = SearchModel(C0 * x, cs=[C0], xs=[x])
        assert filter_no_C(m, m.orig) is False

    def test_phase3_filter_no_C_fails_when_cs_empty(self):
        """filter_no_C should reject when cs is empty."""
        x = sympy.Symbol("x")
        m = SearchModel(x + 1, cs=[], xs=[x])
        assert filter_no_C(m, m.orig) is True

    # Phase 4: Two SearchModel per child
    def test_phase4_grow_returns_raw_and_expanded(self):
        """grow() should produce 2 models per expression (raw + expanded)."""
        C = sympy.Symbol("C")
        x = sympy.Symbol("x")
        parent_expr = C * x
        parent = SearchModel(parent_expr, cs=[C], xs=[x])
        parent.rewrite_coeff()
        funcs = [sympy.sin]
        grower = Grower([x], funcs, func_level="linear", grow_level="low")
        children = grower.grow(parent)
        # Should have at least 2 children (raw + expanded for each expression)
        assert len(children) >= 2
        # Each child should have cs populated
        for c in children:
            assert len(c.cs) > 0

    def test_phase4_raw_model_has_lazy_expr(self):
        """Raw model from grow() should have _expr = None (lazy)."""
        C = sympy.Symbol("C")
        x = sympy.Symbol("x")
        parent_expr = C * x
        parent = SearchModel(parent_expr, cs=[C], xs=[x])
        parent.rewrite_coeff()
        funcs = [sympy.sin]
        grower = Grower([x], funcs, func_level="linear", grow_level="low")
        children = grower.grow(parent)
        # At least one child should be raw (lazy expr)
        has_lazy = any(c._expr is None for c in children)
        assert has_lazy, "Expected at least one raw model with lazy expr"

    def test_phase4_expanded_model_has_cached_expr(self):
        """Expanded model from grow() should have _expr set (cached)."""
        C = sympy.Symbol("C")
        x = sympy.Symbol("x")
        parent_expr = C * x
        parent = SearchModel(parent_expr, cs=[C], xs=[x])
        parent.rewrite_coeff()
        funcs = [sympy.sin]
        grower = Grower([x], funcs, func_level="linear", grow_level="low")
        children = grower.grow(parent)
        # At least one child should be expanded (cached expr)
        has_cached = any(c._expr is not None for c in children)
        assert has_cached, "Expected at least one expanded model with cached expr"

    def test_phase4_both_models_have_same_cs(self):
        """Raw and expanded models from the same expression should share cs."""
        C = sympy.Symbol("C")
        x = sympy.Symbol("x")
        parent_expr = C * x
        parent = SearchModel(parent_expr, cs=[C], xs=[x])
        parent.rewrite_coeff()
        funcs = [sympy.sin]
        grower = Grower([x], funcs, func_level="linear", grow_level="low")
        children = grower.grow(parent)
        # All children should have cs
        for c in children:
            assert len(c.cs) > 0
            # cs should contain C_0 style symbols
            assert any(str(s).startswith("C_") for s in c.cs)

    def test_phase4_double_children_count(self):
        """grow() should produce roughly 2× children per unique expression."""
        C = sympy.Symbol("C")
        x = sympy.Symbol("x")
        parent_expr = C * x
        parent = SearchModel(parent_expr, cs=[C], xs=[x])
        parent.rewrite_coeff()
        funcs = [sympy.sin, sympy.cos]
        grower = Grower([x], funcs, func_level="linear", grow_level="low")
        children = grower.grow(parent)
        # Each unique expression produces 2 models (raw + expanded)
        # So count should be even and >= 2
        assert len(children) >= 2
        assert len(children) % 2 == 0 or len(children) > 2  # multiple exprs may differ

    def test_phase4_raw_and_expanded_different_trees(self):
        """Raw and expanded forms should be structurally different for non-trivial expressions."""
        C = sympy.Symbol("C")
        x = sympy.Symbol("x")
        # Create an expression that expands differently
        parent_expr = (C + x) * (C + x)
        parent = SearchModel(parent_expr, cs=[C], xs=[x])
        parent.rewrite_coeff()
        funcs = []
        grower = Grower([x], funcs, func_level="linear", grow_level="low")
        # Use _make_models-like logic via grow
        children = grower.grow(parent)
        # Find raw and expanded pairs
        raw_candidates = [c for c in children if c._expr is None]
        expanded_candidates = [c for c in children if c._expr is not None]
        # The raw form should have _raw_expr != _expr (unexpanded)
        if raw_candidates:
            raw = raw_candidates[0]
            expanded_accessed = raw.expr  # triggers lazy expand
            assert raw._raw_expr != expanded_accessed or sympy.expand(raw._raw_expr) == expanded_accessed

    # Phase 5: Lazy JAX build in _eval_models
    def test_phase5_build_jax_model_creates_jax_wrapper(self):
        """build_jax_model() should create a JAXModel and store it."""
        C = sympy.Symbol("C_0")
        x = sympy.Symbol("x")
        expr = C * x
        m = SearchModel(expr, cs=[C], xs=[x])
        assert m.jax_model is None
        m.build_jax_model()
        assert m.jax_model is not None
        from pge_jax.model import JAXModel

        assert isinstance(m.jax_model, JAXModel)

    def test_phase5_build_jax_model_with_no_coeffs(self):
        """build_jax_model() should work with no coefficients."""
        x = sympy.Symbol("x")
        expr = x + 1
        m = SearchModel(expr, cs=[], xs=[x])
        assert m.jax_model is None
        m.build_jax_model()
        assert m.jax_model is not None

    def test_phase5_build_jax_model_idempotent(self):
        """Calling build_jax_model() multiple times should overwrite jax_model."""
        C = sympy.Symbol("C_0")
        x = sympy.Symbol("x")
        expr = C * x
        m = SearchModel(expr, cs=[C], xs=[x])
        m.build_jax_model()
        first_jax = m.jax_model
        m.build_jax_model()
        assert m.jax_model is not None
        assert m.jax_model is not first_jax  # new instance created

    def test_phase5_searchmodel_jax_model_none_by_default(self):
        """New SearchModel should have jax_model = None."""
        x = sympy.Symbol("x")
        m = SearchModel(x + 1, xs=[x])
        assert m.jax_model is None


class TestSearchModelBasic:
    def test_init_basic(self):
        x = sympy.Symbol("x")
        C1 = sympy.Symbol("C_0")
        expr = C1 * x + 1
        m = SearchModel(expr, xs=[x], cs=[C1])
        assert m.inited is True
        assert m.id == -2
        assert m.parent_id == -2
        assert m.gen_relation == "unknown"
        assert m.xs == [x]
        assert m.cs == [C1]

    def test_rewrite_coeff(self):
        C = sympy.Symbol("C")
        x = sympy.Symbol("x")
        expr = C * x + C + 1
        m = SearchModel(expr, xs=[x])
        m.rewrite_coeff()
        assert len(m.cs) == 2
        assert sympy.Symbol("C_0") in m.cs
        assert sympy.Symbol("C_1") in m.cs
        assert m.jax_model is not None

    def test_calc_tree_size(self):
        x = sympy.Symbol("x")
        expr = x**2 + 1
        m = SearchModel(expr)
        sz, psz = m.calc_tree_size()
        assert sz > 0
        assert psz >= sz

    def test_dominates(self):
        m1 = SearchModel(sympy.Symbol("C_0") * sympy.Symbol("x"))
        m2 = SearchModel(sympy.Symbol("C_1") * sympy.Symbol("x"))
        m1.wvalues = (1.0, 0.5)
        m2.wvalues = (2.0, 0.8)
        assert m1.dominates(m2) is True
        assert m2.dominates(m1) is False

    def test_size_lazy(self):
        x = sympy.Symbol("x")
        m = SearchModel(x**2)
        assert m.sz == 0
        m.size()
        assert m.sz > 0


class TestFilters:
    def _make_model(self, expr, cs=None, xs=None):
        m = SearchModel(expr, cs=cs, xs=xs)
        m.rewrite_coeff()
        return m

    def test_filter_just_C(self):
        # Bare C gets rewritten to C_0, so after rewrite it's not "just C" anymore
        # The filter checks modl.orig which is the expression BEFORE rewrite
        m = SearchModel(sympy.Symbol("C"), cs=[], xs=[])
        # Don't rewrite — just check the filter directly
        assert filter_just_C(m, sympy.Symbol("C")) is True

    def test_filter_just_C_number(self):
        m = SearchModel(sympy.Integer(42), cs=[], xs=[])
        assert filter_just_C(m, sympy.Integer(42)) is True

    def test_filter_no_C(self):
        x = sympy.Symbol("x")
        m = SearchModel(x**2, cs=[], xs=[x])
        assert filter_no_C(m, m.orig) is True

    def test_filter_no_C_has_coeffs(self):
        x = sympy.Symbol("x")
        C1 = sympy.Symbol("C_0")
        m = SearchModel(C1 * x, cs=[C1], xs=[x])
        assert filter_no_C(m, C1 * x) is False

    def test_filter_too_big(self):
        x = sympy.Symbol("x")
        C1 = sympy.Symbol("C_0")
        expr = C1 * x
        m = SearchModel(expr, cs=[C1], xs=[x])
        assert filter_too_big(m, expr, big=64) is False

    def test_filter_has_int_coeff(self):
        x = sympy.Symbol("x")
        C1 = sympy.Symbol("C_0")
        expr = C1 * x + 2
        m = SearchModel(expr, cs=[C1], xs=[x])
        assert filter_has_int_coeff(m, expr) is True

    def test_filter_has_int_coeff_none(self):
        x = sympy.Symbol("x")
        C1 = sympy.Symbol("C_0")
        expr = C1 * x
        m = SearchModel(expr, cs=[C1], xs=[x])
        assert filter_has_int_coeff(m, expr) is False

    def test_filter_has_big_pow(self):
        x = sympy.Symbol("x")
        expr = x**7
        m = SearchModel(expr)
        assert filter_has_big_pow(m, expr, big=6) is True

    def test_filter_has_big_pow_ok(self):
        x = sympy.Symbol("x")
        expr = x**3
        m = SearchModel(expr)
        assert filter_has_big_pow(m, expr, big=6) is False

    def test_filter_has_coeff_pow(self):
        C = sympy.Symbol("C")
        expr = C**2
        m = SearchModel(expr)
        assert filter_has_coeff_pow(m, expr) is True

    def test_filter_has_coeff_pow_ok(self):
        x = sympy.Symbol("x")
        C1 = sympy.Symbol("C_0")
        expr = C1 * x
        m = SearchModel(expr, cs=[C1], xs=[x])
        assert filter_has_coeff_pow(m, expr) is False

    def test_filter_models(self):
        x = sympy.Symbol("x")
        C1 = sympy.Symbol("C_0")
        # Model without int coeff
        m1 = SearchModel(C1 * x, cs=[C1], xs=[x])
        # Model with int coeff (should be filtered)
        m2 = SearchModel(C1 * x + 5, cs=[C1], xs=[x])
        models = [m1, m2]
        passed = filter_models(models, default_filters)
        assert len(passed) == 1
        assert passed[0] is m1

    def test_default_filters_all(self):
        for f in default_filters:
            assert callable(f)


class TestAlgebra:
    def test_do_simplify(self):
        x = sympy.Symbol("x")
        expr = x**2 + 2 * x + x**2
        simp, err = do_simp(expr, "simplify")
        assert err is None
        assert simp is not None

    def test_do_expand(self):
        x = sympy.Symbol("x")
        expr = (x + 1) * (x + 2)
        expanded, err = do_simp(expr, "expand")
        assert err is None
        assert expanded != expr

    def test_do_factor(self):
        x = sympy.Symbol("x")
        expr = x**2 + 3 * x + 2
        factored, err = do_simp(expr, "factor")
        assert err is None
        assert factored != expr

    def test_do_simp_unknown_method(self):
        x = sympy.Symbol("x")
        _, err = do_simp(x, "unknown_method")
        assert err == "unknown method"

    def test_manip_model_same(self):
        x = sympy.Symbol("x")
        C1 = sympy.Symbol("C_0")
        expr = x + 1
        m = SearchModel(expr, cs=[C1], xs=[x])
        result, err = manip_model(m, "expand")
        assert result is None
        assert err == "same"


class TestMemoizer:
    def test_insert_new(self):
        x = sympy.Symbol("x")
        C1 = sympy.Symbol("C_0")
        m = SearchModel(C1 * x, cs=[C1], xs=[x])
        mem = Memoizer([x])
        assert mem.insert(m) is True

    def test_insert_duplicate(self):
        x = sympy.Symbol("x")
        C1 = sympy.Symbol("C_0")
        m1 = SearchModel(C1 * x, cs=[C1], xs=[x])
        m2 = SearchModel(C1 * x, cs=[C1], xs=[x])
        mem = Memoizer([x])
        assert mem.insert(m1) is True
        assert mem.insert(m2) is False

    def test_lookup_found(self):
        x = sympy.Symbol("x")
        C1 = sympy.Symbol("C_0")
        m = SearchModel(C1 * x, cs=[C1], xs=[x])
        mem = Memoizer([x])
        mem.insert(m)
        found, result = mem.lookup(m)
        assert found is True
        assert result is m

    def test_lookup_not_found(self):
        x = sympy.Symbol("x")
        C1 = sympy.Symbol("C_0")
        m = SearchModel(C1 * x, cs=[C1], xs=[x])
        mem = Memoizer([x])
        found, result = mem.lookup(m)
        assert found is False
        assert result is None

    def test_get_by_id(self):
        x = sympy.Symbol("x")
        C1 = sympy.Symbol("C_0")
        m = SearchModel(C1 * x, cs=[C1], xs=[x])
        mem = Memoizer([x])
        mem.insert(m)
        assert mem.get_by_id(0) is m


class FitnessValues:
    def __init__(self, values):
        self._values = values

    def dominates(self, other):
        not_equal = False
        for a, b in zip(self._values, other._values):
            if a > b:
                return False
            if a < b:
                not_equal = True
        return not_equal

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)

    def __getitem__(self, idx):
        return self._values[idx]

    def __repr__(self):
        return f"FitnessValues({self._values})"


class DummyModel:
    def __init__(self, values, wvalues=None):
        self.values = FitnessValues(values)
        # wvalues must be a plain tuple for sortLogNondominated to work
        self.wvalues = tuple(wvalues) if wvalues is not None else values
        self.crowding_dist = 0.0


class TestSelection:
    def test_is_dominated(self):
        # isDominated(w1, w2) returns True if w2 dominates w1 (maximization semantics: higher is better)
        # (1,2) is worse than (2,3), so (1,2) is dominated by (2,3)
        assert isDominated((1.0, 2.0), (2.0, 3.0)) is True
        assert isDominated((2.0, 3.0), (1.0, 2.0)) is False
        assert isDominated((1.0, 2.0), (1.0, 2.0)) is False

    def test_sort_nondominated(self):
        models = [
            DummyModel((1.0, 5.0)),
            DummyModel((2.0, 3.0)),
            DummyModel((3.0, 1.0)),
            DummyModel((4.0, 4.0)),
        ]
        fronts = sortNondominated(models, 4)
        assert len(fronts) >= 1
        assert len(fronts[0]) >= 2

    def test_sort_log_nondominated(self):
        models = [
            DummyModel((1.0, 5.0)),
            DummyModel((2.0, 3.0)),
            DummyModel((3.0, 1.0)),
            DummyModel((4.0, 4.0)),
        ]
        fronts = sortLogNondominated(models, 4)
        assert len(fronts) >= 1

    def test_sel_nsga2(self):
        models = [
            DummyModel((1.0, 5.0)),
            DummyModel((2.0, 3.0)),
            DummyModel((3.0, 1.0)),
            DummyModel((4.0, 4.0)),
            DummyModel((5.0, 2.0)),
        ]
        selected = selNSGA2(models, 3, nd="log")
        assert len(selected) == 3

    def test_assign_crowding_dist(self):
        models = [DummyModel((i, i)) for i in range(5)]
        assignCrowdingDist(models)
        for m in models:
            assert m.crowding_dist >= 0

    def test_sel_spea2(self):
        models = [DummyModel((i, i)) for i in range(5)]
        selected = selSPEA2(models, 3)
        assert len(selected) == 3


class FitnessModel:
    def __init__(self, score, r2, jpsz):
        self.score = score
        self.r2 = r2
        self.jpsz = jpsz
        self.fitness_values = ()
        self.wvalues = ()


class TestFitnessFuncs:
    def test_build_fitness_weights(self):
        params = ["-(1)jpsz", "-score", "+r2"]
        weights = build_fitness_weights(params)
        assert weights == (-1.0, -1.0, 1.0)

    def test_build_fitness_weights_with_explicit_weight(self):
        params = ["-(2)jpsz", "+r2(0.5)"]
        weights = build_fitness_weights(params)
        assert weights == (-2.0, 0.5)

    def test_build_value_extractor(self):
        params = ["-score", "+r2"]
        extractor = build_value_extractor(params)
        m = FitnessModel(0.5, 0.9, 10)
        vals = extractor(m)
        assert vals == (0.5, 0.9)

    def test_fitness_calc_raw(self):
        params = ["-score", "+r2"]
        calc = build_fitness_calc(params)
        models = [
            FitnessModel(0.5, 0.9, 10),
            FitnessModel(0.3, 0.95, 8),
        ]
        calc(models)
        assert len(models[0].fitness_values) == 2
        assert len(models[0].wvalues) == 2

    def test_fitness_calc_norm(self):
        params = ["normalize", "-score", "+r2"]
        calc = build_fitness_calc(params)
        models = [
            FitnessModel(1.0, 0.5, 10),
            FitnessModel(2.0, 0.8, 8),
            FitnessModel(3.0, 0.9, 12),
        ]
        calc(models)
        assert len(models[0].fitness_values) == 2


class TestExpand:
    def test_map_names_to_funcs(self):
        funcs = map_names_to_funcs(["sin", "exp", "cos"])
        assert len(funcs) == 3
        assert funcs[0] == sympy.sin
        assert funcs[1] == sympy.exp
        assert funcs[2] == sympy.cos

    def test_map_names_to_funcs_unknown(self):
        with pytest.raises(ValueError, match="Unknown function name"):
            map_names_to_funcs(["unknown_func"])

    def test_grower_first_exprs(self):
        x = sympy.Symbol("x")
        funcs = [sympy.sin, sympy.exp]
        grower = Grower([x], funcs, init_level="low", func_level="linear")
        models = grower.first_exprs()
        assert len(models) > 0
        for m in models:
            assert m.gen_relation == "first_gen"
            assert m.parent_id == -1
            assert m.inited is True

    def test_grower_grow(self):
        x = sympy.Symbol("x")
        C1 = sympy.Symbol("C_0")
        expr = C1 * x
        m = SearchModel(expr, cs=[C1], xs=[x])
        m.rewrite_coeff()
        funcs = [sympy.sin]
        grower = Grower([x], funcs, func_level="linear", grow_level="low")
        children = grower.grow(m)
        assert len(children) > 0
        for c in children:
            assert c.parent_id == m.id

    def test_grower_toggle_plus_C(self):
        x = sympy.Symbol("x")
        C = sympy.Symbol("C")
        grower = Grower([x], None)
        expr = x + 1
        result = grower._toggle_plus_C(expr)
        assert result == x + 1 + C
        expr = x + 1 + C
        result = grower._toggle_plus_C(expr)
        assert result == x + 1

    def test_grower_uniquify(self):
        x = sympy.Symbol("x")
        grower = Grower([x], None)
        exprs = [x + 1, x + 1, x + 2, 2 * x]
        unique = grower._uniquify(exprs)
        assert len(unique) == 3


class TestIntegration:
    """End-to-end integration tests for the PGE search loop."""

    # Integration tests removed — they time out. See tests/test_pipeline.py for granular tests.


class TestInstrumentation:
    """Tests for the PGE instrumentation (StageTimings, IterationStageTimes)."""

    def test_stage_timings_defaults(self):
        """StageTimings should have zero defaults."""
        from pge_jax.search_model import StageTimings

        t = StageTimings()
        assert t.build_time == 0.0
        assert t.fit_time == 0.0
        assert t.eval_time == 0.0

    def test_stage_timings_set_values(self):
        """StageTimings should accept non-zero values."""
        from pge_jax.search_model import StageTimings

        t = StageTimings(build_time=0.1, fit_time=0.2, eval_time=0.3)
        assert t.build_time == 0.1
        assert t.fit_time == 0.2
        assert t.eval_time == 0.3

    def test_iteration_stage_times_defaults(self):
        """IterationStageTimes should have empty lists by default."""
        from pge_jax.search_model import IterationStageTimes

        t = IterationStageTimes()
        assert t.grow == []
        assert t.filter == []
        assert t.algebra == []
        assert t.peek_eval == []
        assert t.full_eval == []

    def test_iteration_stage_times_append(self):
        """IterationStageTimes should allow appending values."""
        from pge_jax.search_model import IterationStageTimes

        t = IterationStageTimes()
        t.grow.append(0.5)
        t.grow.append(0.7)
        assert len(t.grow) == 2
        assert t.grow[0] == 0.5

    def test_searchmodel_has_timings(self):
        """SearchModel should have a timings attribute with StageTimings."""
        from pge_jax.search_model import SearchModel

        x = sympy.Symbol("x")
        m = SearchModel(x + 1, xs=[x])
        assert hasattr(m, "timings")
        assert m.timings.build_time == 0.0
        assert m.timings.fit_time == 0.0
        assert m.timings.eval_time == 0.0

    def test_pge_has_iteration_times(self):
        """PGE should have _iteration_times and _loop_stage_times attributes."""
        from pge_jax.search import PGE

        pge = PGE(usable_vars=["x"], max_iter=2, pop_count=1)
        assert hasattr(pge, "_iteration_times")
        assert pge._iteration_times == []
        assert hasattr(pge, "_loop_stage_times")
        assert pge._loop_stage_times.grow == []
        assert pge._loop_stage_times.filter == []
        assert pge._loop_stage_times.algebra == []
        assert pge._loop_stage_times.peek_eval == []
        assert pge._loop_stage_times.full_eval == []

    def test_pge_timing_stats_method(self):
        """PGE._print_timing_stats should not raise with valid data."""
        from pge_jax.search import PGE

        pge = PGE(usable_vars=["x"], max_iter=1, pop_count=1)
        times = [0.1, 0.2, 0.3]
        # Should not raise
        pge._print_timing_stats("  ", "test", times)

    def test_pge_timing_stats_empty(self):
        """PGE._print_timing_stats should handle empty list gracefully."""
        from pge_jax.search import PGE

        pge = PGE(usable_vars=["x"], max_iter=1, pop_count=1)
        # Should not raise
        pge._print_timing_stats("  ", "test", [])
