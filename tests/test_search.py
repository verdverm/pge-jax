"""Tests for the PGE search loop modules."""

from __future__ import annotations

import sympy
import pytest

from pge_jax.search_model import SearchModel
from pge_jax.filters import (
    default_filters,
    filter_has_big_pow,
    filter_has_coeff_pow,
    filter_has_int_coeff,
    filter_just_C,
    filter_model,
    filter_models,
    filter_no_C,
    filter_too_big,
)
from pge_jax.algebra import do_simp, manip_model
from pge_jax.memoize import Memoizer
from pge_jax.selection import (
    assignCrowdingDist,
    isDominated,
    selNSGA2,
    selSPEA2,
    sortLogNondominated,
    sortNondominated,
)
from pge_jax.fitness_funcs import (
    build_fitness_calc,
    build_fitness_weights,
    build_value_extractor,
)
from pge_jax.expand import Grower, map_names_to_funcs


class TestSearchModel:
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
