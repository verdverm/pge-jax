"""Table-driven tests validating memoization deduplicates expressions correctly."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="paired with stashed search.py code")

import pytest
import sympy

from pge_jax.search_model import SearchModel
from pge_jax.search import PGE


def _make_pge(algebra_methods=None):
    return PGE(
        usable_vars=["x0"],
        usable_funcs=["sin"],
        algebra_methods=algebra_methods or ["expand", "factor"],
        max_iter=1,
        pop_count=1,
        random_seed=42,
    )


# ---------------------------------------------------------------------------
# Table: expressions that should be deduplicated (same sympy structure)
# ---------------------------------------------------------------------------
_DEDUP_CASES = [
    (lambda x, C: C[0] * x[0], 1, "simple C*x"),
    (lambda x, C: C[0] * x[0] + C[1], 2, "C*x + C"),
    (lambda x, C: C[0] * x[0] ** 2, 1, "C*x^2"),
    (lambda x, C: C[0] * x[0] + C[1] * x[0] ** 2, 2, "C*x + C*x^2"),
    (lambda x, C: sympy.sin(C[0] * x[0]), 1, "sin(C*x)"),
    (lambda x, C: C[0] * sympy.sin(x[0]), 1, "C*sin(x)"),
    (lambda x, C: C[0] * sympy.sin(x[0]) + C[1], 2, "C*sin(x) + C"),
    (lambda x, C: C[0] * x[0] / (C[1] * x[0] + C[2]), 3, "C*x / (C*x + C)"),
    (lambda x, C: sympy.exp(C[0] * x[0]), 1, "exp(C*x)"),
    (lambda x, C: C[0] * x[0] ** 3 + C[1] * x[0] ** 2 + C[2] * x[0] + C[3], 4, "cubic polynomial"),
]


# ---------------------------------------------------------------------------
# Table: expressions that should NOT be deduplicated (different sympy structure)
# ---------------------------------------------------------------------------
_NOT_DUP_CASES = [
    (lambda x, C: C[0] * x[0], lambda x, C: C[0] * x[0] + C[1], 1, 2, "C*x vs C*x + C"),
    (lambda x, C: C[0] * x[0], lambda x, C: C[0] * x[0] ** 2, 1, 1, "C*x vs C*x^2"),
    (lambda x, C: C[0] * sympy.sin(x[0]), lambda x, C: C[0] * sympy.cos(x[0]), 1, 1, "C*sin(x) vs C*cos(x)"),
    (lambda x, C: C[0] * x[0] + C[1], lambda x, C: C[0] * x[0] + C[1] + C[2], 2, 3, "C*x+C vs C*x+C+C"),
    (lambda x, C: C[0] * x[0], lambda x, C: sympy.sin(C[0] * x[0]), 1, 1, "C*x vs sin(C*x)"),
    (
        lambda x, C: C[0] * sympy.sin(x[0]) + C[1],
        lambda x, C: C[0] * sympy.cos(x[0]) + C[1],
        2,
        2,
        "C*sin(x)+C vs C*cos(x)+C",
    ),
]


class TestMemoizationDedup:
    @pytest.mark.parametrize("expr_factory,cs_count,desc", _DEDUP_CASES)
    def test_identical_expressions_deduped(self, expr_factory, cs_count, desc):
        x = [sympy.Symbol("x0")]
        C = [sympy.Symbol(f"C_{i}") for i in range(cs_count)]
        expr = expr_factory(x, C)
        pge = _make_pge()
        m1 = SearchModel(expr, cs=list(C), xs=list(x))
        m2 = SearchModel(expr, cs=list(C), xs=list(x))
        assert m1.orig == m2.orig
        assert hash(m1.orig) == hash(m2.orig)
        result = pge._run_loop_pipeline([m1, m2])
        assert len(result) == 1, f"{desc}: expected 1 result, got {len(result)}"
        assert result[0] is m1, f"{desc}: first model should pass through"
        assert m2 not in result, f"{desc}: duplicate should be rejected"

    @pytest.mark.parametrize("expr_factory,cs_count,desc", _DEDUP_CASES)
    def test_dedup_assigns_id_only_to_first(self, expr_factory, cs_count, desc):
        x = [sympy.Symbol("x0")]
        C = [sympy.Symbol(f"C_{i}") for i in range(cs_count)]
        expr = expr_factory(x, C)
        pge = _make_pge()
        m1 = SearchModel(expr, cs=list(C), xs=list(x))
        m2 = SearchModel(expr, cs=list(C), xs=list(x))
        pge._run_loop_pipeline([m1, m2])
        assert m1.id == 0, f"{desc}: first model should get ID 0"
        assert m2.id == -2, f"{desc}: duplicate should not get ID"

    @pytest.mark.parametrize("expr_factory,cs_count,desc", _DEDUP_CASES)
    def test_dedup_hmap_size(self, expr_factory, cs_count, desc):
        x = [sympy.Symbol("x0")]
        C = [sympy.Symbol(f"C_{i}") for i in range(cs_count)]
        expr = expr_factory(x, C)
        pge = _make_pge()
        m1 = SearchModel(expr, cs=list(C), xs=list(x))
        m2 = SearchModel(expr, cs=list(C), xs=list(x))
        pge._run_loop_pipeline([m1, m2])
        assert len(pge.hmap) == 1, f"{desc}: hmap should have 1 entry"
        assert len(pge.models) == 1, f"{desc}: models should have 1 entry"

    @pytest.mark.parametrize("expr_factory,cs_count,desc", _DEDUP_CASES)
    def test_three_identical_deduped(self, expr_factory, cs_count, desc):
        x = [sympy.Symbol("x0")]
        C = [sympy.Symbol(f"C_{i}") for i in range(cs_count)]
        expr = expr_factory(x, C)
        pge = _make_pge()
        m1 = SearchModel(expr, cs=list(C), xs=list(x))
        m2 = SearchModel(expr, cs=list(C), xs=list(x))
        m3 = SearchModel(expr, cs=list(C), xs=list(x))
        result = pge._run_loop_pipeline([m1, m2, m3])
        assert len(result) == 1, f"{desc}: expected 1 result from 3 identical"
        assert result[0] is m1, f"{desc}: first model should be the one in result"


class TestMemoizationNotDedup:
    @pytest.mark.parametrize("expr_a,expr_b,cs_a,cs_b,desc", _NOT_DUP_CASES)
    def test_different_expressions_both_pass(self, expr_a, expr_b, cs_a, cs_b, desc):
        x = [sympy.Symbol("x0")]
        C_a = [sympy.Symbol(f"C_{i}") for i in range(cs_a)]
        C_b = [sympy.Symbol(f"C_{i}") for i in range(cs_b)]
        pge = _make_pge()
        m1 = SearchModel(expr_a(x, C_a), cs=list(C_a), xs=list(x))
        m2 = SearchModel(expr_b(x, C_b), cs=list(C_b), xs=list(x))
        assert m1.orig != m2.orig, f"{desc}: expressions should be different"
        result = pge._run_loop_pipeline([m1, m2])
        assert len(result) == 2, f"{desc}: expected 2 results, got {len(result)}"
        assert m1 in result, f"{desc}: first model should be in result"
        assert m2 in result, f"{desc}: second model should be in result"

    @pytest.mark.parametrize("expr_a,expr_b,cs_a,cs_b,desc", _NOT_DUP_CASES)
    def test_different_expressions_different_ids(self, expr_a, expr_b, cs_a, cs_b, desc):
        x = [sympy.Symbol("x0")]
        C_a = [sympy.Symbol(f"C_{i}") for i in range(cs_a)]
        C_b = [sympy.Symbol(f"C_{i}") for i in range(cs_b)]
        pge = _make_pge()
        m1 = SearchModel(expr_a(x, C_a), cs=list(C_a), xs=list(x))
        m2 = SearchModel(expr_b(x, C_b), cs=list(C_b), xs=list(x))
        pge._run_loop_pipeline([m1, m2])
        assert m1.id == 0, f"{desc}: first model should get ID 0"
        assert m2.id == 1, f"{desc}: second model should get ID 1"


class TestMemoizationEdgeCases:
    def test_empty_input(self):
        pge = _make_pge()
        result = pge._run_loop_pipeline([])
        assert result == []
        assert len(pge.hmap) == 0

    def test_single_model_no_dedup_needed(self):
        x = [sympy.Symbol("x0")]
        C = [sympy.Symbol("C_0")]
        pge = _make_pge()
        m = SearchModel(C[0] * x[0], cs=C, xs=x)
        result = pge._run_loop_pipeline([m])
        assert len(result) == 1
        assert m.id == 0

    def test_mixed_dedup_and_unique(self):
        x = [sympy.Symbol("x0")]
        C0 = [sympy.Symbol("C_0")]
        C1 = [sympy.Symbol("C_1")]
        pge = _make_pge()
        m1 = SearchModel(C0[0] * x[0], cs=C0, xs=x)
        m2 = SearchModel(C1[0] * x[0] ** 2, cs=C1, xs=x)
        m3 = SearchModel(C0[0] * x[0], cs=C0, xs=x)  # dup of m1
        result = pge._run_loop_pipeline([m1, m2, m3])
        assert len(result) == 2, "2 unique expressions should pass through"
        assert m1 in result
        assert m2 in result
        assert m3 not in result

    def test_filter_then_memoize(self):
        x = [sympy.Symbol("x0")]
        C0 = [sympy.Symbol("C_0")]
        pge = _make_pge()
        m_bad = SearchModel(C0[0] * x[0] + 5, cs=C0, xs=x)
        m_good = SearchModel(C0[0] * x[0], cs=C0, xs=x)
        result = pge._run_loop_pipeline([m_bad, m_good])
        assert m_bad not in result
        assert m_good in result
        assert len(pge.hmap) == 1

    def test_algebra_can_produce_duplicate(self):
        x = [sympy.Symbol("x0")]
        C0 = [sympy.Symbol("C_0")]
        C1 = [sympy.Symbol("C_1")]
        pge = _make_pge()
        m1 = SearchModel(C0[0] * x[0], cs=C0, xs=x)
        m2 = SearchModel(C1[0] * x[0] ** 2, cs=C1, xs=x)
        result = pge._run_loop_pipeline([m1, m2])
        assert len(result) >= 2

    def test_no_algebra_methods(self):
        pge = _make_pge(algebra_methods=[])
        x = [sympy.Symbol("x0")]
        C0 = [sympy.Symbol("C_0")]
        m1 = SearchModel(C0[0] * x[0], cs=C0, xs=x)
        m2 = SearchModel(C0[0] * x[0], cs=C0, xs=x)
        result = pge._run_loop_pipeline([m1, m2])
        assert len(result) == 1

    def test_complex_expression_dedup(self):
        x = [sympy.Symbol("x0")]
        C = [sympy.Symbol(f"C_{i}") for i in range(3)]
        expr = C[0] * x[0] / (C[1] * x[0] + C[2])
        pge = _make_pge()
        m1 = SearchModel(expr, cs=list(C), xs=list(x))
        m2 = SearchModel(expr, cs=list(C), xs=list(x))
        result = pge._run_loop_pipeline([m1, m2])
        assert len(result) == 1
        assert result[0] is m1

    def test_expression_order_preserved(self):
        x = [sympy.Symbol("x0")]
        C0 = [sympy.Symbol("C_0")]
        C1 = [sympy.Symbol("C_1")]
        pge = _make_pge()
        m1 = SearchModel(C0[0] * x[0], cs=C0, xs=x)
        m2 = SearchModel(C1[0] * x[0] ** 2, cs=C1, xs=x)
        m3 = SearchModel(C0[0] * x[0], cs=C0, xs=x)  # dup of m1
        result = pge._run_loop_pipeline([m1, m2, m3])
        assert len(result) == 2
        assert result[0] is m1
        assert result[1] is m2
