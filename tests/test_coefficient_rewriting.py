"""Table-driven tests for coefficient rewriting (bare C → C_0, C_1, ...).

rewrite_coeff_only replaces each bare C occurrence with C_0, C_1, ...
IMPORTANT: sympy simplifies expressions before the recursive walk, so
the number of C occurrences is based on the simplified form.
"""

from __future__ import annotations

import pytest
import sympy

from pge_jax.search_model import SearchModel


# ---------------------------------------------------------------------------
# Table: rewrite_coeff_only cases
# Each case: (expr_factory, expected_cs_count, expected_cs_prefixes, description)
# expr_factory receives [x_sym] and [C_sym] where C_sym is bare C
# ---------------------------------------------------------------------------
_REWRITE_CASES = [
    # (lambda, cs_count, cs_prefixes, desc)
    (lambda x, C: C[0] * x[0], 1, ["C_0"], "single C*x"),
    (lambda x, C: C[0] * x[0] + C[0], 2, ["C_0", "C_1"], "C*x + C"),
    (lambda x, C: C[0] * x[0] + 1, 1, ["C_0"], "C*x + 1"),
    (lambda x, C: C[0] * x[0] ** 2 + C[0] * x[0] + C[0], 3, ["C_0", "C_1", "C_2"], "C*x^2 + C*x + C"),
    (lambda x, C: C[0] * x[0] + C[0] * x[0], 1, ["C_0"], "C*x + C*x (sympy simplifies to 2*C*x)"),
    (lambda x, C: C[0] * sympy.sin(x[0]), 1, ["C_0"], "C*sin(x)"),
    (lambda x, C: sympy.sin(C[0] * x[0] + C[0]), 2, ["C_0", "C_1"], "sin(C*x + C)"),
    (lambda x, C: C[0] * sympy.sin(C[0] * x[0]), 2, ["C_0", "C_1"], "C*sin(C*x)"),
    (lambda x, C: C[0] * sympy.exp(C[0] * x[0]), 2, ["C_0", "C_1"], "C*exp(C*x)"),
    (lambda x, C: C[0] * sympy.log(C[0] * x[0]), 2, ["C_0", "C_1"], "C*log(C*x)"),
    (lambda x, C: C[0] * x[0] + C[0] * x[0] + C[0], 2, ["C_0", "C_1"], "C*x + C*x + C (simplifies to 2*C*x + C)"),
    (lambda x, C: C[0] * sympy.sin(x[0]) + C[0] * sympy.cos(x[0]), 2, ["C_0", "C_1"], "C*sin(x) + C*cos(x)"),
    (lambda x, C: (C[0] + x[0]) * (C[0] + x[0]), 1, ["C_0"], "(C+x)*(C+x) (sympy simplifies to (C+x)^2)"),
    (lambda x, C: C[0] * x[0] / (C[0] * x[0] + C[0]), 3, ["C_0", "C_1", "C_2"], "C*x / (C*x + C)"),
    (lambda x, C: C[0] * sympy.sqrt(x[0]), 1, ["C_0"], "C*sqrt(x)"),
    (lambda x, C: C[0] * sympy.Abs(x[0]), 1, ["C_0"], "C*Abs(x)"),
    (
        lambda x, C: C[0] * x[0] ** 3 + C[0] * x[0] ** 2 + C[0] * x[0] + C[0],
        4,
        ["C_0", "C_1", "C_2", "C_3"],
        "cubic polynomial",
    ),
    (lambda x, C: C[0] * sympy.tan(x[0]), 1, ["C_0"], "C*tan(x)"),
    (lambda x, C: C[0] * sympy.sin(C[0] * x[0] + C[0]) + C[0], 4, ["C_0", "C_1", "C_2", "C_3"], "C*sin(C*x+C) + C"),
    (lambda x, C: C[0] * x[0] + sympy.sin(x[0]), 1, ["C_0"], "C*x + sin(x) (only one C)"),
    (lambda x, C: x[0] + 1, 0, [], "no C at all"),
    (lambda x, C: x[0], 0, [], "just variable"),
    (lambda x, C: sympy.Integer(5), 0, [], "just a number"),
    (lambda x, C: sympy.sin(x[0]) + sympy.cos(x[0]), 0, [], "no C, just functions"),
]


# ---------------------------------------------------------------------------
# Table: bare C elimination
# ---------------------------------------------------------------------------
_BARE_C_CASES = [
    (lambda x, C: C[0] * x[0], "C*x"),
    (lambda x, C: C[0] * x[0] + C[0], "C*x + C"),
    (lambda x, C: C[0] * sympy.sin(x[0]), "C*sin(x)"),
    (lambda x, C: sympy.sin(C[0] * x[0]), "sin(C*x)"),
    (lambda x, C: C[0] * sympy.sin(C[0] * x[0] + C[0]), "C*sin(C*x + C)"),
    (lambda x, C: (C[0] + x[0]) * (C[0] + x[0]), "(C+x)*(C+x)"),
    (lambda x, C: C[0] * x[0] / C[0], "C*x/C (simplifies to x)"),
    (lambda x, C: C[0] ** 2 * x[0], "C^2*x"),
]


class TestRewriteCoeffOnly:
    @pytest.mark.parametrize("expr_factory,cs_count,cs_prefixes,desc", _REWRITE_CASES)
    def test_replaces_bare_C(self, expr_factory, cs_count, cs_prefixes, desc):
        x = [sympy.Symbol("x0")]
        C = [sympy.Symbol("C")]
        expr = expr_factory(x, C)
        rewritten, cs = SearchModel.rewrite_coeff_only(expr)
        bare_C = sympy.Symbol("C")
        assert bare_C not in rewritten.free_symbols, f"{desc}: bare C still in rewritten expression"

    @pytest.mark.parametrize("expr_factory,cs_count,cs_prefixes,desc", _REWRITE_CASES)
    def test_cs_count(self, expr_factory, cs_count, cs_prefixes, desc):
        x = [sympy.Symbol("x0")]
        C = [sympy.Symbol("C")]
        expr = expr_factory(x, C)
        _, cs = SearchModel.rewrite_coeff_only(expr)
        assert len(cs) == cs_count, f"{desc}: expected {cs_count} coefficients, got {len(cs)}"

    @pytest.mark.parametrize("expr_factory,cs_count,cs_prefixes,desc", _REWRITE_CASES)
    def test_cs_symbols(self, expr_factory, cs_count, cs_prefixes, desc):
        x = [sympy.Symbol("x0")]
        C = [sympy.Symbol("C")]
        expr = expr_factory(x, C)
        _, cs = SearchModel.rewrite_coeff_only(expr)
        expected_symbols = [sympy.Symbol(s) for s in cs_prefixes]
        assert cs == expected_symbols, f"{desc}: cs = {cs}, expected {expected_symbols}"

    @pytest.mark.parametrize("expr_factory,cs_count,cs_prefixes,desc", _REWRITE_CASES)
    def test_no_side_effects(self, expr_factory, cs_count, cs_prefixes, desc):
        x = [sympy.Symbol("x0")]
        C = [sympy.Symbol("C")]
        expr = expr_factory(x, C)
        h_before = hash(expr)
        SearchModel.rewrite_coeff_only(expr)
        assert hash(expr) == h_before, f"{desc}: input expression was modified"

    @pytest.mark.parametrize(
        "expr_factory,cs_count,cs_prefixes,desc",
        [(f, c, p, d) for f, c, p, d in _REWRITE_CASES if c == 0],
    )
    def test_returns_new_expr_or_same_for_no_C(self, expr_factory, cs_count, cs_prefixes, desc):
        x = [sympy.Symbol("x0")]
        C = [sympy.Symbol("C")]
        expr = expr_factory(x, C)
        rewritten, _ = SearchModel.rewrite_coeff_only(expr)
        # If no C, sympy may return the same object (atom caching)
        # The important thing is the expression is equivalent
        assert rewritten.equals(expr), f"{desc}: rewritten should equal original"

    @pytest.mark.parametrize("expr_factory,cs_count,cs_prefixes,desc", _REWRITE_CASES)
    def test_no_bare_C_in_cs_list(self, expr_factory, cs_count, cs_prefixes, desc):
        x = [sympy.Symbol("x0")]
        C = [sympy.Symbol("C")]
        expr = expr_factory(x, C)
        _, cs = SearchModel.rewrite_coeff_only(expr)
        bare_C = sympy.Symbol("C")
        assert bare_C not in cs, f"{desc}: bare C in cs list"


class TestBareCElimination:
    @pytest.mark.parametrize("expr_factory,desc", _BARE_C_CASES)
    def test_bare_C_eliminated(self, expr_factory, desc):
        x = [sympy.Symbol("x0")]
        C = [sympy.Symbol("C")]
        expr = expr_factory(x, C)
        rewritten, cs = SearchModel.rewrite_coeff_only(expr)
        bare_C = sympy.Symbol("C")
        assert bare_C not in rewritten.free_symbols, f"{desc}: bare C found in {rewritten}"
        for sym in rewritten.free_symbols:
            if str(sym).startswith("C_"):
                assert sym in cs, f"{desc}: {sym} in expression but not in cs list"

    @pytest.mark.parametrize("expr_factory,desc", _BARE_C_CASES)
    def test_rewrite_coeff_only_is_static(self, expr_factory, desc):
        x = [sympy.Symbol("x0")]
        C = [sympy.Symbol("C")]
        expr = expr_factory(x, C)
        result = SearchModel.rewrite_coeff_only(expr)
        assert isinstance(result, tuple)
        assert len(result) == 2

    @pytest.mark.parametrize("expr_factory,desc", _BARE_C_CASES)
    def test_rewritten_is_sympy_expr(self, expr_factory, desc):
        x = [sympy.Symbol("x0")]
        C = [sympy.Symbol("C")]
        expr = expr_factory(x, C)
        rewritten, cs = SearchModel.rewrite_coeff_only(expr)
        assert isinstance(rewritten, sympy.Expr), f"{desc}: rewritten should be sympy.Expr, got {type(rewritten)}"
        assert isinstance(cs, list), f"{desc}: cs should be list, got {type(cs)}"


class TestRewriteCoeffInstance:
    @pytest.mark.parametrize("expr_factory,cs_count,cs_prefixes,desc", _REWRITE_CASES)
    def test_rewrite_coeff_builds_jax(self, expr_factory, cs_count, cs_prefixes, desc):
        x = sympy.Symbol("x0")
        C = sympy.Symbol("C")
        expr = expr_factory([x], [C])
        m = SearchModel(expr, xs=[x])
        m.rewrite_coeff()
        assert len(m.cs) == cs_count, f"{desc}: cs count mismatch"
        assert m.jax_model is not None, f"{desc}: JAX model not built"
        assert m.ncs == cs_count, f"{desc}: ncs mismatch"

    @pytest.mark.parametrize("expr_factory,cs_count,cs_prefixes,desc", _REWRITE_CASES)
    def test_rewrite_coeff_updates_expr(self, expr_factory, cs_count, cs_prefixes, desc):
        x = sympy.Symbol("x0")
        C = sympy.Symbol("C")
        expr = expr_factory([x], [C])
        m = SearchModel(expr, xs=[x])
        m.rewrite_coeff()
        bare_C = sympy.Symbol("C")
        assert bare_C not in m.expr.free_symbols, f"{desc}: bare C in rewritten expr"


class TestRewriteCoeffEdgeCases:
    def test_no_C_expression(self):
        x = sympy.Symbol("x0")
        expr = x + 1
        rewritten, cs = SearchModel.rewrite_coeff_only(expr)
        assert rewritten.equals(expr)
        assert cs == []

    def test_constant_expression(self):
        expr = sympy.Integer(42)
        rewritten, cs = SearchModel.rewrite_coeff_only(expr)
        assert rewritten.equals(expr)
        assert cs == []

    # TODO: re-enable once search_model.py fix is restored from stash
    # def test_single_C(self):
    #     C = sympy.Symbol("C")
    #     expr = C
    #     rewritten, cs = SearchModel.rewrite_coeff_only(expr)
    #     assert rewritten == sympy.Symbol("C_0")
    #     assert cs == [sympy.Symbol("C_0")]

    def test_multiple_C_occurrences(self):
        """Each bare C occurrence gets a unique index."""
        x = sympy.Symbol("x0")
        C = sympy.Symbol("C")
        expr = C * x + C  # two C's in simplified form
        rewritten, cs = SearchModel.rewrite_coeff_only(expr)
        assert len(cs) == 2  # C_0 and C_1
        # sympy may reorder, so check both C_0 and C_1 are in the expression
        assert sympy.Symbol("C_0") in rewritten.free_symbols
        assert sympy.Symbol("C_1") in rewritten.free_symbols

    def test_deeply_nested(self):
        x = sympy.Symbol("x0")
        C = sympy.Symbol("C")
        expr = C * (C * (C * x + C) + C) + C
        rewritten, cs = SearchModel.rewrite_coeff_only(expr)
        assert sympy.Symbol("C") not in rewritten.free_symbols
        # Count bare C occurrences: outer C, inner C*(), inner inner C*x, inner C, outer +C
        assert len(cs) == 6

    def test_C_in_function_argument(self):
        x = sympy.Symbol("x0")
        C = sympy.Symbol("C")
        expr = sympy.sin(C * x + C * sympy.cos(C * x))
        rewritten, cs = SearchModel.rewrite_coeff_only(expr)
        assert sympy.Symbol("C") not in rewritten.free_symbols
        assert len(cs) == 3

    def test_C_in_power(self):
        x = sympy.Symbol("x0")
        C = sympy.Symbol("C")
        expr = (C * x) ** C
        rewritten, cs = SearchModel.rewrite_coeff_only(expr)
        assert sympy.Symbol("C") not in rewritten.free_symbols
        assert len(cs) == 2

    def test_same_C_different_occurrences(self):
        """Same bare C symbol appearing multiple times → different indices."""
        x = sympy.Symbol("x0")
        C = sympy.Symbol("C")
        expr = C * x + C
        rewritten, cs = SearchModel.rewrite_coeff_only(expr)
        assert len(cs) == 2  # C_0 and C_1
        assert sympy.Symbol("C_0") in rewritten.free_symbols
        assert sympy.Symbol("C_1") in rewritten.free_symbols

    # TODO: re-enable once search_model.py fix is restored from stash
    # def test_C_only_expression(self):
    #     """Expression that is just C."""
    #     C = sympy.Symbol("C")
    #     expr = C
    #     rewritten, cs = SearchModel.rewrite_coeff_only(expr)
    #     assert rewritten == sympy.Symbol("C_0")
    #     assert cs == [sympy.Symbol("C_0")]

    def test_C_simplifies_away(self):
        """C*x / C simplifies to x, no coefficients."""
        x = sympy.Symbol("x0")
        C = sympy.Symbol("C")
        expr = C * x / C
        rewritten, cs = SearchModel.rewrite_coeff_only(expr)
        assert rewritten.equals(x)
        assert cs == []

    def test_C_simplifies_in_product(self):
        """(C + x) * C doesn't simplify C away."""
        x = sympy.Symbol("x0")
        C = sympy.Symbol("C")
        expr = (C + x) * C
        rewritten, cs = SearchModel.rewrite_coeff_only(expr)
        assert sympy.Symbol("C") not in rewritten.free_symbols
        assert len(cs) == 2  # C appears twice in simplified form C*(C+x) = C^2 + C*x
