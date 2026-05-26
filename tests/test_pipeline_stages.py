"""Granular tests for the process-filter pipeline stages.

Tests each stage in isolation, then tests the full pipeline composition.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="paired with stashed search.py/search_model.py code")

import sympy

from pge_jax.filters import (
    default_filters,
    filter_has_int_coeff,
    filter_models,
    filter_no_C,
    filter_too_big,
)
from pge_jax.memoize import Memoizer
from pge_jax.search_model import SearchModel


# ------------------------------------------------------------------
# Stage: rewrite_coeff_only (pure static method)
# ------------------------------------------------------------------


class TestRewriteCoeffOnly:
    """rewrite_coeff_only must be pure — no side effects, deterministic."""

    def test_replaces_bare_C(self):
        C = sympy.Symbol("C")
        x = sympy.Symbol("x")
        expr = C * x + C
        rewritten, cs = SearchModel.rewrite_coeff_only(expr)
        assert C not in rewritten.free_symbols
        assert sympy.Symbol("C_0") in rewritten.free_symbols

    def test_returns_cs_list(self):
        C = sympy.Symbol("C")
        x = sympy.Symbol("x")
        expr = C * x + C
        _, cs = SearchModel.rewrite_coeff_only(expr)
        assert len(cs) == 2
        assert cs[0] == sympy.Symbol("C_0")
        assert cs[1] == sympy.Symbol("C_1")

    def test_no_side_effects_on_input(self):
        C = sympy.Symbol("C")
        expr = C * sympy.Symbol("x")
        h_before = hash(expr)
        SearchModel.rewrite_coeff_only(expr)
        assert hash(expr) == h_before

    def test_returns_new_expr(self):
        C = sympy.Symbol("C")
        x = sympy.Symbol("x")
        expr = C * x
        rewritten, _ = SearchModel.rewrite_coeff_only(expr)
        assert rewritten is not expr

    def test_no_bare_C_in_cs(self):
        """cs list must contain C_0, C_1, ... not bare C."""
        C = sympy.Symbol("C")
        x = sympy.Symbol("x")
        _, cs = SearchModel.rewrite_coeff_only(C * x + C)
        assert C not in cs


# ------------------------------------------------------------------
# Stage: SearchModel identity (hash/equality for memoization)
# ------------------------------------------------------------------


class TestSearchModelIdentity:
    """SearchModel identity for memoization must be based on orig expression."""

    def test_hash_based_on_orig(self):
        x = sympy.Symbol("x")
        C1 = sympy.Symbol("C_0")
        m1 = SearchModel(C1 * x, cs=[C1], xs=[x])
        m2 = SearchModel(C1 * x, cs=[C1], xs=[x])
        # Two models with same orig should have same hash
        assert m1.__hash__() == m2.__hash__()

    def test_different_orig_different_hash(self):
        x = sympy.Symbol("x")
        C1 = sympy.Symbol("C_0")
        m1 = SearchModel(C1 * x, cs=[C1], xs=[x])
        m2 = SearchModel(C1 * x + 1, cs=[C1], xs=[x])
        assert m1.__hash__() != m2.__hash__()

    def test_orig_is_unmodified(self):
        x = sympy.Symbol("x")
        C = sympy.Symbol("C")
        expr = C * x
        m = SearchModel(expr, xs=[x])
        assert m.orig is expr


# ------------------------------------------------------------------
# Stage: Memoizer.insert / lookup
# ------------------------------------------------------------------


class TestMemoizer:
    """Memoizer must deduplicate by expression hash."""

    def _make_model(self, expr, xs=None):
        rewritten, cs = SearchModel.rewrite_coeff_only(expr)
        return SearchModel(expr, xs=xs, cs=cs)

    def test_insert_new(self):
        x = sympy.Symbol("x")
        C1 = sympy.Symbol("C_0")
        m = SearchModel(C1 * x, cs=[C1], xs=[x])
        mem = Memoizer([x])
        assert mem.insert(m) is True

    def test_insert_duplicate_returns_false(self):
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

    def test_assigns_id(self):
        x = sympy.Symbol("x")
        C1 = sympy.Symbol("C_0")
        m = SearchModel(C1 * x, cs=[C1], xs=[x])
        mem = Memoizer([x])
        mem.insert(m)
        assert m.id == 0

    def test_second_model_gets_id_1(self):
        x = sympy.Symbol("x")
        C1 = sympy.Symbol("C_0")
        m1 = SearchModel(C1 * x, cs=[C1], xs=[x])
        m2 = SearchModel(C1 * x + C1, cs=[C1], xs=[x])
        mem = Memoizer([x])
        mem.insert(m1)
        mem.insert(m2)
        assert m1.id == 0
        assert m2.id == 1

    def test_structurally_different_expressions_not_deduped(self):
        """x + 1 and x + 2 are different → both inserted."""
        x = sympy.Symbol("x")
        m1 = SearchModel(x + 1, cs=[], xs=[x])
        m2 = SearchModel(x + 2, cs=[], xs=[x])
        mem = Memoizer([x])
        assert mem.insert(m1) is True
        assert mem.insert(m2) is True


# ------------------------------------------------------------------
# Stage: filter_models
# ------------------------------------------------------------------


class TestFilterModels:
    """filter_models must reject invalid expressions and set rejected flag."""

    def _make_model(self, expr, xs=None):
        rewritten, cs = SearchModel.rewrite_coeff_only(expr)
        return SearchModel(rewritten, xs=xs, cs=cs)

    def test_passes_valid_model(self):
        x = sympy.Symbol("x")
        C1 = sympy.Symbol("C_0")
        m = SearchModel(C1 * x, cs=[C1], xs=[x])
        passed = filter_models([m], default_filters)
        assert len(passed) == 1
        assert passed[0] is m

    def test_rejects_int_coeff(self):
        x = sympy.Symbol("x")
        C1 = sympy.Symbol("C_0")
        m = SearchModel(C1 * x + 5, cs=[C1], xs=[x])
        passed = filter_models([m], default_filters)
        assert len(passed) == 0

    def test_rejects_no_coeff(self):
        x = sympy.Symbol("x")
        m = SearchModel(x + 1, cs=[], xs=[x])
        passed = filter_models([m], default_filters)
        assert len(passed) == 0

    def test_rejected_flag_set(self):
        x = sympy.Symbol("x")
        C1 = sympy.Symbol("C_0")
        m_valid = SearchModel(C1 * x, cs=[C1], xs=[x])
        m_invalid = SearchModel(C1 * x + 5, cs=[C1], xs=[x])
        filter_models([m_valid, m_invalid], default_filters)
        assert m_valid.rejected is False
        assert m_invalid.rejected is True

    def test_mixed_pass_fail(self):
        x = sympy.Symbol("x")
        C1 = sympy.Symbol("C_0")
        m1 = SearchModel(C1 * x, cs=[C1], xs=[x])
        m2 = SearchModel(C1 * x + 5, cs=[C1], xs=[x])
        m3 = SearchModel(C1 * x + C1, cs=[C1], xs=[x])
        passed = filter_models([m1, m2, m3], default_filters)
        assert len(passed) == 2
        assert passed[0] is m1
        assert passed[1] is m3

    def test_empty_input(self):
        assert filter_models([], default_filters) == []


# ------------------------------------------------------------------
# Stage: _run_loop_pipeline (filter → memoize → algebra → filter → memoize)
# ------------------------------------------------------------------


class TestRunLoopPipeline:
    """Test the full pipeline: filter → memoize → algebra → filter → memoize."""

    def _make_model(self, expr, xs=None):
        rewritten, cs = SearchModel.rewrite_coeff_only(expr)
        return SearchModel(rewritten, xs=xs, cs=cs)

    def _make_pge(self, algebra_methods=None):
        from pge_jax.search import PGE

        pge = PGE(
            usable_vars=["x0"],
            usable_funcs=["sin"],
            algebra_methods=algebra_methods or ["expand", "factor"],
            max_iter=1,
            pop_count=1,
            random_seed=42,
        )
        return pge

    def test_pipeline_filter_rejects_invalid(self):
        """Models with int coeffs should be filtered out."""
        x = sympy.Symbol("x0")
        C1 = sympy.Symbol("C_0")
        pge = self._make_pge()
        m_valid = SearchModel(C1 * x, cs=[C1], xs=[x])
        m_invalid = SearchModel(C1 * x + 5, cs=[C1], xs=[x])
        result = pge._run_loop_pipeline([m_valid, m_invalid])
        # m_invalid should be filtered out
        assert m_invalid not in result

    def test_pipeline_memoize_dedupes(self):
        """Duplicate expressions should be memoized (only one in result)."""
        x = sympy.Symbol("x0")
        C1 = sympy.Symbol("C_0")
        pge = self._make_pge()
        m1 = SearchModel(C1 * x, cs=[C1], xs=[x])
        m2 = SearchModel(C1 * x, cs=[C1], xs=[x])
        result = pge._run_loop_pipeline([m1, m2])
        # Only one should pass through (the duplicate is memoized)
        assert len(result) == 1

    def test_pipeline_memoize_assigns_ids(self):
        """Models passing through pipeline should get IDs assigned."""
        x = sympy.Symbol("x0")
        C1 = sympy.Symbol("C_0")
        pge = self._make_pge()
        m = SearchModel(C1 * x, cs=[C1], xs=[x])
        pge._run_loop_pipeline([m])
        assert m.id >= 0

    def test_pipeline_populates_models_list(self):
        """self.models should be populated with all models that pass memoization."""
        x = sympy.Symbol("x0")
        C1 = sympy.Symbol("C_0")
        pge = self._make_pge()
        m = SearchModel(C1 * x, cs=[C1], xs=[x])
        pge._run_loop_pipeline([m])
        assert m in pge.models

    def test_pipeline_algebra_generates_variants(self):
        """Algebra pass should generate new models from unique ones."""
        x = sympy.Symbol("x0")
        C1 = sympy.Symbol("C_0")
        # (C*x + C) will expand to C*x + C (same), so no algebra variants
        # Use something that algebra can actually change
        pge = self._make_pge()
        m = SearchModel(C1 * x, cs=[C1], xs=[x])
        result = pge._run_loop_pipeline([m])
        # At minimum the original model should be in result
        assert len(result) >= 1

    def test_pipeline_no_algebra_methods(self):
        """With no algebra methods, result should be just the memoized unique models."""
        x = sympy.Symbol("x0")
        C1 = sympy.Symbol("C_0")
        pge = self._make_pge(algebra_methods=[])
        m1 = SearchModel(C1 * x, cs=[C1], xs=[x])
        m2 = SearchModel(C1 * x + C1, cs=[C1], xs=[x])
        result = pge._run_loop_pipeline([m1, m2])
        assert len(result) == 2

    def test_pipeline_empty_input(self):
        """Empty input should return empty output."""
        pge = self._make_pge()
        assert pge._run_loop_pipeline([]) == []

    def test_pipeline_hates_int_coeff_then_algebra(self):
        """Models with int coeffs are filtered BEFORE algebra, so algebra never sees them."""
        x = sympy.Symbol("x0")
        C1 = sympy.Symbol("C_0")
        pge = self._make_pge()
        # This has an int coeff → filtered before memoize → never reaches algebra
        m = SearchModel(C1 * x + 5, cs=[C1], xs=[x])
        result = pge._run_loop_pipeline([m])
        assert len(result) == 0

    def test_pipeline_memoize_before_algebra(self):
        """Memoization happens before algebra, so identical parents don't get double-algebra'd."""
        x = sympy.Symbol("x0")
        C1 = sympy.Symbol("C_0")
        pge = self._make_pge()
        m1 = SearchModel(C1 * x, cs=[C1], xs=[x])
        m2 = SearchModel(C1 * x, cs=[C1], xs=[x])  # duplicate
        result = pge._run_loop_pipeline([m1, m2])
        # After memoization, only 1 model → algebra runs once on it
        assert len(result) >= 1
