"""Expression validity filters for the PGE search loop.

Each filter takes a ``SearchModel`` and a sympy ``Expr`` and returns
``True`` if the expression should be **rejected** (removed from the
population).  Filters are applied via
:func:`filter_models` which walks the full expression tree.
"""

from __future__ import annotations

from sympy import Symbol, preorder_traversal  # type: ignore[import-untyped]
from sympy.core.numbers import Integer, One, Zero  # type: ignore[import-untyped]

from pge_jax.search_model import SearchModel

C = Symbol("C")


def filter_models(models: list[SearchModel], filters: list) -> list[SearchModel]:
    """Return only models that pass every filter.

    Parameters
    ----------
    models:
        Candidate models to test.
    filters:
        List of filter callables.  Each receives ``(model, expr)`` and
        returns ``True`` to reject.

    Returns
    -------
    list[SearchModel]
        Models that passed all filters.
    """
    return [modl for modl in models if not filter_model(modl, modl.orig, filters)]


def filter_model(modl: SearchModel, expr, filters: list) -> bool:
    """Walk *expr* tree and return ``True`` if any filter rejects it.

    Parameters
    ----------
    modl:
        The model being tested.
    expr:
        Sympy expression to check.
    filters:
        List of filter callables.

    Returns
    -------
    bool
        ``True`` if the expression should be rejected.
    """
    for e in preorder_traversal(expr):
        for f in filters:
            if f(modl, e):
                return True
    return False


def filter_just_C(modl: SearchModel, expr) -> bool:
    """Reject expressions that are just the bare coefficient symbol ``C`` or a number."""
    if modl.orig is C or modl.orig == C or modl.orig.is_Number:
        return True
    return False


def filter_no_C(modl: SearchModel, expr) -> bool:
    """Reject expressions with no coefficients (constant expressions)."""
    if len(modl.cs) == 0:
        return True
    return False


def filter_too_big(modl: SearchModel, expr, big: int = 64) -> bool:
    """Reject models whose tree size exceeds *big*."""
    if modl.size() > big:
        return True
    return False


def filter_has_int_coeff(modl: SearchModel, expr) -> bool:
    """Reject expressions with hardcoded integer coefficients (not using ``C`` symbols).

    A coefficient is considered "hardcoded" if an ``Add`` or ``Mul`` node
    has a non-zero / non-one coefficient extracted via
    ``as_coeff_Add`` / ``as_coeff_Mul``.
    """
    if expr.is_Mul:
        cs, _ = expr.as_coeff_Mul()
        if type(cs) not in (One, Integer(1)):
            return True
    if expr.is_Add:
        cs, _ = expr.as_coeff_Add()
        if type(cs) not in (Zero, Integer(0)):
            return True
    return False


def filter_has_big_pow(modl: SearchModel, expr, big: int = 6) -> bool:
    """Reject expressions with power exponents whose absolute value exceeds *big*."""
    if expr.is_Pow:
        B, E = expr.as_base_exp()
        if abs(E) > big:
            return True
    return False


def filter_has_coeff_pow(modl: SearchModel, expr) -> bool:
    """Reject expressions like ``C^2`` (coefficient raised to a power)."""
    if expr.is_Pow:
        B, E = expr.as_base_exp()
        if B is C or B == C or B.is_Number:
            return True
    return False


default_filters: list = [
    filter_too_big,
    filter_has_int_coeff,
    filter_has_big_pow,
    filter_just_C,
    filter_no_C,
    filter_has_coeff_pow,
]
