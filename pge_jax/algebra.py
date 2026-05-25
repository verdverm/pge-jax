"""Symbolic algebraic manipulation for the PGE search loop.

Applies simplification methods (``"simplify"``, ``"expand"``,
``"factor"``) to expressions.  If the expression is unchanged, returns
``("same",)`` so the caller can skip downstream processing.
"""

from __future__ import annotations

from typing import Tuple

import sympy  # type: ignore[import-untyped]

from pge_jax.search_model import SearchModel


def manip_model(modl: SearchModel, method: str) -> Tuple[SearchModel | None, str | None]:
    """Apply a simplification method to a model's expression.

    Parameters
    ----------
    modl:
        The model to manipulate.
    method:
        One of ``"simplify"``, ``"expand"``, ``"factor"``.

    Returns
    -------
    tuple[SearchModel | None, str | None]
        ``(new_model, None)`` if the expression changed,
        ``(None, "same")`` if unchanged,
        ``(None, error_message)`` on failure.
    """
    expr, err = do_simp(modl.expr, method)
    if err is not None:
        return None, err
    if expr is None:
        return None, None

    if expr == modl.expr:
        return None, "same"

    ret_modl = SearchModel(expr, xs=modl.xs, cs=modl.cs)
    ret_modl.rewrite_coeff()
    return ret_modl, None


def do_simp(expr, method: str):
    """Apply a sympy simplification method to *expr*.

    Parameters
    ----------
    expr:
        Sympy expression.
    method:
        One of ``"simplify"``, ``"expand"``, ``"factor"``.

    Returns
    -------
    tuple
        ``(simplified_expr, None)`` on success,
        ``(None, error_message)`` on failure.
    """
    if method == "simplify":
        simp = sympy.simplify(expr)
    elif method == "expand":
        simp = sympy.expand(expr)
    elif method == "factor":
        simp = sympy.factor(expr)
    else:
        return None, "unknown method"
    return simp, None
