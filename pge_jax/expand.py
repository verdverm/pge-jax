"""Expression expansion (Grower) for the PGE search loop.

Implements grammar-based expression enumeration.  Takes existing models
and produces expanded children via variable substitution, addition
extension, multiplication extension, and shrinker operators.
"""

from __future__ import annotations

from itertools import combinations as combos_woutR
from itertools import combinations_with_replacement as combos_withR
from typing import List, Optional

import sympy

from pge_jax.search_model import SearchModel

C = sympy.Symbol("C")

BASIC_BASE = [sympy.exp, sympy.cos, sympy.sin]
BASIC_MISC = [sympy.Abs, sympy.sqrt, sympy.log, sympy.exp]
BASIC_TRIG = [sympy.cos, sympy.sin, sympy.tan]
HYPER_TRIG = [sympy.cosh, sympy.sinh, sympy.tanh]


def map_names_to_funcs(names: List[str]) -> List:
    """Map function name strings to sympy function classes.

    Parameters
    ----------
    names:
        List of function name strings (e.g. ``["sin", "exp"]``).

    Returns
    -------
    list
        List of sympy function classes.
    """
    funcs = []
    name_to_func = {
        "sqrt": sympy.sqrt,
        "abs": sympy.Abs,
        "sin": sympy.sin,
        "cos": sympy.cos,
        "tan": sympy.tan,
        "exp": sympy.exp,
        "log": sympy.log,
        "sinh": sympy.sinh,
        "cosh": sympy.cosh,
        "tanh": sympy.tanh,
    }
    for name in names:
        if name not in name_to_func:
            raise ValueError(f"Unknown function name: {name!r}")
        funcs.append(name_to_func[name])
    return funcs


class Grower:
    """Grammar-based expression enumeration.

    Pre-computes expression pools (variable powers, products, function
    expressions) and provides operators to expand existing models.

    Parameters
    ----------
    xs:
        Input variable symbols (single ``sympy.Symbol`` or list).
    funcs:
        List of sympy function classes to use (e.g. ``[sympy.sin, sympy.exp]``).
    **kwargs:
        Policy configuration overrides.  Supported keys:

        func_level : str
            ``"linear"`` or ``"nonlin"`` — affine transforms or not.
        init_level : str
            ``"low"``, ``"med"``, ``"high"`` — first-gen complexity.
        grow_level : str
            ``"low"``, ``"med"``, ``"high"`` — expansion complexity.
        subs_level : str
            Variable substitution complexity.
        adds_level : str
            Addition extension complexity.
        muls_level : str
            Multiplication extension complexity.
        add_xtop : bool
            Extend at top level of additions.
        shrinker : bool
            Try removing terms from additions.
        limiting_depth : int
            Depth limit for variable substitution (default 4).
        grow_filter : bool
            Avoid duplicate terms in additions.
    """

    def __init__(
        self,
        xs,
        funcs: Optional[List],
        **kwargs,
    ):
        # Normalise single symbol to list
        if isinstance(xs, sympy.Symbol):
            xs = [xs]
        self.xs = xs
        self.funcs = funcs

        # Policy defaults
        self.func_level: str = "linear"
        self.init_level: str = "low"
        self.add_xtop: bool = False
        self.shrinker: bool = False
        self.limiting_depth: int = 4
        self.grow_filter: bool = False
        self.grow_level: str = "low"
        self.subs_level: str = "low"
        self.adds_level: str = "low"
        self.muls_level: str = "low"

        # Apply kwargs overrides
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Build pre-computed expression pools
        self._build_pools()

    # ------------------------------------------------------------------
    # Pool construction
    # ------------------------------------------------------------------

    def _build_pools(self) -> None:
        """Build all pre-computed expression pools."""
        # Variable powers: x^n, x^-n for n=1..4
        self.xs_pow1: list = [x ** (n * (p + 1)) for p in range(1) for n in [-1, 1] for x in self.xs]
        self.xs_pow2: list = [x ** (n * (p + 1)) for p in range(2) for n in [-1, 1] for x in self.xs]
        self.xs_pow3: list = [x ** (n * (p + 1)) for p in range(3) for n in [-1, 1] for x in self.xs]
        self.xs_pow4: list = [x ** (n * (p + 1)) for p in range(4) for n in [-1, 1] for x in self.xs]

        # Variable products (without coefficient)
        self.wout_c_xs1_muls: list = list(self.xs_pow1)
        self.wout_c_xs2_muls: list = [a * b for a, b in combos_withR(self.xs_pow1, 2)] + self.wout_c_xs1_muls
        self.wout_c_xs3_muls: list = [a * b * c for a, b, c in combos_withR(self.xs_pow1, 3)] + self.wout_c_xs2_muls
        self.wout_c_xs4_muls: list = [
            a * b * c * d for a, b, c, d in combos_withR(self.xs_pow1, 4)
        ] + self.wout_c_xs3_muls
        self.wout_c_xs3_muls = self._uniquify(self.wout_c_xs3_muls)

        # Coefficient-scaled variable products
        self.with_c_xs1_muls: list = [C * m for m in self.wout_c_xs1_muls]
        self.with_c_xs2_muls: list = [C * m for m in self.wout_c_xs2_muls]
        self.with_c_xs3_muls: list = [C * m for m in self.wout_c_xs3_muls]
        self.with_c_xs4_muls: list = [C * m for m in self.wout_c_xs4_muls]

        # Function expressions
        if self.funcs is not None:
            self.wout_c_linear_funcs: list = [f(x) for f in self.funcs for x in self.wout_c_xs1_muls]
            self.wout_c_nonlin_funcs: list = [f(C * x + C) for f in self.funcs for x in self.wout_c_xs1_muls]
            self.with_c_linear_funcs: list = [f(x) for f in self.funcs for x in self.wout_c_xs1_muls]
            self.with_c_nonlin_funcs: list = [f(C * x + C) for f in self.funcs for x in self.wout_c_xs1_muls]
        else:
            self.wout_c_linear_funcs = []
            self.wout_c_nonlin_funcs = []
            self.with_c_linear_funcs = []
            self.with_c_nonlin_funcs = []

        # Combined function expressions (with inverse)
        self.with_c_func_exprs: list = []
        self.wout_c_func_exprs: list = []
        if self.func_level == "linear":
            base = self.with_c_linear_funcs
            alt = self.wout_c_linear_funcs
        elif self.func_level in ("nonlin", "nonlinear"):
            base = self.with_c_nonlin_funcs
            alt = self.wout_c_nonlin_funcs
        else:
            raise ValueError(f"Unknown func_level: {self.func_level!r}")

        self.with_c_func_exprs = base + [f ** (-1) for f in base]
        self.wout_c_func_exprs = alt + [f ** (-1) for f in alt]

        # Expansion term pools
        self.init_var_subs()
        self.init_add_extends()
        self.init_mul_extends()

    def init_var_subs(self) -> None:
        """Build variable substitution term pools."""
        add_terms = [C * x + C for x in self.xs]

        self.var_sub_dep_lim_terms = list(self.wout_c_xs2_muls)
        self.var_sub_dep_terms = self.var_sub_dep_lim_terms + list(self.wout_c_func_exprs)

        if self.subs_level == "low":
            self.var_sub_lim_terms = list(self.wout_c_xs2_muls)
            self.var_sub_terms = self.var_sub_lim_terms + list(self.wout_c_func_exprs)
        elif self.subs_level == "med":
            self.var_sub_lim_terms = list(self.wout_c_xs2_muls) + add_terms
            self.var_sub_terms = self.var_sub_lim_terms + list(self.wout_c_func_exprs)
        elif self.subs_level == "high":
            self.var_sub_dep_lim_terms += add_terms
            self.var_sub_dep_terms += add_terms
            self.var_sub_lim_terms = list(self.wout_c_xs3_muls) + add_terms
            self.var_sub_terms = self.var_sub_lim_terms + list(self.wout_c_func_exprs)
        else:
            raise ValueError(f"Unknown subs_level: {self.subs_level!r}")

        self.var_sub_terms = self._uniquify(self.var_sub_terms)
        self.var_sub_lim_terms = self._uniquify(self.var_sub_lim_terms)

    def init_add_extends(self) -> None:
        """Build addition extension term pools."""
        if self.adds_level == "low":
            self.add_extend_lim_terms = list(self.with_c_xs1_muls)
            self.add_extend_terms = list(self.with_c_xs1_muls) + list(self.with_c_func_exprs)
        elif self.adds_level == "med":
            self.add_extend_lim_terms = list(self.with_c_xs2_muls)
            self.add_extend_terms = list(self.with_c_xs2_muls) + list(self.with_c_func_exprs)
        elif self.adds_level == "high":
            self.add_extend_lim_terms = list(self.with_c_xs2_muls)
            cross = [x * f for f in self.with_c_func_exprs for x in self.with_c_xs1_muls]
            self.add_extend_terms = list(self.with_c_xs2_muls) + list(self.with_c_func_exprs) + cross
        else:
            raise ValueError(f"Unknown adds_level: {self.adds_level!r}")

        self.add_extend_terms = self._uniquify(self.add_extend_terms)
        self.add_extend_lim_terms = self._uniquify(self.add_extend_lim_terms)

    def init_mul_extends(self) -> None:
        """Build multiplication extension term pools."""
        if self.muls_level == "low":
            self.mul_extend_lim_terms = list(self.wout_c_xs1_muls)
            self.mul_extend_terms = list(self.wout_c_xs1_muls) + list(self.wout_c_func_exprs)
        elif self.muls_level == "med":
            self.mul_extend_lim_terms = list(self.wout_c_xs2_muls)
            self.mul_extend_terms = list(self.wout_c_xs2_muls) + list(self.wout_c_func_exprs)
        elif self.muls_level == "high":
            self.mul_extend_lim_terms = list(self.wout_c_xs2_muls)
            cross = [x * f for f in self.wout_c_func_exprs for x in self.wout_c_xs1_muls]
            self.mul_extend_terms = list(self.wout_c_xs2_muls) + list(self.wout_c_func_exprs) + cross
        else:
            raise ValueError(f"Unknown muls_level: {self.muls_level!r}")

        self.mul_extend_terms = self._uniquify(self.mul_extend_terms)
        self.mul_extend_lim_terms = self._uniquify(self.mul_extend_lim_terms)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def first_exprs(self) -> List[SearchModel]:
        """Generate the initial generation of models.

        Returns
        -------
        list[SearchModel]
            First-generation models with ``gen_relation="first_gen"``.
        """
        if self.init_level == "low":
            if len(self.xs) > 3:
                mul_exprs = self.with_c_xs1_muls
            else:
                mul_exprs = self.with_c_xs2_muls
        elif self.init_level == "med":
            if len(self.xs) > 3:
                mul_exprs = self.with_c_xs1_muls + self.with_c_func_exprs
            elif len(self.xs) > 1:
                mul_exprs = self.with_c_xs2_muls + self.with_c_func_exprs
            else:
                mul_exprs = self.with_c_xs3_muls + self.with_c_func_exprs
        elif self.init_level == "high":
            if len(self.xs) > 3:
                mul_exprs = self.with_c_xs2_muls + self.with_c_func_exprs
            elif len(self.xs) > 1:
                mul_exprs = self.with_c_xs3_muls + self.with_c_func_exprs
            else:
                mul_exprs = self.with_c_xs4_muls + self.with_c_func_exprs
        else:
            raise ValueError(f"Unknown init_level: {self.init_level!r}")

        mid_exprs = mul_exprs
        add_exprs = [a + b for a, b in combos_woutR(mid_exprs, 2)]
        if self.init_level == "high":
            add_exprs += [a + b + c for a, b, c in combos_woutR(mid_exprs, 3)]

        exprs_set = mid_exprs + add_exprs + self.with_c_func_exprs

        # Always add the +C variants
        plus_C_exprs = [sympy.Add(expr, C) for expr in exprs_set]
        ret_exprs = exprs_set + plus_C_exprs

        # Uniquify
        ret_exprs = self._uniquify(ret_exprs)

        models = [SearchModel(e, xs=self.xs) for e in ret_exprs]
        for m in models:
            m.gen_relation = "first_gen"
            m.parent_id = -1
            m.rewrite_coeff()  # Build JAX wrapper

        return models

    def grow(self, M: SearchModel) -> List[SearchModel]:
        """Expand a single model via all growth operators.

        Parameters
        ----------
        M:
            Model to expand.

        Returns
        -------
        list[SearchModel]
            Child models from all expansion operators.
        """
        var_expands = self._var_sub(M.orig)
        add_expands = self._add_extend(M.orig)
        mul_expands = self._mul_extend(M.orig)

        var_expands_C = [self._toggle_plus_C(e) for e in var_expands]
        add_expands_C = [self._toggle_plus_C(e) for e in add_expands]
        mul_expands_C = [self._toggle_plus_C(e) for e in mul_expands]

        var_expands = self._uniquify(var_expands + var_expands_C)
        add_expands = self._uniquify(add_expands + add_expands_C)
        mul_expands = self._uniquify(mul_expands + mul_expands_C)

        add_biggers = []
        if self.add_xtop:
            add_biggers = self._add_extend_top_level(M.orig)
            add_biggers_C = [self._toggle_plus_C(e) for e in add_biggers]
            add_biggers = self._uniquify(add_biggers + add_biggers_C)

        shrunk = []
        if self.shrinker:
            shrunk = self._shrinker(M.orig)

        var_models = [SearchModel(e, p_id=M.id, reln="var_xpnd") for e in var_expands if e != C]
        big_models = [SearchModel(e, p_id=M.id, reln="add_bigr") for e in add_biggers if e != C]
        add_models = [SearchModel(e, p_id=M.id, reln="add_xpnd") for e in add_expands if e != C]
        mul_models = [SearchModel(e, p_id=M.id, reln="mul_xpnd") for e in mul_expands if e != C]
        shrunk_models = [SearchModel(e, p_id=M.id, reln="shrunk") for e in shrunk if e != C]

        return var_models + big_models + add_models + mul_models + shrunk_models

    # ------------------------------------------------------------------
    # Expansion operators
    # ------------------------------------------------------------------

    def _var_sub(self, expr, limit_sub: bool = False, depth: int = 1) -> list:
        """Variable substitution — replace input variables with complex terms."""
        new_exprs: list = []

        if expr.is_Atom:
            return new_exprs

        args_sets: list = []
        for i, e in enumerate(expr.args):
            if not e.is_Atom:
                lim_sub = limit_sub or not (e.is_Add or e.is_Mul)
                ee = self._var_sub(e, lim_sub, depth=depth + 1)
                if len(ee) > 0:
                    args_map = set(expr.args)
                    for vs in ee:
                        if self.grow_filter and expr.is_Add and vs in args_map:
                            continue
                        cloned_args = list(expr.args)
                        cloned_args[i] = vs
                        args_sets.append(cloned_args)
            elif e in self.xs:
                sub_terms = None
                if depth >= self.limiting_depth:
                    sub_terms = self.var_sub_dep_terms
                else:
                    sub_terms = self.var_sub_terms
                if limit_sub:
                    if depth >= self.limiting_depth:
                        sub_terms = self.var_sub_dep_lim_terms
                    else:
                        sub_terms = self.var_sub_lim_terms

                for vs in sub_terms:
                    cloned_args = list(expr.args)
                    cloned_args[i] = vs
                    args_sets.append(cloned_args)

        for args in args_sets:
            args = tuple(args)
            tmp = expr.func(*args)
            new_exprs.append(tmp)

        return self._uniquify(new_exprs)

    def _add_extend(self, expr, limit_sub: bool = False) -> list:
        """Addition extension — add terms to Add nodes."""
        new_exprs: list = []

        if expr.is_Atom:
            return new_exprs

        args_sets: list = []

        if expr.is_Add:
            sub_terms = self.add_extend_terms
            if limit_sub:
                sub_terms = self.add_extend_lim_terms
            args_map = set(expr.args)
            for term in sub_terms:
                if self.grow_filter and term in args_map:
                    continue
                cloned_args = list(expr.args)
                cloned_args.append(term)
                args_sets.append(cloned_args)

        for i, e in enumerate(expr.args):
            if not e.is_Atom:
                lim_sub = limit_sub or not (e.is_Add or e.is_Mul)
                ee = self._add_extend(e, limit_sub=lim_sub)
                if len(ee) > 0:
                    args_map = set(expr.args)
                    for vs in ee:
                        if self.grow_filter and expr.is_Add and vs in args_map:
                            continue
                        cloned_args = list(expr.args)
                        cloned_args[i] = vs
                        args_sets.append(cloned_args)

        for args in args_sets:
            args = tuple(args)
            tmp = expr.func(*args)
            new_exprs.append(tmp)

        return self._uniquify(new_exprs)

    def _mul_extend(self, expr, limit_sub: bool = False) -> list:
        """Multiplication extension — add factors to Mul nodes."""
        new_exprs: list = []

        if expr.is_Atom:
            return new_exprs

        args_sets: list = []

        if expr.is_Mul:
            sub_terms = self.mul_extend_terms
            if limit_sub:
                sub_terms = self.mul_extend_lim_terms
            for term in sub_terms:
                cloned_args = list(expr.args)
                cloned_args.append(term)
                args_sets.append(cloned_args)

        for i, e in enumerate(expr.args):
            if not e.is_Atom:
                lim_sub = limit_sub or not (e.is_Add or e.is_Mul)
                ee = self._mul_extend(e, limit_sub=lim_sub)
                if len(ee) > 0:
                    args_map = set(expr.args)
                    for vs in ee:
                        if self.grow_filter and expr.is_Add and vs in args_map:
                            continue
                        cloned_args = list(expr.args)
                        cloned_args[i] = vs
                        args_sets.append(cloned_args)

        for args in args_sets:
            args = tuple(args)
            tmp = expr.func(*args)
            new_exprs.append(tmp)

        return self._uniquify(new_exprs)

    def _add_extend_top_level(self, expr) -> list:
        """Top-level addition extension (non-recursive)."""
        if not expr.is_Add:
            return []

        new_terms: list = []
        for e in expr.args:
            if e.is_Mul:
                for term in self.mul_extend_terms:
                    cloned_args = list(e.args)
                    cloned_args.append(term)
                    new_mul = e.func(*cloned_args)
                    new_terms.append(new_mul)

        new_exprs: list = []
        args_map = set(expr.args)
        for term in new_terms:
            if self.grow_filter and term in args_map:
                continue
            cloned_args = list(expr.args)
            cloned_args.append(term)
            bigger_add = expr.func(*cloned_args)
            new_exprs.append(bigger_add)

        return self._uniquify(new_exprs)

    def _shrinker(self, expr) -> list:
        """Term removal from Add nodes (recursive)."""
        if expr.is_Atom:
            return []

        new_exprs: list = []

        if expr.is_Add:
            for i, e in enumerate(expr.args):
                cloned_args = list(expr.args)
                del cloned_args[i]
                smaller_add = expr.func(*cloned_args)
                new_exprs.append(smaller_add)

        args_sets: list = []
        for i, e in enumerate(expr.args):
            if not e.is_Atom:
                ee = self._shrinker(e)
                if len(ee) > 0:
                    args_map = set(expr.args)
                    for vs in ee:
                        if self.grow_filter and expr.is_Add and vs in args_map:
                            continue
                        cloned_args = list(expr.args)
                        cloned_args[i] = vs
                        args_sets.append(cloned_args)

        for args in args_sets:
            args = tuple(args)
            tmp = expr.func(*args)
            new_exprs.append(tmp)

        return self._uniquify(new_exprs)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _uniquify(self, exprs: list) -> list:
        """Remove duplicate expressions via ``evalf()`` hashing."""
        pass_set = set()
        for p in exprs:
            s = p.evalf()
            pass_set.add(s)
        return list(pass_set)

    def _toggle_plus_C(self, expr) -> sympy.Expr:
        """Add or remove a bare ``C`` from an addition."""
        if expr.is_Add:
            hasC = any(e == C for e in expr.args)
            if not hasC:
                return expr + C
            args = tuple(e for e in expr.args if e != C)
            return expr.func(*args)
        return sympy.Add(expr, C)
