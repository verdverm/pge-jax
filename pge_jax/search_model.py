"""Search-loop-aware model with state tracking and size metrics.

This is the PGE search-loop equivalent of pypge's ``Model`` class.
It wraps a sympy expression, manages lifecycle state flags, computes
tree-size metrics, and holds fitness values for multi-objective
selection.

Unlike :class:`JAXModel` (which is the pure JAX evaluation wrapper),
this class is designed to be passed through the full PGE pipeline:
filtering, memoization, expansion, evaluation, and selection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Tuple

import sympy

if TYPE_CHECKING:
    from pge_jax.model import JAXModel


class SearchModel:
    """Wraps a sympy expression and manages its search-loop lifecycle.

    Parameters
    ----------
    expr:
        A sympy expression.  Bare ``sympy.Symbol('C')`` leaves are
        treated as optimisable coefficients; all other free symbols
        are input variables.
    xs:
        Explicit list of input-variable symbols.  If ``None`` they
        are inferred from *expr*.
    cs:
        Explicit list of coefficient symbols.  If ``None`` they are
        inferred from *expr*.
    p_id:
        Parent model ID (-2 for root/first-generation models).
    reln:
        Generation relation string (e.g. ``"first_gen"``,
        ``"var_xpnd"``, ``"add_xpnd"``, ``"mul_xpnd"``, ``"shrunk"``).

    Attributes
    ----------
    id : int
        Unique model ID (assigned by :class:`Memoizer`).
    iter_id : int
        Iteration in which this model was created.
    parent_id : int
        Parent model ID.
    gen_relation : str
        How this model was generated.
    orig : sympy.Expr
        Original expression before coefficient rewriting.
    expr : sympy.Expr
        Expanded expression after ``rewrite_coeff()``.
    xs : list[sympy.Symbol]
        Input variable symbols.
    cs : list[sympy.Symbol]
        Coefficient symbols (``C_0, C_1, ...``).
    jax_model : JAXModel | None
        The underlying JAX evaluation wrapper (built lazily).
    sz : int
        Tree size (node count).
    psz : int
        Penalised tree size (+2 per function node).
    jsz : int
        Jacobian size.
    jpsz : int
        Penalised Jacobian size.
    ncs : int
        Number of coefficients.
    peek_score, peek_r2, peek_evar, peek_aic, peek_bic, peek_redchi : float
        Partial fitness computed on a subset of data.
    score, r2, evar, aic, bic, chisqr, redchi, mae, rmae : float
        Full fitness computed on all training data.
    improve_score, improve_r2, improve_evar, improve_aic, improve_bic, improve_redchi : float
        Improvement over parent model.
    fitness_values : tuple[float, ...]
        Raw fitness values for selection.
    wvalues : tuple[float, ...]
        Weighted fitness values (sign × value).
    crowding_dist : float
        Crowding distance for diversity maintenance.
    """

    # Lifecycle state flags
    inited: bool
    memoized: bool
    algebrad: bool
    peeked: bool
    peek_queued: bool
    peek_popped: bool
    evaluated: bool
    queued: bool
    popped: bool
    expanded: bool
    finalized: bool
    errored: bool

    def __init__(
        self,
        expr: sympy.Expr,
        xs=None,
        cs=None,
        p_id: int = -2,
        reln: str = "unknown",
    ):
        # Identification
        self.id: int = -2
        self.iter_id: int = -2
        self.parent_id: int = p_id
        self.gen_relation: str = reln

        # State flags
        self.inited = False
        self.memoized = False
        self.algebrad = False
        self.peeked = False
        self.peek_queued = False
        self.peek_popped = False
        self.evaluated = False
        self.queued = False
        self.popped = False
        self.expanded = False
        self.finalized = False
        self.errored = False

        # Expression
        self.orig: sympy.Expr = expr
        self.expr: sympy.Expr = sympy.expand(expr)
        self.pretty: Optional[str] = None

        # Variables and coefficients
        self.xs: list = list(xs) if xs is not None else []
        self.cs: list = list(cs) if cs is not None else []

        # JAX evaluation wrapper (built lazily)
        self.jax_model: Optional[JAXModel] = None

        # Size metrics
        self.sz: int = 0
        self.psz: int = 0
        self.jsz: int = 0
        self.jpsz: int = 0
        self.ncs: int = len(self.cs)

        # Peek (partial) fitness
        self.peek_score: Optional[float] = None
        self.peek_r2: Optional[float] = None
        self.peek_evar: Optional[float] = None
        self.peek_aic: Optional[float] = None
        self.peek_bic: Optional[float] = None
        self.peek_redchi: Optional[float] = None

        # Full fitness
        self.score: Optional[float] = None
        self.r2: Optional[float] = None
        self.evar: Optional[float] = None
        self.aic: Optional[float] = None
        self.bic: Optional[float] = None
        self.chisqr: Optional[float] = None
        self.redchi: Optional[float] = None
        self.mae: Optional[float] = None
        self.rmae: Optional[float] = None

        # Improvement over parent
        self.improve_score: Optional[float] = None
        self.improve_r2: Optional[float] = None
        self.improve_evar: Optional[float] = None
        self.improve_aic: Optional[float] = None
        self.improve_bic: Optional[float] = None
        self.improve_redchi: Optional[float] = None

        # Fitness for selection
        self.fitness_values: Tuple[float, ...] = ()
        self.wvalues: Tuple[float, ...] = ()
        self.crowding_dist: float = 0.0

        # Fit bookkeeping
        self.peek_nfev: int = 0
        self.eval_nfev: int = 0
        self.total_fev: int = 0

        # All done
        self.inited = True

    def __hash__(self) -> int:
        return self.id

    def __str__(self) -> str:
        if self.pretty is None:
            self.pretty_expr()
        fs = (
            "{:5d}  {:5d}  {:5d}    {:2d}   {:2d}   {:2d}   {:2d}"
            "  {:15.6f}  {:10.6f}  {:10.6f}  {:15.6f}  {:15.6f}  {:15.6f}  |  {:s}"
        )
        return fs.format(
            self.id,
            self.iter_id,
            self.parent_id,
            self.size(),
            self.psz,
            self.jsz,
            self.jpsz,
            self.score or 0,
            self.r2 or 0,
            self.evar or 0,
            self.aic or 0,
            self.bic or 0,
            self.redchi or 0,
            self.pretty,
        )

    def pretty_expr(self, float_format: str = "%.4e") -> str:
        """Substitute fitted coefficient values into the expression."""
        if self.jax_model is not None and self.jax_model.c_values is not None:
            subs = dict(zip(self.cs, self.jax_model.c_values.tolist()[: self.ncs]))
            self.pretty = str(self.expr.subs(subs))
        else:
            self.pretty = str(self.expr)
        return self.pretty

    # ------------------------------------------------------------------
    # Size calculation
    # ------------------------------------------------------------------

    def size(self) -> int:
        """Return the tree size, computing it lazily if needed."""
        if self.sz == 0:
            self.sz, self.psz = self.calc_tree_size()
            self.jsz, self.jpsz = self.calc_jac_size()
        return self.sz

    def calc_tree_size(self) -> Tuple[int, int]:
        """Compute tree size and penalised tree size.

        Walks ``sympy.preorder_traversal``, counts nodes, and adds
        +2 penalty per function node.  Integer leaf values are added
        as their absolute value.

        Returns
        -------
        tuple[int, int]
            ``(size, penalised_size)``.
        """
        i = 0
        p = 0
        for e in sympy.preorder_traversal(self.expr):
            if e.is_Integer:
                i += int(abs(e))
                continue
            if e.is_Function:
                p += 2
            i += 1
        return i, i + p

    def calc_jac_size(self) -> Tuple[int, int]:
        """Compute Jacobian size and penalised Jacobian size.

        Returns
        -------
        tuple[int, int]
            ``(size, penalised_size)``.
        """
        # Build the Jacobian symbolically
        if len(self.cs) == 0:
            return 0, 0
        jac_exprs = [sympy.diff(self.expr, c) for c in self.cs]
        i, p = 0, 0
        for jac in jac_exprs:
            for e in sympy.preorder_traversal(jac):
                if e.is_Integer:
                    i += int(abs(e))
                    continue
                if e.is_Function:
                    p += 2
                i += 1
        return i, i + p

    # ------------------------------------------------------------------
    # Coefficient rewriting (converts bare C to C_0, C_1, ...)
    # ------------------------------------------------------------------

    def rewrite_coeff(self) -> None:
        """Rewrite bare ``sympy.Symbol('C')`` into ``C_0, C_1, ...``.

        Replaces all bare ``C`` atoms in ``self.orig`` with
        sequentially-numbered coefficient symbols, updates
        ``self.expr`` and ``self.cs``, and builds the JAX wrapper.
        """
        from pge_jax.model import JAXModel as _JAXModel

        C_sym = sympy.Symbol("C")
        ii = 0
        expr, ii = self._rewrite_coeff_helper(self.orig, 0)
        self.expr = expr
        self.cs = [sympy.Symbol(f"C_{i}") for i in range(ii)]
        self.ncs = len(self.cs)

        # Build JAX wrapper
        self.jax_model = _JAXModel(expr, cs=self.cs, xs=self.xs)

    def _rewrite_coeff_helper(self, expr, ii: int):
        """Recursive helper for :meth:`rewrite_coeff`."""
        ret = expr
        if not expr.is_Atom:
            args = []
            for e in expr.args:
                if not e.is_Atom:
                    ee, ii = self._rewrite_coeff_helper(e, ii)
                    args.append(ee)
                elif e == sympy.Symbol("C"):
                    args.append(sympy.Symbol(f"C_{ii}"))
                    ii += 1
                else:
                    args.append(e)
            args = tuple(args)
            ret = expr.func(*args)
        return ret, ii

    # ------------------------------------------------------------------
    # Fitness comparison helpers
    # ------------------------------------------------------------------

    def dominates(self, other: "SearchModel") -> bool:
        """Check if this model dominates *other* in fitness.

        Uses weighted fitness values (``wvalues``).  A model dominates
        another if it is no worse in all objectives and strictly better
        in at least one (minimisation semantics: lower is better).

        Parameters
        ----------
        other:
            Another model to compare against.

        Returns
        -------
        bool
            ``True`` if this model dominates *other*.
        """
        if not hasattr(other, "wvalues") or not self.wvalues:
            return False
        not_equal = False
        for self_w, other_w in zip(self.wvalues, other.wvalues):
            if self_w > other_w:
                return False
            if self_w < other_w:
                not_equal = True
        return not_equal
