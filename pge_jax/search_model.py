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

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Tuple

import sympy  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from pge_jax.model import JAXModel


@dataclass
class StageTimings:
    """Per-model timing for each evaluation stage.

    All values are in seconds.  Only populated for models that were
    successfully evaluated (not errored).

    Attributes
    ----------
    build_time:
        Time to compile the JAX model (sympy.lambdify + jax.jacfwd).
    fit_time:
        Time spent in the Levenberg-Marquardt optimizer.
    eval_time:
        Time to predict on full data and compute regression metrics.
    """

    build_time: float = 0.0
    fit_time: float = 0.0
    eval_time: float = 0.0


@dataclass
class IterationStageTimes:
    """Per-iteration stage timing aggregates.

    Stores per-iteration values (not summed) so callers can compute
    min / max / avg without double-counting.

    Attributes
    ----------
    grow:
        List of grow-phase durations per iteration (seconds).
    filter:
        List of filter-phase durations per iteration (seconds).
    algebra:
        List of algebra-phase durations per iteration (seconds).
    peek_eval:
        List of peek-evaluation-phase durations per iteration (seconds).
    full_eval:
        List of full-evaluation-phase durations per iteration (seconds).
    """

    grow: list[float] = field(default_factory=list)
    filter: list[float] = field(default_factory=list)
    algebra: list[float] = field(default_factory=list)
    peek_eval: list[float] = field(default_factory=list)
    full_eval: list[float] = field(default_factory=list)


@dataclass
class FilterBreakdown:
    """Per-filter rejection counts for one filter run.

    Attributes
    ----------
    filter_name:
        Name of the filter function.
    rejections:
        Number of models this filter rejected.
    """

    filter_name: str
    rejections: int


@dataclass
class MemoizeRecord:
    """Memoization statistics for one memoize call.

    Attributes
    ----------
    in_count:
        Number of models passed to memoize.
    unique:
        Number of new unique models inserted.
    duplicates:
        Number of duplicates skipped.
    """

    in_count: int
    unique: int
    duplicates: int


@dataclass
class GrowOperatorStats:
    """Per-operator growth statistics for one parent model.

    Attributes
    ----------
    operator:
        Operator name (e.g. ``"var_xpnd"``, ``"add_xpnd"``, ``"mul_xpnd"``,
        ``"shrunk"``, ``"add_bigr"``).
    raw_exprs:
        Number of raw expressions produced before uniquify.
    unique_exprs:
        Number of unique expressions after uniquify.
    children:
        Number of SearchModel children (raw + expanded = 2× unique).
    avg_size:
        Average tree size of produced expressions.
    """

    operator: str
    raw_exprs: int
    unique_exprs: int
    children: int
    avg_size: float


@dataclass
class GrowOperatorTime:
    """Per-operator timing for one parent model.

    Attributes
    ----------
    operator:
        Operator name (e.g. ``"var_xpnd"``, ``"add_xpnd"``, ``"mul_xpnd"``,
        ``"shrunk"``, ``"add_bigr"``).
    time:
        Duration spent on this operator (seconds).
    """

    operator: str
    time: float


@dataclass
class IterationSubTimes:
    """Per-iteration sub-stage timing details.

    Attributes
    ----------
    grow_expander:
        List of lists of dicts — one per iteration, each containing
        dicts with ``expander`` (index), ``time``, ``popped``, ``children``.
    grow_operators:
        List of lists of ``GrowOperatorStats`` — one per iteration,
        one per parent model grown.
    grow_operator_times:
        List of lists of ``GrowOperatorTime`` — one per iteration,
        one per parent model grown, per operator.
    filter_in:
        List of model counts fed into filter per iteration.
    filter_out:
        List of model counts passing filter per iteration.
    filter_breakdown:
        List of lists of ``FilterBreakdown`` — one per iteration,
        one per filter function.
    filter_time:
        Filter phase duration per iteration.
    memoize:
        List of ``MemoizeRecord`` for each memoize call across all iterations.
    algebra_methods:
        List of lists of dicts — one per iteration, per algebra method.
    """

    grow_expander: list[list[dict]] = field(default_factory=list)
    grow_operators: list[list[list[GrowOperatorStats]]] = field(default_factory=list)
    grow_operator_times: list[list[list[GrowOperatorTime]]] = field(default_factory=list)
    filter_in: list[int] = field(default_factory=list)
    filter_out: list[int] = field(default_factory=list)
    filter_breakdown: list[list[FilterBreakdown]] = field(default_factory=list)
    filter_time: list[float] = field(default_factory=list)
    memoize: list[MemoizeRecord] = field(default_factory=list)
    algebra_methods: list[list[dict]] = field(default_factory=list)


@dataclass
class IterationProgress:
    """Per-iteration summary statistics.

    Attributes
    ----------
    iteration:
        Iteration number (0-based).
    elapsed:
        Wall-clock time since search started (seconds).
    iter_dur:
        Duration of this iteration (seconds).
    grown:
        Number of models grown in this iteration.
    filtered_in:
        Models fed into filter.
    filtered_out:
        Models passing filter.
    algebrad:
        Number of algebraic variants produced.
    algebrad_unique:
        Unique algebraic variants after memoization.
    peek_evaluated:
        Number of models peek-evaluated.
    fully_evaluated:
        Number of models fully evaluated.
    best_score:
        Best score seen so far (lowest).
    population_size:
        Size of nsga2_list in the first expander.
    avg_grown_size:
        Average tree size of grown models.
    avg_filtered_size:
        Average tree size of models passing filter.
    avg_algebra_size:
        Average tree size of algebra variants.
    """

    iteration: int = 0
    elapsed: float = 0.0
    iter_dur: float = 0.0
    grown: int = 0
    filtered_in: int = 0
    filtered_out: int = 0
    algebrad: int = 0
    algebrad_unique: int = 0
    peek_evaluated: int = 0
    fully_evaluated: int = 0
    best_score: float = float("inf")
    population_size: int = 0
    avg_grown_size: float = 0.0
    avg_filtered_size: float = 0.0
    avg_algebra_size: float = 0.0


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
        self._raw_expr: sympy.Expr = expr
        self._expr: Optional[sympy.Expr] = None
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

        # Per-model stage timing
        self.timings: StageTimings = StageTimings()

        # All done
        self.inited = True

    def __hash__(self) -> int:
        return self.id

    @property
    def expr(self) -> sympy.Expr:
        """Expanded expression, computed lazily on first access."""
        if self._expr is None:
            self._expr = sympy.expand(self._raw_expr)
        return self._expr

    @expr.setter
    def expr(self, value: sympy.Expr) -> None:
        """Set the expanded expression (caches it)."""
        self._expr = value

    @property
    def values(self) -> Tuple[float, ...]:
        """Raw fitness values for DEAP-style selection compatibility."""
        return self.fitness_values

    @staticmethod
    def fmt(v, n=2):
        """Format a number: plain for small exponents, E notation for large.

        Values with exponent in [-n, n] use plain format (e.g. 185, 0.12).
        Values outside that range use E notation (e.g. 1.5e5, 1.2e-5).

        Parameters
        ----------
        v:
            Value to format.
        n:
            Exponent threshold.  n=2 for the default table, n=3 for
            wider plain-format range.

        Returns
        -------
        str
            Formatted number string.
        """
        if v is None or v == 0:
            return "0"
        if not math.isfinite(v):
            return str(v)
        exp = math.floor(math.log10(abs(v)))
        if -n <= exp <= n:
            return f"{v:.{n + 1}g}"
        s = f"{v:.{n}e}"
        mantissa, exponent = s.split("e")
        mantissa = mantissa.rstrip("0").rstrip(".")
        return f"{mantissa}e{int(exponent)}"

    def __str__(self) -> str:
        if self.pretty is None:
            self.pretty_expr()
        vals = [self.score, self.r2, self.evar, self.aic, self.bic, self.redchi]
        headers = ["score", "r2", "evar", "aic", "bic", "redchi"]
        formatted = [SearchModel.fmt(v) for v in vals]
        widths = [max(len(h), len(f)) for h, f in zip(headers, formatted)]
        int_strs = [
            f"{self.id:5d}",
            f"{self.iter_id:5d}",
            f"{self.parent_id:5d}",
            f"{self.size():3d}",
            f"{self.psz:3d}",
            f"{self.jsz:3d}",
            f"{self.jpsz:3d}",
        ]
        float_strs = [f"{f:>{w}}" for f, w in zip(formatted, widths)]
        return "  ".join(int_strs) + "    " + "  ".join(float_strs) + "  |  " + (self.pretty or "")

    def pretty_expr(self, n: int = 2) -> str:
        """Substitute fitted coefficient values into the expression."""
        if self.jax_model is not None and self.jax_model.c_values is not None:
            subs = {}
            for i, c in enumerate(self.cs[: self.ncs]):
                v = float(self.jax_model.c_values[i])
                subs[c] = SearchModel.fmt(v, n)
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

    @staticmethod
    def rewrite_coeff_only(expr: sympy.Expr) -> Tuple[sympy.Expr, list]:
        """Rewrite bare ``sympy.Symbol('C')`` into ``C_0, C_1, ...``.

        Pure symbol rewriting with no side effects.

        Parameters
        ----------
        expr:
            Expression containing bare ``sympy.Symbol('C')`` atoms.

        Returns
        -------
        tuple[sympy.Expr, list[sympy.Symbol]]
            ``(rewritten_expr, cs_list)`` where *cs_list* contains
            ``C_0, C_1, ...`` in order of first appearance.
        """
        ii = 0
        rewritten, ii = SearchModel._rewrite_coeff_helper_static(expr, 0)
        cs = [sympy.Symbol(f"C_{i}") for i in range(ii)]
        return rewritten, cs

    @staticmethod
    def _rewrite_coeff_helper_static(expr, ii: int):
        """Static recursive helper for :meth:`rewrite_coeff_only`."""
        ret = expr
        if not expr.is_Atom:
            args = []
            for e in expr.args:
                if not e.is_Atom:
                    ee, ii = SearchModel._rewrite_coeff_helper_static(e, ii)
                    args.append(ee)
                elif e == sympy.Symbol("C"):
                    args.append(sympy.Symbol(f"C_{ii}"))
                    ii += 1
                else:
                    args.append(e)
            args = list(args)
            ret = expr.func(*args)
        return ret, ii

    def rewrite_coeff(self) -> None:
        """Rewrite bare ``sympy.Symbol('C')`` into ``C_0, C_1, ...``.

        Replaces all bare ``C`` atoms in ``self.orig`` with
        sequentially-numbered coefficient symbols, updates
        ``self.expr`` and ``self.cs``, and builds the JAX wrapper.
        """
        rewritten, cs = self.rewrite_coeff_only(self.orig)
        self.expr = rewritten
        self.cs = cs
        self.ncs = len(self.cs)
        self.build_jax_model()

    def build_jax_model(self) -> None:
        """Build the JAX evaluation wrapper from the current expression.

        Creates a :class:`JAXModel` from ``self.expr``, ``self.cs``, and
        ``self.xs``, storing it in ``self.jax_model``.
        """
        from pge_jax.model import JAXModel as _JAXModel

        self.jax_model = _JAXModel(self.expr, cs=self.cs, xs=self.xs)

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
