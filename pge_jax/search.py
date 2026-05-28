"""PGE search loop — Prioritized Grammar Enumeration.

Orchestrates the full symbolic regression search: expression generation,
filtering, memoization, algebraic manipulation, progressive evaluation,
and multi-objective selection.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import jax.numpy as jnp
import numpy as np
import sympy  # type: ignore[import-untyped]

from pge_jax.algebra import manip_model
from pge_jax.evaluate import evaluate, fit_model
from pge_jax.expand import Grower, GrowOperatorStats, map_names_to_funcs
from pge_jax.filters import filter_models, filter_models_with_stats
from pge_jax.search_model import (
    GrowOperatorTime,
    IterationProgress,
    IterationStageTimes,
    IterationSubTimes,
    MemoizeRecord,
    SearchModel,
)
from pge_jax.selection import (
    selNSGA2,
    sortLogNondominated,
)

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Default fitness parameters (minimise size + error, maximise R²)
# ---------------------------------------------------------------------------

DEFAULT_FITNESS_PARAMS = [
    "-(1)jpsz",
    "-score",
    "+r2",
    "+bic",
]


@dataclass
class ExpanderConfig:
    """Configuration for a single expander in a multi-expander run.

    Parameters
    ----------
    pop_count:
        Number of models to pop from the expander's population per iteration.
    usable_funcs:
        Function name strings (e.g. ``["sin", "exp"]``).
    grow_params:
        Keyword arguments passed to :class:`Grower`.
    """

    pop_count: int = 3
    usable_funcs: List[str] = field(default_factory=lambda: ["sin", "cos", "exp", "log"])
    grow_params: Dict[str, Any] = field(default_factory=dict)


class PGE:
    """Prioritized Grammar Enumeration search loop.

    Parameters
    ----------
    usable_vars:
        Variable symbol names (e.g. ``["x", "y"]``) or a list of
        ``sympy.Symbol`` objects.
    usable_funcs:
        Function name strings for the default expander.
    max_iter:
        Maximum number of search iterations.
    pop_count:
        Number of models to pop from the population per expander per
        iteration.
    peek_count:
        Number of models to pop from the peek heap for full evaluation.
    peek_fraction:
        Fraction of training data to use for peek evaluation (0 = skip peek,
        values between 0 and 1 exclusive).
    min_size:
        Minimum tree size.
    max_size:
        Maximum tree size (default 64).
    max_power:
        Maximum allowed power exponent.
    algebra_methods:
        List of algebraic manipulation methods (``"expand"``, ``"factor"``,
        ``"simplify"``).
    err_method:
        Error metric name for ``score`` (``"mse"``, ``"rmse"``, etc.).
    fitness_func_params:
        Multi-objective fitness parameters.
    multi_expander_params:
        List of :class:`ExpanderConfig` for multi-expander runs.
    grow_params:
        Keyword arguments for the default expander's :class:`Grower`.
    random_seed:
        Random seed for reproducibility.

    Attributes
    ----------
    final : list[SearchModel]
        All fully evaluated, finalized models.
    final_paretos : list[list[SearchModel]] | None
        Final Pareto fronts (set after :meth:`finalize`).
    """

    def __init__(
        self,
        usable_vars=None,
        usable_funcs: List[str] | None = None,
        max_iter: int = 100,
        pop_count: int = 3,
        peek_count: int = 6,
        peek_fraction: float = 0.0,
        min_size: int = 1,
        max_size: int = 64,
        max_power: int = 5,
        algebra_methods: List[str] | None = None,
        err_method: str = "mse",
        fitness_func_params: List[str] | None = None,
        multi_expander_params: List[Dict] | None = None,
        grow_params: Dict[str, Any] | None = None,
        random_seed: int = 23,
    ):
        random.seed(random_seed)
        np.random.seed(random_seed)

        # Timing
        self._phase_times: Dict[str, float] = {}

        # Per-iteration stage times (not summed — allows min/max/avg)
        self._loop_stage_times: IterationStageTimes = IterationStageTimes()

        # Per-iteration total wall times (not summed)
        self._iteration_times: List[float] = []

        # Per-iteration sub-stage details
        self._loop_sub_times: IterationSubTimes = IterationSubTimes()

        # Per-iteration progress summaries
        self._iteration_progress: List[IterationProgress] = []

        # Config
        self.max_iter: int = max_iter
        self.pop_count: int = pop_count
        self.peek_count: int = peek_count
        self.peek_fraction: float = peek_fraction
        self.min_size: int = min_size
        self.max_size: int = max_size
        self.max_power: int = max_power
        self.algebra_methods: List[str] = algebra_methods or ["expand", "factor"]
        self.err_method: str = err_method

        # Variables
        if usable_vars is not None:
            if isinstance(usable_vars, str):
                self.vars = sympy.symbols(usable_vars)
            elif isinstance(usable_vars, list) and all(isinstance(v, sympy.Symbol) for v in usable_vars):
                self.vars = usable_vars
            else:
                self.vars = sympy.symbols(usable_vars)
            if isinstance(self.vars, sympy.Symbol):
                self.vars = [self.vars]
        else:
            self.vars = []

        # Fitness
        self.fitness_func_params = fitness_func_params or DEFAULT_FITNESS_PARAMS
        self.fitness_calc = None  # set in fit() after seeing models

        # Resolve defaults
        _usable_funcs = usable_funcs if usable_funcs is not None else ["sin", "cos", "exp", "log"]

        # Multi-expanders
        self.multi_expander_params = multi_expander_params or [
            {
                "pop_count": pop_count,
                "usable_funcs": _usable_funcs,
                "grow_params": grow_params or {},
            }
        ]
        self.multi_expanders: List[Dict] = []

        # Data
        self.X_train: np.ndarray | None = None
        self.Y_train: np.ndarray | None = None
        self.X_peek: np.ndarray | None = None
        self.Y_peek: np.ndarray | None = None
        self.eval_npts: int = 0

        # Search state
        self.curr_iter: int = -1
        self.models: List[SearchModel] = []
        self.hmap: Dict[int, SearchModel] = {}
        self.final: List[SearchModel] = []
        self.final_paretos: Optional[List[List[SearchModel]]] = None
        self.nsga2_peek: List[SearchModel] = []
        self.start_time: float = 0.0
        self.curr_time: float = 0.0
        self.peekd_models: int = 0
        self.evald_models: int = 0
        self.peek_nfev: int = 0
        self.eval_nfev: int = 0

    # ------------------------------------------------------------------
    # sklearn-style API
    # ------------------------------------------------------------------

    def fit(self, X_train, Y_train) -> "PGE":
        """Run the PGE search loop.

        Parameters
        ----------
        X_train:
            Training input data, shape ``(n_samples, n_features)``.
        Y_train:
            Training target data, shape ``(n_samples,)``.

        Returns
        -------
        PGE
            ``self`` for chaining.
        """
        self.start_time = time.time()
        self._phase_times = {}
        self._print_start_params(X_train, Y_train)
        t0 = time.time()
        self._set_data(X_train, Y_train)
        self._phase_times["data_setup"] = time.time() - t0
        t0 = time.time()
        self._preloop()
        self._phase_times["preloop"] = time.time() - t0
        t0 = time.time()
        self._loop(self.max_iter)
        self._phase_times["search_loop"] = time.time() - t0
        self._phase_times.update(self._loop_phase_times)
        self.finalize()
        return self

    def _print_start_params(self, X_train, Y_train) -> None:
        """Print search configuration and data summary at start."""
        X_train = np.asarray(X_train)
        Y_train = np.asarray(Y_train)
        print("PGE Search Parameters:")
        print(f"  variables:       {self.vars}")
        print(f"  max_iter:        {self.max_iter}")
        print(f"  pop_count:       {self.pop_count}")
        print(f"  peek_count:      {self.peek_count}")
        print(f"  peek_fraction:   {self.peek_fraction}")
        print(f"  min_size:        {self.min_size}")
        print(f"  max_size:        {self.max_size}")
        print(f"  max_power:       {self.max_power}")
        print(f"  algebra_methods: {self.algebra_methods}")
        print(f"  err_method:      {self.err_method}")
        print(f"  fitness_params:  {self.fitness_func_params}")
        print(f"  num_expanders:   {len(self.multi_expander_params)}")
        for i, p in enumerate(self.multi_expander_params):
            print(
                f"    expander {i}:  pop_count={p.get('pop_count', self.pop_count)}, funcs={p.get('usable_funcs', [])}"
            )
        print(f"  training data:   {X_train.shape[0]} samples, {X_train.shape[1]} features")
        print(f"  Y range:         [{float(Y_train.min()):.4f}, {float(Y_train.max()):.4f}]")
        print()

    # ------------------------------------------------------------------
    # Data setup
    # ------------------------------------------------------------------

    def _set_data(self, X_train, Y_train) -> None:
        """Set training and peek data."""
        self.X_train = np.asarray(X_train, dtype=np.float64)
        self.Y_train = np.asarray(Y_train, dtype=np.float64)
        assert self.Y_train is not None
        self.eval_npts = len(self.Y_train)

        # Sample peek data
        if self.peek_fraction > 0 and self.peek_fraction < 1:
            self.peek_npts = max(1, int(self.peek_fraction * self.eval_npts))
            pos = np.random.choice(self.eval_npts, self.peek_npts, replace=False)
            assert self.X_train is not None
            assert self.Y_train is not None
            self.X_peek = self.X_train[pos, :]
            self.Y_peek = self.Y_train[pos]
        else:
            self.peek_npts = self.eval_npts
            assert self.X_train is not None
            assert self.Y_train is not None
            self.X_peek = self.X_train
            self.Y_peek = self.Y_train

    # ------------------------------------------------------------------
    # Preloop: generate, filter, memoize, algebra, evaluate initial population
    # ------------------------------------------------------------------

    def _preloop(self) -> None:
        """Generate and evaluate the initial population."""
        preloop_start = time.time()

        # Build expanders
        self.multi_expanders = []
        for p in self.multi_expander_params:
            funcs = map_names_to_funcs(p.get("usable_funcs", ["sin", "cos", "exp", "log"]))
            grow_params = p.get("grow_params", {})
            grower = Grower(self.vars, funcs, **grow_params)
            self.multi_expanders.append(
                {
                    "pop_count": p.get("pop_count", self.pop_count),
                    "nsga2_list": [],
                    "grower": grower,
                }
            )

        # Generate first expressions
        t0 = time.time()
        initial_exprs = self.multi_expanders[0]["grower"].first_exprs()
        self._assign_iter_id(initial_exprs)
        preloop_grow_dur = time.time() - t0

        # Process-filter pipeline
        t0 = time.time()
        filter_in = len(initial_exprs)
        filtered, filter_breakdown = filter_models_with_stats(initial_exprs, self._get_default_filters())
        filter_out = len(filtered)
        preloop_filter_dur = time.time() - t0

        # Memoize filtered
        t0 = time.time()
        unique, preloop_memo_dups = self._memoize_models(filtered)
        preloop_memo_dur = time.time() - t0

        # Algebra pass
        t0 = time.time()
        algebrad: List[SearchModel] = []
        algebrad_unique: List[SearchModel] = []
        preloop_method_times: list[dict] = []
        if self.algebra_methods:
            for meth in self.algebra_methods:
                m0 = time.time()
                algebrad = self._algebra_models(unique)
                self._assign_iter_id(algebrad)
                algebrad_filtered = filter_models(algebrad, self._get_default_filters())
                algebrad_unique, _ = self._memoize_models(algebrad_filtered)
                method_dur = time.time() - m0
                preloop_method_times.append(
                    {
                        "method": meth,
                        "time": method_dur,
                        "in_count": len(unique),
                        "out_count": len(algebrad_unique),
                    }
                )
        preloop_algebra_dur = time.time() - t0

        # Combine algebra and non-algebra models
        candidates = algebrad_unique + unique

        # Peek evaluation
        preloop_peek_dur = 0.0
        if self.peek_fraction == 0:
            to_eval = candidates
        else:
            t0 = time.time()
            evaluated = self._eval_models(candidates, peek=True)
            self._peek_push_models(evaluated)
            preloop_peek_dur = time.time() - t0
            to_eval = self._peek_pop() + self._peek_pop()  # double pop first time

        # Full evaluation
        t0 = time.time()
        evaluated = self._eval_models(to_eval)
        preloop_full_dur = time.time() - t0

        # Push to final and population
        self._final_push(evaluated)
        self.multi_expanders[0]["nsga2_list"].extend(evaluated)

        self.curr_time = time.time()

        # Track as iteration 0 for preloop
        self._loop_stage_times.grow.append(preloop_grow_dur)
        self._loop_stage_times.filter.append(preloop_filter_dur)
        self._loop_stage_times.algebra.append(preloop_algebra_dur)
        self._loop_stage_times.peek_eval.append(preloop_peek_dur)
        self._loop_stage_times.full_eval.append(preloop_full_dur)
        self._loop_stage_times.memo_time.append(preloop_memo_dur)
        self._loop_stage_times.memo_dups.append(preloop_memo_dups)
        self._loop_sub_times.grow_expander.append([])
        self._loop_sub_times.grow_operators.append([])
        self._loop_sub_times.grow_operator_times.append([])
        self._loop_sub_times.filter_in.append(filter_in)
        self._loop_sub_times.filter_out.append(filter_out)
        self._loop_sub_times.filter_breakdown.append(filter_breakdown)
        self._loop_sub_times.filter_time.append(preloop_filter_dur)
        self._loop_sub_times.algebra_methods.append(preloop_method_times)
        self._loop_sub_times.memoize.append(
            MemoizeRecord(in_count=filter_in, unique=filter_out, duplicates=preloop_memo_dups)
        )

        # Print preloop line
        best = f"{self._current_best_score():.6g}" if self._current_best_score() != float("inf") else "N/A"
        preloop_dur = time.time() - self.start_time - (self._phase_times.get("data_setup", 0))
        print(
            f"  preloop  | "
            f"elapsed {time.time() - self.start_time:7.2f}s | "
            f"iter {preloop_dur:.3f}s | "
            f"grown {len(initial_exprs):4d} | "
            f"filter {filter_in:6d}->{filter_out:6d} | "
            f"alg {len(algebrad):3d}+{len(algebrad_unique):3d} | "
            f"memo {preloop_memo_dups:6d} | "
            f"peek {0:6d} | "
            f"full {len(evaluated):6d} | "
            f"best {best} | "
            f"pop {len(self.multi_expanders[0]['nsga2_list']):4d}"
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _loop(self, iterations: int) -> None:
        """Run the main search loop for *iterations* iterations."""
        self._loop_phase_times: Dict[str, float] = {}

        for I in range(iterations):
            self.curr_iter = I
            iter_start = time.time()

            t0 = time.time()

            # Multi-expander and grow — track per-expander sub-timing and per-operator stats
            expanded: List[SearchModel] = []
            prev: List[SearchModel] = []
            expander_times: list[dict] = []
            operator_stats: list[list[GrowOperatorStats]] = []
            operator_times: list[list[GrowOperatorTime]] = []
            for i, expander in enumerate(self.multi_expanders):
                nsga2_list = expander["nsga2_list"]
                pop_count = expander["pop_count"]
                grower = expander["grower"]

                if prev:
                    nsga2_list.extend(prev)
                    prev = []

                popped, nsga2_list = self._heap_pop(nsga2_list, pop_count)
                expander["nsga2_list"] = nsga2_list

                t1 = time.time()
                parent_ops: list[GrowOperatorStats] = []
                parent_op_times: list[GrowOperatorTime] = []
                for p in popped:
                    children, ops, op_times = grower.grow_with_stats(p)
                    expanded.extend(children)
                    parent_ops.extend(ops)
                    parent_op_times.extend(op_times)
                grower_time = time.time() - t1

                expander_times.append(
                    {"expander": i, "time": grower_time, "popped": len(popped), "children": len(expanded)}
                )
                operator_stats.append(parent_ops)
                operator_times.append(parent_op_times)

            grow_dur = time.time() - t0
            self._loop_phase_times["grow"] = self._loop_phase_times.get("grow", 0) + grow_dur
            self._loop_stage_times.grow.append(grow_dur)
            self._loop_sub_times.grow_expander.append(expander_times)
            self._loop_sub_times.grow_operators.append(operator_stats)
            self._loop_sub_times.grow_operator_times.append(operator_times)

            if not expanded:
                self._record_iteration_progress(time.time(), 0.0, I, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
                continue

            t0 = time.time()

            self._assign_iter_id(expanded)

            # Filter — track in/out counts, per-filter breakdown, and timing
            filter_in = len(expanded)
            filter_t0 = time.time()
            filtered, filter_breakdown = filter_models_with_stats(expanded, self._get_default_filters())
            filter_dur = time.time() - filter_t0
            filter_out = len(filtered)

            self._loop_phase_times["filter"] = self._loop_phase_times.get("filter", 0) + filter_dur
            self._loop_stage_times.filter.append(filter_dur)
            self._loop_sub_times.filter_in.append(filter_in)
            self._loop_sub_times.filter_out.append(filter_out)
            self._loop_sub_times.filter_breakdown.append(filter_breakdown)
            self._loop_sub_times.filter_time.append(filter_dur)

            # Size tracking
            avg_grown_size = sum(m.size() for m in expanded) / len(expanded) if expanded else 0.0
            avg_filtered_size = sum(m.size() for m in filtered) / len(filtered) if filtered else 0.0

            t0 = time.time()

            # Algebra — track per-method sub-timing
            algebrad: List[SearchModel] = []
            algebrad_unique: List[SearchModel] = []
            method_times: list[dict] = []
            unique: List[SearchModel] = []
            if self.algebra_methods:
                unique, _ = self._memoize_models(filtered)
                for meth in self.algebra_methods:
                    m0 = time.time()
                    algebrad = self._algebra_models(unique)
                    self._assign_iter_id(algebrad)
                    algebrad_filtered = filter_models(algebrad, self._get_default_filters())
                    algebrad_unique, _ = self._memoize_models(algebrad_filtered)
                    method_dur = time.time() - m0
                    method_times.append(
                        {
                            "method": meth,
                            "time": method_dur,
                            "in_count": len(unique),
                            "out_count": len(algebrad_unique),
                        }
                    )

            algebra_dur = time.time() - t0
            self._loop_phase_times["algebra"] = self._loop_phase_times.get("algebra", 0) + algebra_dur
            self._loop_stage_times.algebra.append(algebra_dur)
            self._loop_sub_times.algebra_methods.append(method_times)

            t0 = time.time()

            # Memoize non-algebra models
            unique, iter_memo_dups = self._memoize_models(filtered)
            memo_dur = time.time() - t0
            self._loop_phase_times["memo"] = self._loop_phase_times.get("memo", 0) + memo_dur

            # Combine algebra and non-algebra models
            candidates = algebrad_unique + unique

            # Peek evaluate
            to_eval: List[SearchModel] = []
            peek_count = 0
            if self.peek_fraction == 0:
                to_eval = candidates
            else:
                peek_evaluated = self._eval_models(candidates, peek=True)
                peek_count = len(peek_evaluated)
                self._peek_push_models(peek_evaluated)
                to_eval = self._peek_pop()

            peek_eval_dur = time.time() - t0
            self._loop_phase_times["peek_eval"] = self._loop_phase_times.get("peek_eval", 0) + peek_eval_dur
            self._loop_stage_times.peek_eval.append(peek_eval_dur)

            t0 = time.time()

            # Full evaluate
            full_count = 0
            if to_eval:
                full_evaluated = self._eval_models(to_eval)
                full_count = len(full_evaluated)
                self._final_push(full_evaluated)
                self.multi_expanders[0]["nsga2_list"].extend(full_evaluated)

            full_eval_dur = time.time() - t0
            self._loop_phase_times["full_eval"] = self._loop_phase_times.get("full_eval", 0) + full_eval_dur
            self._loop_stage_times.full_eval.append(full_eval_dur)

            self.curr_time = time.time()
            iter_dur = time.time() - iter_start
            self._iteration_times.append(iter_dur)

            # Progress stats
            elapsed = time.time() - self.start_time
            best_score = self._current_best_score()
            pop_size = len(self.multi_expanders[0]["nsga2_list"]) if self.multi_expanders else 0
            avg_algebra_size = sum(m.size() for m in algebrad_unique) / len(algebrad_unique) if algebrad_unique else 0.0

            # Count memo duplicates this iteration
            iter_memo_dups = (
                sum(
                    r.duplicates
                    for r in self._loop_sub_times.memoize[-len(self._loop_sub_times.algebra_methods) - 2 :]
                    if r.duplicates > 0
                )
                if self._loop_sub_times.memoize
                else 0
            )
            self._loop_stage_times.memo_time.append(memo_dur)
            self._loop_stage_times.memo_dups.append(iter_memo_dups)

            progress = IterationProgress(
                iteration=I,
                elapsed=elapsed,
                iter_dur=iter_dur,
                grown=len(expanded),
                filtered_in=filter_in,
                filtered_out=filter_out,
                algebrad=len(algebrad),
                algebrad_unique=len(algebrad_unique),
                peek_evaluated=peek_count,
                fully_evaluated=full_count,
                best_score=best_score,
                population_size=pop_size,
                avg_grown_size=avg_grown_size,
                avg_filtered_size=avg_filtered_size,
                avg_algebra_size=avg_algebra_size,
                memoized=iter_memo_dups,
            )
            self._iteration_progress.append(progress)
            self._print_iteration_progress(progress)

    def _current_best_score(self) -> float:
        """Return the best (lowest) score among all evaluated models."""
        best = float("inf")
        for m in self.final:
            if m.score is not None and m.score < best:
                best = m.score
        return best

    def _record_iteration_progress(
        self,
        iter_start: float,
        iter_dur: float,
        I: int,
        grown: int,
        filtered_in: int,
        filtered_out: int,
        algebrad: int,
        algebrad_unique: int,
        peek_evaluated: int,
        fully_evaluated: int,
        best_score: float = float("inf"),
        population_size: int = 0,
        avg_grown_size: float = 0.0,
        avg_filtered_size: float = 0.0,
        avg_algebra_size: float = 0.0,
    ) -> None:
        """Record an empty iteration progress (used when expanded is empty)."""
        self._iteration_progress.append(
            IterationProgress(
                iteration=I,
                elapsed=time.time() - self.start_time,
                iter_dur=iter_dur,
                grown=grown,
                filtered_in=filtered_in,
                filtered_out=filtered_out,
                algebrad=algebrad,
                algebrad_unique=algebrad_unique,
                peek_evaluated=peek_evaluated,
                fully_evaluated=fully_evaluated,
                best_score=best_score,
                population_size=population_size,
                avg_grown_size=avg_grown_size,
                avg_filtered_size=avg_filtered_size,
                avg_algebra_size=avg_algebra_size,
            )
        )

    def _print_iteration_progress(self, progress: IterationProgress) -> None:
        """Print a one-line summary for each iteration."""
        best = f"{progress.best_score:.6g}" if progress.best_score != float("inf") else "N/A"
        print(
            f"  iter {progress.iteration:3d} | "
            f"elapsed {progress.elapsed:7.2f}s | "
            f"iter {progress.iter_dur:.3f}s | "
            f"grown {progress.grown:4d} | "
            f"filter {progress.filtered_in:6d}->{progress.filtered_out:6d} | "
            f"alg {progress.algebrad:3d}+{progress.algebrad_unique:3d} | "
            f"memo {progress.memoized:6d} | "
            f"peek {progress.peek_evaluated:6d} | "
            f"full {progress.fully_evaluated:6d} | "
            f"best {best} | "
            f"pop {progress.population_size:4d}"
        )

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    def _eval_models(self, models: List[SearchModel], peek: bool = False) -> List[SearchModel]:
        """Fit and evaluate a list of models.

        Models that fail during evaluation are removed from the returned list.

        Parameters
        ----------
        models:
            Models to evaluate.
        peek:
            If ``True``, store results in peek attributes.

        Returns
        -------
        list[SearchModel]
            Models that were successfully evaluated.
        """
        succeeded = []
        for modl in models:
            if modl.errored:
                continue

            # Build JAX model lazily if not yet created
            if modl.jax_model is None:
                t0 = time.time()
                try:
                    modl.build_jax_model()
                except Exception:
                    modl.errored = True
                    continue
                modl.timings.build_time = time.time() - t0

            if modl.jax_model is None:
                modl.errored = True
                continue

            try:
                if self.X_peek is None or self.Y_peek is None:
                    modl.errored = True
                    continue
                xs_inputs = [jnp.asarray(self.X_peek[:, i], dtype=jnp.float64) for i in range(self.X_peek.shape[1])]
                y_true = jnp.asarray(self.Y_peek, dtype=jnp.float64)

                t0 = time.time()
                fit_result = fit_model(modl.jax_model, y_true, *xs_inputs, max_iter=200)
                modl.timings.fit_time = time.time() - t0

                if not fit_result.success:
                    modl.errored = True
                    continue

                # Store fitted coefficients
                modl.jax_model.c_values = np.asarray(fit_result.coefficients)

                # Evaluate on full data for score
                if self.X_train is None or self.Y_train is None:
                    modl.errored = True
                    continue
                full_xs = [jnp.asarray(self.X_train[:, i], dtype=jnp.float64) for i in range(self.X_train.shape[1])]
                full_y = jnp.asarray(self.Y_train, dtype=jnp.float64)
                y_pred = modl.jax_model.jax_fun(fit_result.coefficients, *full_xs)

                t0 = time.time()
                eval_result = evaluate(modl.jax_model, full_y, y_pred)
                modl.timings.eval_time = time.time() - t0

                # Update model with results
                modl.score = eval_result.score
                modl.r2 = eval_result.r2
                modl.evar = eval_result.evar
                modl.aic = eval_result.aic
                modl.bic = eval_result.bic
                modl.chisqr = eval_result.chisqr
                modl.redchi = eval_result.redchi
                modl.mae = eval_result.mae
                modl.rmae = eval_result.rmae
                modl.eval_nfev += fit_result.nfev

                if peek:
                    modl.peek_score = eval_result.score
                    modl.peek_r2 = eval_result.r2
                    modl.peek_evar = eval_result.evar
                    modl.peek_aic = eval_result.aic
                    modl.peek_bic = eval_result.bic
                    modl.peek_redchi = eval_result.redchi
                    modl.peek_nfev = fit_result.nfev
                    self.peek_nfev += modl.peek_nfev
                    self.peekd_models += 1
                    modl.peeked = True
                else:
                    self.eval_nfev += modl.eval_nfev
                    self.evald_models += 1
                    modl.evaluated = True

                # Track parent improvement
                self._compute_improvements(modl)
                succeeded.append(modl)

            except Exception:
                modl.errored = True

        return succeeded

    def _compute_improvements(self, modl: SearchModel) -> None:
        """Compute improvement over parent model."""
        if modl.parent_id >= 0 and modl.parent_id < len(self.models):
            parent = self.models[modl.parent_id]
            if parent.score is not None and modl.score is not None:
                modl.improve_score = parent.score - modl.score
            if parent.r2 is not None and modl.r2 is not None:
                modl.improve_r2 = modl.r2 - parent.r2
            if parent.evar is not None and modl.evar is not None:
                modl.improve_evar = modl.evar - parent.evar
            if parent.aic is not None and modl.aic is not None:
                modl.improve_aic = parent.aic - modl.aic
            if parent.bic is not None and modl.bic is not None:
                modl.improve_bic = parent.bic - modl.bic
            if parent.redchi is not None and modl.redchi is not None:
                modl.improve_redchi = parent.redchi - modl.redchi
        else:
            # First generation — small negative improvement
            if modl.score is not None:
                modl.improve_score = -0.000001 * modl.score
            if modl.r2 is not None:
                modl.improve_r2 = -0.000001 * modl.r2
            if modl.evar is not None:
                modl.improve_evar = -0.000001 * modl.evar
            if modl.aic is not None:
                modl.improve_aic = -0.000001 * modl.aic
            if modl.bic is not None:
                modl.improve_bic = -0.000001 * modl.bic
            if modl.redchi is not None:
                modl.improve_redchi = -0.000001 * modl.redchi

    # ------------------------------------------------------------------
    # Heap operations
    # ------------------------------------------------------------------

    def _peek_push_models(self, models: List[SearchModel]) -> None:
        """Push peek-evaluated models to the peek heap."""
        ms = [m for m in models if m is not None and not m.errored and m.score is not None]
        self.nsga2_peek.extend(ms)
        for m in ms:
            m.peek_queued = True

    def _peek_pop(self) -> List[SearchModel]:
        """Pop best models from the peek heap for full evaluation."""
        popped, self.nsga2_peek = self._heap_pop(self.nsga2_peek, self.peek_count)
        for p in popped:
            p.peek_popped = True
        return popped

    def _final_push(self, models: List[SearchModel]) -> None:
        """Push fully evaluated models to the final list."""
        ms = [m for m in models if m is not None and not m.errored and m.score is not None]
        for p in ms:
            self.final.append(p)
            p.finalized = True

    def _heap_pop(self, heap_list: List[SearchModel], pop_count: int) -> Tuple[List[SearchModel], List[SearchModel]]:
        """Select *pop_count* best models from *heap_list* via NSGA-II.

        Returns
        -------
        tuple[list, list]
            ``(popped, remaining)``.
        """
        # Set fitness values
        self._compute_fitness(heap_list)

        popped = selNSGA2(heap_list, pop_count, nd="log")

        remaining = [m for m in heap_list if m not in popped]

        return popped, remaining

    def _compute_fitness(self, models: List[SearchModel]) -> None:
        """Compute fitness values for a list of models.

        Uses the configured fitness function parameters.
        """
        # Build fitness calculator on first call
        if self.fitness_calc is None and models:
            from pge_jax.fitness_funcs import build_fitness_calc

            self.fitness_calc = build_fitness_calc(self.fitness_func_params)

        if self.fitness_calc is not None:
            self.fitness_calc(models)

    # ------------------------------------------------------------------
    # Filtering, memoization, algebra
    # ------------------------------------------------------------------

    def _get_default_filters(self) -> list:
        """Return the default filter list with configured max_size."""
        from pge_jax.filters import (
            filter_has_big_pow,
            filter_has_coeff_pow,
            filter_has_int_coeff,
            filter_just_C,
            filter_no_C,
            filter_too_big,
        )

        return [
            lambda m, e: filter_too_big(m, e, big=self.max_size),
            filter_has_int_coeff,
            lambda m, e: filter_has_big_pow(m, e, big=self.max_power),
            filter_just_C,
            filter_no_C,
            filter_has_coeff_pow,
        ]

    def _memoize_models(self, models: List[SearchModel]) -> Tuple[List[SearchModel], int]:
        """Deduplicate models and assign IDs.

        Parameters
        ----------
        models:
            Models to deduplicate.

        Returns
        -------
        tuple[list[SearchModel], int]
            ``(unique_models, rejected_count)``.
        """
        unique = []
        rejected = 0
        for m in models:
            h = m.orig.__hash__()
            r = self.hmap.get(h, None)
            if r is None:
                m.id = len(self.models)
                self.models.append(m)
                self.hmap[h] = m
                m.memoized = True
                unique.append(m)
            else:
                rejected += 1
        self._loop_sub_times.memoize.append(
            MemoizeRecord(in_count=len(models), unique=len(unique), duplicates=rejected)
        )
        return unique, rejected

    def _algebra_models(self, models: List[SearchModel]) -> List[SearchModel]:
        """Apply algebraic manipulation to models.

        Returns
        -------
        list[SearchModel]
            New models whose expressions changed.
        """
        alges: List[SearchModel] = []
        for modl in models:
            for meth in self.algebra_methods:
                try:
                    manipd, err = manip_model(modl, meth)
                    if err is not None:
                        if err == "same":
                            continue
                        continue
                    if manipd is not None:
                        manipd.parent_id = modl.id
                        manipd.gen_relation = meth
                        alges.append(manipd)
                except Exception:
                    continue
            modl.algebrad = True
        return alges

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _assign_iter_id(self, expr_list: List[SearchModel]) -> None:
        """Set ``iter_id`` for a list of models."""
        for e in expr_list:
            e.iter_id = self.curr_iter

    def print_best(self, count: int = 16) -> None:
        """Print the best models from the final list."""
        if not self.final:
            print("No models evaluated yet.")
            return

        # Compute fitness
        self._compute_fitness(self.final)

        # Sort into Pareto fronts
        fronts = sortLogNondominated(self.final, count)

        print(f"Best {min(count, len(self.final))} of {len(self.final)}")
        print("-" * 100)

        cnt = 0
        for front in fronts:
            for m in front:
                if cnt >= count:
                    break
                cnt += 1
                m.pretty_expr()
                print(f"  {m}")
            if cnt >= count:
                break

        print("-" * 100)

    # ------------------------------------------------------------------
    # Finalize
    # ------------------------------------------------------------------

    def finalize(self, n_solutions: int = 32, n: int = 2) -> None:
        """Generate final Pareto fronts and print results.

        Parameters
        ----------
        n_solutions:
            Maximum number of rows to print in the final results table.
        n:
            Exponent threshold for number formatting.  Values with
            exponent in [-n, n] use plain format; others use E notation.
        """
        print("\nFinalizing\n")

        # Combine final + remaining population
        combined = self.final + [m for exp in self.multi_expanders for m in exp["nsga2_list"]]

        # Deduplicate by structural expression equality
        seen_exprs: set[str] = set()
        final: List[SearchModel] = []
        for m in combined:
            key = str(sympy.sympify(m.orig))
            if key not in seen_exprs:
                seen_exprs.add(key)
                final.append(m)

        # Compute fitness
        self._compute_fitness(final)

        # Generate Pareto fronts
        self.final_paretos = sortLogNondominated(final, len(final))

        # Print everything
        self.print_results(n_solutions, n)
        self.print_summary()
        self.print_timing()

    # ------------------------------------------------------------------
    # Print results table
    # ------------------------------------------------------------------

    def print_results(self, count: int = 32, n: int = 2) -> None:
        """Print the Pareto-front results table.

        Parameters
        ----------
        count:
            Maximum number of rows to print.
        n:
            Exponent threshold for number formatting.
        """
        if not self.final_paretos:
            print("No results yet. Call finalize() first.")
            return

        num_headers = [
            "id",
            "iter",
            "parent",
            "sz",
            "psz",
            "jsz",
            "jpsz",
            "score",
            "r2",
            "evar",
            "aic",
            "bic",
            "redchi",
        ]

        rows: List[tuple] = []
        exprs: List[str] = []
        for front in self.final_paretos:
            for m in front:
                if len(rows) >= count:
                    break
                m.pretty_expr(n)
                rows.append(
                    (
                        m.id,
                        m.iter_id,
                        m.parent_id,
                        m.size(),
                        m.psz,
                        m.jsz,
                        m.jpsz,
                        m.score,
                        m.r2,
                        m.evar,
                        m.aic,
                        m.bic,
                        m.redchi,
                    )
                )
                exprs.append(m.pretty or "")
            if len(rows) >= count:
                break

        # Format all values and compute column widths
        all_formatted: List[List[str]] = []
        for row in rows:
            formatted = []
            for i, val in enumerate(row):
                if i < 7:
                    formatted.append(str(val))
                else:
                    formatted.append(SearchModel.fmt(val, n))
            all_formatted.append(formatted)

        widths = [len(h) for h in num_headers]
        for formatted in all_formatted:
            for i, f in enumerate(formatted):
                widths[i] = max(widths[i], len(f))
        widths = [w + 1 for w in widths]

        print(f"Best {min(count, len(rows))} of {len(rows)}")
        print("-" * 100)
        print("  ".join(f"{h:>{w}}" for h, w in zip(num_headers, widths)) + "  |  expr")
        print("  ".join("-" * w for w in widths) + "  |  " + "-" * 20)
        for formatted, expr in zip(all_formatted, exprs):
            print("  ".join(f"{f:>{w}}" for f, w in zip(formatted, widths)) + "  |  " + expr)
        print("-" * 100)

    # ------------------------------------------------------------------
    # Print summary (model counts, evaluation stats)
    # ------------------------------------------------------------------

    def print_summary(self) -> None:
        """Print model counts and evaluation statistics."""
        total_memo_dups = sum(r.duplicates for r in self._loop_sub_times.memoize)
        print(f"\nnum peekd models:  {self.peekd_models}")
        print(f"num evald models:  {self.evald_models}")
        print(f"num memoized:      {total_memo_dups}")
        print(f"num peek evals:    {self.peek_nfev} ({self.peek_nfev * self.peek_npts} point-evals)")
        print(f"num eval evals:    {self.eval_nfev} ({self.eval_nfev * self.eval_npts} point-evals)")
        print(f"num total evals:   {self.peek_nfev * self.peek_npts + self.eval_nfev * self.eval_npts}")
        print(f"  peek_npts:       {self.peek_npts}")
        print(f"  eval_npts:       {self.eval_npts}")

    # ------------------------------------------------------------------
    # Print timing
    # ------------------------------------------------------------------

    def print_timing(self) -> None:
        """Print phase times, iteration stats, and per-model stage times."""
        runtime = time.time() - self.start_time
        loop_sub = {
            k: v
            for k, v in self._phase_times.items()
            if k in ("grow", "filter", "algebra", "memo", "peek_eval", "full_eval")
        }
        if loop_sub:
            self._phase_times["search_loop"] = sum(loop_sub.values())
            for k in loop_sub:
                del self._phase_times[k]
        self._phase_times["finalize"] = runtime - sum(self._phase_times.values())

        print(f"\nTOTAL RUN TIME: {runtime:.4f} seconds")
        print("=" * 80)
        print("\nPhase Times:")
        for phase, dur in self._phase_times.items():
            pct = dur / runtime * 100 if runtime > 0 else 0
            print(f"  {phase:12s}   {dur:8.3f}s  ({pct:5.1f}%)")
        if loop_sub:
            print("\n  Search Loop Breakdown:")
            self._print_search_loop_tree(loop_sub)

        # Per-iteration timing stats (min / max / avg)
        if self._iteration_times:
            print("\n  Iteration Times:")
            self._print_timing_stats("    ", "total", self._iteration_times)

        # Per-iteration stage timing stats — tree-style with subtasks
        self._print_stage_tree()

        # Per-iteration sub-stage details
        if self._loop_sub_times.grow_operators:
            print("\n  Grow Per-Operator:")
            self._print_grow_operator_details()

        if self._loop_sub_times.filter_in:
            print("\n  Filter Breakdown:")
            self._print_filter_details()

        if self._loop_sub_times.algebra_methods:
            print("\n  Algebra Per-Method:")
            self._print_algebra_method_details()

        if self._loop_sub_times.memoize:
            print("\n  Memoize Breakdown:")
            for i, rec in enumerate(self._loop_sub_times.memoize):
                print(f"    memo {i}:  in={rec.in_count}  unique={rec.unique}  duplicates={rec.duplicates}")

        # Per-model timing stats aggregated from SearchModel.timings
        build_times = [m.timings.build_time for m in self.final if m.timings.build_time > 0]
        fit_times = [m.timings.fit_time for m in self.final if m.timings.fit_time > 0]
        eval_times = [m.timings.eval_time for m in self.final if m.timings.eval_time > 0]

        if build_times or fit_times or eval_times:
            print("\n  Per-Model Stage Times:")
            if build_times:
                self._print_timing_stats("    ", "build", build_times)
            if fit_times:
                self._print_timing_stats("    ", "fit", fit_times)
            if eval_times:
                self._print_timing_stats("    ", "eval", eval_times)

    def _print_search_loop_tree(self, loop_sub: Dict[str, float]) -> None:
        """Print search loop breakdown with indented subtrees."""
        search_loop_total = self._phase_times["search_loop"]

        # Build subtask maps
        grow_sub: Dict[str, float] = {}
        if self._loop_sub_times.grow_operator_times:
            for it, parent_times_list in enumerate(self._loop_sub_times.grow_operator_times):
                for parent_times in parent_times_list:
                    for ot in parent_times:
                        grow_sub[ot.operator] = grow_sub.get(ot.operator, 0) + ot.time

        algebra_sub: Dict[str, float] = {}
        if self._loop_sub_times.algebra_methods:
            for method_list in self._loop_sub_times.algebra_methods:
                for m in method_list:
                    algebra_sub[m["method"]] = algebra_sub.get(m["method"], 0) + m["time"]

        # Print each stage with its subtree
        stages = [
            ("grow", loop_sub.get("grow", 0)),
            ("filter", loop_sub.get("filter", 0)),
            ("algebra", loop_sub.get("algebra", 0)),
            ("memo", loop_sub.get("memo", 0)),
            ("peek_eval", loop_sub.get("peek_eval", 0)),
            ("full_eval", loop_sub.get("full_eval", 0)),
        ]

        for stage_name, stage_dur in stages:
            if stage_dur == 0:
                continue
            pct_of_loop = stage_dur / search_loop_total * 100 if search_loop_total > 0 else 0
            print(f"    {stage_name:12s} {stage_dur:8.3f}s  ({pct_of_loop:5.1f}% of search_loop)")

            # Print subtree
            if stage_name == "grow" and grow_sub:
                grow_total = sum(grow_sub.values())
                for op_name in sorted(grow_sub):
                    op_dur = grow_sub[op_name]
                    pct_of_stage = op_dur / grow_total * 100 if grow_total > 0 else 0
                    print(f"      {op_name:12s} {op_dur:8.3f}s  ({pct_of_stage:5.1f}% of grow)")
            elif stage_name == "algebra" and algebra_sub:
                alg_total = sum(algebra_sub.values())
                for meth_name in sorted(algebra_sub):
                    meth_dur = algebra_sub[meth_name]
                    pct_of_stage = meth_dur / alg_total * 100 if alg_total > 0 else 0
                    print(f"      {meth_name:12s} {meth_dur:8.3f}s  ({pct_of_stage:5.1f}% of algebra)")

    def _print_grow_operator_details(self) -> None:
        """Print per-operator grow stats aggregated per iteration (one line per operator)."""
        for it, parent_ops_list in enumerate(self._loop_sub_times.grow_operators):
            # Aggregate: sum raw/unique/children across all parents, weighted avg for size
            agg: dict[str, dict] = {}
            for ops in parent_ops_list:
                for op in ops:
                    if op.operator not in agg:
                        agg[op.operator] = {"raw": 0, "unique": 0, "children": 0, "total_size": 0.0, "count": 0}
                    agg[op.operator]["raw"] += op.raw_exprs
                    agg[op.operator]["unique"] += op.unique_exprs
                    agg[op.operator]["children"] += op.children
                    agg[op.operator]["total_size"] += op.avg_size * op.children
                    agg[op.operator]["count"] += 1
            print(f"    iter {it}:")
            for op_name in sorted(agg):
                a = agg[op_name]
                avg_sz = a["total_size"] / max(a["children"], 1)
                print(
                    f"      {op_name:12s}:  raw={a['raw']:5d}  unique={a['unique']:5d}  "
                    f"children={a['children']:5d}  avg_size={avg_sz:.1f}"
                )

    def _print_filter_details(self) -> None:
        """Print filter in/out counts and per-filter rejection breakdown per iteration."""
        for it, (fin, fout) in enumerate(zip(self._loop_sub_times.filter_in, self._loop_sub_times.filter_out)):
            print(f"    iter {it}:  in={fin}  out={fout}  (kept {fout / max(fin, 1) * 100:.0f}%)")
            if it < len(self._loop_sub_times.filter_breakdown):
                breakdown = self._loop_sub_times.filter_breakdown[it]
                for fb in breakdown:
                    if fb.rejections > 0:
                        print(f"      {fb.filter_name:25s}:  {fb.rejections:6d} rejections")

    def _print_stage_tree(self) -> None:
        """Print per-iteration stage timing in a tree format with subtasks."""
        n_iters = len(self._loop_stage_times.grow)
        if n_iters == 0:
            return

        for it in range(n_iters):
            grow_t = self._loop_stage_times.grow[it] if it < len(self._loop_stage_times.grow) else 0
            filter_t = self._loop_stage_times.filter[it] if it < len(self._loop_stage_times.filter) else 0
            algebra_t = self._loop_stage_times.algebra[it] if it < len(self._loop_stage_times.algebra) else 0
            peek_t = self._loop_stage_times.peek_eval[it] if it < len(self._loop_stage_times.peek_eval) else 0
            full_t = self._loop_stage_times.full_eval[it] if it < len(self._loop_stage_times.full_eval) else 0

            print(f"    iter {it}:")
            print(f"      grow          {grow_t:.4f}s")
            # Show grow subtasks
            if it < len(self._loop_sub_times.grow_operator_times):
                grow_agg: Dict[str, float] = {}
                for parent_times in self._loop_sub_times.grow_operator_times[it]:
                    for ot in parent_times:
                        grow_agg[ot.operator] = grow_agg.get(ot.operator, 0) + ot.time
                grow_total = sum(grow_agg.values())
                for op_name in sorted(grow_agg):
                    op_t = grow_agg[op_name]
                    pct = op_t / grow_total * 100 if grow_total > 0 else 0
                    print(f"        {op_name:12s}  {op_t:.4f}s  ({pct:5.1f}%)")
            print(f"      filter        {filter_t:.4f}s")
            print(f"      algebra       {algebra_t:.4f}s")
            # Show algebra subtasks
            if it < len(self._loop_sub_times.algebra_methods):
                alg_total = sum(m["time"] for m in self._loop_sub_times.algebra_methods[it])
                for method_info in self._loop_sub_times.algebra_methods[it]:
                    meth_t = method_info["time"]
                    pct = meth_t / alg_total * 100 if alg_total > 0 else 0
                    print(f"        {method_info['method']:12s}  {meth_t:.4f}s  ({pct:5.1f}%)")
            print(
                f"      memo          {self._loop_stage_times.memo_time[it]:.4f}s ({self._loop_stage_times.memo_dups[it]} dups)"
            )
            print(f"      peek_eval     {peek_t:.4f}s")
            print(f"      full_eval     {full_t:.4f}s")

    def _print_algebra_method_details(self) -> None:
        """Print per-algebra-method timing breakdown per iteration."""
        for it, method_list in enumerate(self._loop_sub_times.algebra_methods):
            print(f"    iter {it}:")
            for m in method_list:
                print(f"      {m['method']:10s}:  {m['time']:.4f}s  in={m['in_count']}  out={m['out_count']}")

    def _print_timing_stats(self, prefix: str, label: str, times: List[float]) -> None:
        """Print min / max / avg timing stats for a list of durations.

        Parameters
        ----------
        prefix:
            String prefix for each output line.
        label:
            Human-readable label for the timing category.
        times:
            List of durations in seconds.
        """
        n = len(times)
        if n == 0:
            return
        mn = min(times)
        mx = max(times)
        avg = sum(times) / n
        total = sum(times)
        print(f"{prefix}{label:12s}  count={n:6d}  min={mn:.6f}s  max={mx:.6f}s  avg={avg:.6f}s  total={total:.3f}s")

    def get_final_paretos(self) -> Optional[List[List[SearchModel]]]:
        """Return the final Pareto fronts.

        Returns
        -------
        list[list[SearchModel]] | None
            List of Pareto fronts, or ``None`` if not yet finalised.
        """
        return self.final_paretos

    def get_best_model(self) -> Optional[SearchModel]:
        """Return the single best model (lowest score across all Pareto fronts).

        Returns
        -------
        SearchModel | None
        """
        if not self.final_paretos:
            return None
        best = None
        best_score = float("inf")
        for front in self.final_paretos:
            for m in front:
                if m.score is not None and m.score < best_score:
                    best_score = m.score
                    best = m
        return best
