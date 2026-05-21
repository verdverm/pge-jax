"""PGE search loop — Prioritized Grammar Enumeration.

Orchestrates the full symbolic regression search: expression generation,
filtering, memoization, algebraic manipulation, progressive evaluation,
and multi-objective selection.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import jax.numpy as jnp
import numpy as np
import sympy
from tqdm import tqdm

from pge_jax.algebra import manip_model
from pge_jax.expand import Grower, map_names_to_funcs
from pge_jax.evaluate import fit_model, evaluate
from pge_jax.filters import filter_models
from pge_jax.memoize import Memoizer
from pge_jax.model import JAXModel
from pge_jax.search_model import SearchModel
from pge_jax.selection import (
    assignCrowdingDist,
    selNSGA2,
    sortLogNondominated,
)

if TYPE_CHECKING:
    from pge_jax.fitness_funcs import Callable

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
    peek_npts:
        Number of data points for peek evaluation (0 = skip peek).
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
        usable_funcs: List[str] = DEFAULT_FITNESS_PARAMS,
        max_iter: int = 100,
        pop_count: int = 3,
        peek_count: int = 6,
        peek_npts: int = 16,
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

        # Config
        self.max_iter: int = max_iter
        self.pop_count: int = pop_count
        self.peek_count: int = peek_count
        self.peek_npts: int = peek_npts
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

        # Multi-expanders
        self.multi_expander_params = multi_expander_params or [
            {
                "pop_count": pop_count,
                "usable_funcs": usable_funcs
                if usable_funcs != DEFAULT_FITNESS_PARAMS
                else ["sin", "cos", "exp", "log"],
                "grow_params": grow_params or {},
            }
        ]
        self.multi_expanders: List[Dict] = []

        # Data
        self.X_train = None
        self.Y_train = None
        self.X_peek = None
        self.Y_peek = None
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
        self._set_data(X_train, Y_train)
        self._preloop()
        self._loop(self.max_iter)
        self.finalize()
        return self

    # ------------------------------------------------------------------
    # Data setup
    # ------------------------------------------------------------------

    def _set_data(self, X_train, Y_train) -> None:
        """Set training and peek data."""
        self.X_train = np.asarray(X_train, dtype=np.float64)
        self.Y_train = np.asarray(Y_train, dtype=np.float64)
        self.eval_npts = len(self.Y_train)

        # Sample peek data
        if self.peek_npts > 0 and self.peek_npts < self.eval_npts:
            pos = np.random.choice(self.eval_npts, self.peek_npts, replace=False)
            self.X_peek = self.X_train[:, pos]
            self.Y_peek = self.Y_train[pos]
        else:
            self.X_peek = self.X_train
            self.Y_peek = self.Y_train

    # ------------------------------------------------------------------
    # Preloop: generate, filter, memoize, algebra, evaluate initial population
    # ------------------------------------------------------------------

    def _preloop(self) -> None:
        """Generate and evaluate the initial population."""
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
        first_exprs = self.multi_expanders[0]["grower"].first_exprs()
        self._assign_iter_id(first_exprs)

        # Filter
        to_memo = filter_models(first_exprs, self._get_default_filters())

        # Memoize
        to_alge = []
        if self.algebra_methods:
            to_alge = self._memoize_models(to_memo)

            # Algebra
            algebrad = self._algebra_models(to_alge)
            self._assign_iter_id(algebrad)

            # Filter + memoize algebra results
            to_memo = filter_models(algebrad, self._get_default_filters())
            to_memo = self._memoize_models(to_memo)

        to_peek = self._memoize_models(to_memo)

        # Combine algebra and non-algebra models
        to_peek = to_alge + to_peek

        # Evaluate
        if self.peek_npts == 0:
            to_eval = to_peek
        else:
            self._eval_models(to_peek, peek=True)
            self._peek_push_models(to_peek)
            to_eval = self._peek_pop() + self._peek_pop()  # double pop first time

        self._eval_models(to_eval)

        # Push to final and population
        self._final_push(to_eval)
        self.multi_expanders[0]["nsga2_list"].extend(to_eval)

        self.curr_time = time.time()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _loop(self, iterations: int) -> None:
        """Run the main search loop for *iterations* iterations."""
        for I in range(iterations):
            self.curr_iter = I

            # Multi-expand and grow
            expanded = []
            prev = []
            for i, expander in enumerate(self.multi_expanders):
                nsga2_list = expander["nsga2_list"]
                pop_count = expander["pop_count"]
                grower = expander["grower"]

                if prev:
                    nsga2_list.extend(prev)
                    prev = []

                popped, nsga2_list = self._heap_pop(nsga2_list, pop_count)
                expander["nsga2_list"] = nsga2_list

                # Grow each popped model
                for p in popped:
                    children = grower.grow(p)
                    expanded.extend(children)

            if not expanded:
                continue

            self._assign_iter_id(expanded)

            # Filter
            to_memo = filter_models(expanded, self._get_default_filters())

            # Algebra
            to_alge = []
            if self.algebra_methods:
                to_alge = self._memoize_models(to_memo)
                algebrad = self._algebra_models(to_alge)
                self._assign_iter_id(algebrad)
                to_memo = filter_models(algebrad, self._get_default_filters())
                to_memo = self._memoize_models(to_memo)

            to_peek = self._memoize_models(to_memo)
            to_peek = to_alge + to_peek

            # Peek evaluate
            if self.peek_npts == 0:
                to_eval = to_peek
            else:
                self._eval_models(to_peek, peek=True)
                self._peek_push_models(to_peek)
                to_eval = self._peek_pop()

            # Full evaluate
            if to_eval:
                self._eval_models(to_eval)
                self._final_push(to_eval)
                self.multi_expanders[0]["nsga2_list"].extend(to_eval)

            self.curr_time = time.time()

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    def _eval_models(self, models: List[SearchModel], peek: bool = False) -> None:
        """Fit and evaluate a list of models.

        Parameters
        ----------
        models:
            Models to evaluate.
        peek:
            If ``True``, store results in peek attributes.
        """
        for modl in models:
            if modl.errored or modl.jax_model is None:
                modl.errored = True
                continue

            try:
                if self.X_peek is None or self.Y_peek is None:
                    modl.errored = True
                    continue
                xs_inputs = [jnp.asarray(self.X_peek[:, i], dtype=jnp.float64) for i in range(self.X_peek.shape[1])]
                y_true = jnp.asarray(self.Y_peek, dtype=jnp.float64)

                fit_result = fit_model(modl.jax_model, y_true, *xs_inputs, max_iter=200)

                if not fit_result.success:
                    modl.errored = True
                    continue

                # Store fitted coefficients
                modl.jax_model.c_values = fit_result.coefficients

                # Evaluate on full data for score
                if self.X_train is None or self.Y_train is None:
                    modl.errored = True
                    continue
                full_xs = [jnp.asarray(self.X_train[:, i], dtype=jnp.float64) for i in range(self.X_train.shape[1])]
                full_y = jnp.asarray(self.Y_train, dtype=jnp.float64)
                y_pred = modl.jax_model.jax_fun(fit_result.coefficients, *full_xs)

                eval_result = evaluate(modl.jax_model, full_y, y_pred)

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

            except Exception:
                modl.errored = True

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
            filter_just_C,
            filter_has_big_pow,
            filter_has_coeff_pow,
            filter_has_int_coeff,
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

    def _memoize_models(self, models: List[SearchModel]) -> List[SearchModel]:
        """Deduplicate models and assign IDs.

        Returns
        -------
        list[SearchModel]
            Only newly inserted (unique) models.
        """
        unique = []
        for m in models:
            h = m.orig.__hash__()
            r = self.hmap.get(h, None)
            if r is None:
                m.id = len(self.models)
                self.models.append(m)
                self.hmap[h] = m
                m.memoized = True
                unique.append(m)
        return unique

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

    def finalize(self, nfronts: int = 4) -> None:
        """Generate final Pareto fronts and print results."""
        print("\nFinalizing\n")

        # Combine final + remaining population
        final = self.final + [m for exp in self.multi_expanders for m in exp["nsga2_list"]]

        # Compute fitness
        self._compute_fitness(final)

        # Generate Pareto fronts
        self.final_paretos = sortLogNondominated(final, len(final))

        # Print results
        print("Final Results")
        if self.final_paretos:
            print(str(self.final_paretos[0][0]))
            print("-" * 100)
            for i, front in enumerate(self.final_paretos[:nfronts]):
                for m in front:
                    print(f"  {m}")
                print("")

        print(f"\nnum peekd models:  {self.peekd_models}")
        print(f"num evald models:  {self.evald_models}")
        print(f"num peek evals:    {self.peek_nfev} ({self.peek_nfev * self.peek_npts} point-evals)")
        print(f"num eval evals:    {self.eval_nfev} ({self.eval_nfev * self.eval_npts} point-evals)")
        print(f"num total evals:   {self.peek_nfev * self.peek_npts + self.eval_nfev * self.eval_npts}")

        runtime = time.time() - self.start_time
        print(f"TOTAL RUN TIME: {runtime:.4f} seconds")

    def get_final_paretos(self) -> Optional[List[List[SearchModel]]]:
        """Return the final Pareto fronts.

        Returns
        -------
        list[list[SearchModel]] | None
            List of Pareto fronts, or ``None`` if not yet finalised.
        """
        return self.final_paretos

    def get_best_model(self) -> Optional[SearchModel]:
        """Return the single best model (lowest RMSE on the first Pareto front).

        Returns
        -------
        SearchModel | None
        """
        if not self.final_paretos or not self.final_paretos[0]:
            return None
        return self.final_paretos[0][0]
