"""Multi-objective fitness construction for the PGE search loop.

Dynamically constructs fitness evaluation from parameter lists like
``["normalize", "-(1)jpsz", "-score", "+bic"]``.  No DEAP dependency —
fitness values are stored directly as tuples on model objects.
"""

from __future__ import annotations

from typing import Callable, List, Tuple


def build_fitness_calc(params: List[str]):
    """Build a fitness calculator from parameter list.

    Parameters
    ----------
    params:
        List of fitness parameters.  Each parameter can be:
        - ``"normalize"`` — normalise objectives by L2 norm across population
        - ``"-score"`` or ``"+r2"`` — attribute name with minimisation /
          maximisation sign
        - ``"-(1)jpsz"`` — same with explicit weight

    Returns
    -------
    callable
        A function that accepts a list of models and sets their
        ``fitness_values`` and ``wvalues`` attributes.
    """
    norm = "normalize" in params
    if norm:
        params = [p for p in params if p != "normalize"]

    weights = build_fitness_weights(params)
    extractor = build_value_extractor(params)

    if norm:
        return fitness_calc_norm(extractor, weights)
    return fitness_calc_raw(extractor, weights)


def fitness_calc_raw(
    extractor: Callable,
    weights: Tuple[float, ...],
) -> Callable:
    """Raw fitness calculator — no normalisation.

    Parameters
    ----------
    extractor:
        Function that extracts objective values from a model.
    weights:
        Fitness weights (negative = minimise, positive = maximise).

    Returns
    -------
    callable
        Fitness calculator function.
    """

    def calculator(models: List) -> None:
        for modl in models:
            vals = extractor(modl)
            wvals = tuple(w * v for w, v in zip(weights, vals))
            modl.fitness_values = vals
            modl.wvalues = wvals

    return calculator


def fitness_calc_norm(
    extractor: Callable,
    weights: Tuple[float, ...],
) -> Callable:
    """Normalised fitness calculator — L2 normalises each objective.

    Parameters
    ----------
    extractor:
        Function that extracts objective values from a model.
    weights:
        Fitness weights.

    Returns
    -------
    callable
        Fitness calculator function.
    """
    import numpy as np

    def calculator(models: List) -> None:
        vals = []
        for modl in models:
            vs = extractor(modl)
            vals.append(vs)

        npvals = np.array(vals).T
        normed = []
        for col in npvals:
            norm = col / np.linalg.norm(col)
            normed.append(norm)

        normd_vals = np.array(normed).T

        for i, modl in enumerate(models):
            raw = tuple(normd_vals[i])
            wvals = tuple(w * v for w, v in zip(weights, raw))
            modl.fitness_values = raw
            modl.wvalues = wvals

    return calculator


def build_fitness_weights(params: List[str]) -> Tuple[float, ...]:
    """Parse fitness parameters to extract weights.

    Parameters
    ----------
    params:
        List of fitness parameter strings.

    Returns
    -------
    tuple[float, ...]
        Weights for each objective.
    """
    weights = []
    for p in params:
        W = 1.0
        lp = p.index("(") if "(" in p else -1
        rp = p.index(")") if ")" in p else -1
        if lp >= 0 and rp >= 0:
            W = float(p[lp + 1 : rp])
        if p[0] == "-":
            weights.append(-1.0 * W)
        elif p[0] == "+":
            weights.append(1.0 * W)
        else:
            weights.append(-1.0)  # default: minimise
    return tuple(weights)


def build_value_extractor(params: List[str]) -> Callable:
    """Create a function that extracts specified attributes from models.

    Parameters
    ----------
    params:
        List of fitness parameter strings (after weight parsing).

    Returns
    -------
    callable
        Extractor function that takes a model and returns a tuple of values.
    """
    ps = []
    for i, p in enumerate(params):
        if ")" in p:
            rp = p.index(")") + 1
            ps.append(p[rp:])
        else:
            ps.append(p[1:])

    def extractor(modl) -> Tuple:
        vals = []
        for attr in ps:
            v = getattr(modl, attr)
            vals.append(v)
        return tuple(vals)

    return extractor
