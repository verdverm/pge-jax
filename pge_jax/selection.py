"""Multi-objective selection operators for the PGE search loop.

Ported from DEAP.  Provides NSGA-II, SPEA-II, and tournament selection
with dominance + crowding distance.  Fitness values are stored directly
as tuples on model objects via ``values`` (selNSGA2) and ``wvalues``
(sortLogNondominated) attributes.
"""

from __future__ import annotations

import bisect
import math
import random
from collections import defaultdict
from itertools import chain
from operator import attrgetter, itemgetter
from typing import Any, List, Sequence, Tuple

random.seed(23)


# ---------------------------------------------------------------------------
# NSGA-II
# ---------------------------------------------------------------------------


def selNSGA2(
    individuals: List[Any],
    k: int,
    nd: str = "standard",
) -> List[Any]:
    """Apply NSGA-II selection operator.

    Parameters
    ----------
    individuals:
        List of individuals to select from.
    k:
        Number of individuals to select.
    nd:
        Non-dominated sorting method: ``"standard"`` or ``"log"``.

    Returns
    -------
    list
        Selected individuals (references to input objects).
    """
    if not individuals:
        return []

    if nd == "standard":
        pareto_fronts = sortNondominated(individuals, k)
    elif nd == "log":
        pareto_fronts = sortLogNondominated(individuals, k)
    else:
        raise ValueError(f'selNSGA2: invalid nd method "{nd}"')

    for front in pareto_fronts:
        assignCrowdingDist(front)

    chosen = list(chain(*pareto_fronts[:-1]))
    k = k - len(chosen)
    if k > 0:
        sorted_front = sorted(
            pareto_fronts[-1],
            key=attrgetter("crowding_dist"),
            reverse=True,
        )
        chosen.extend(sorted_front[:k])

    return chosen


def sortNondominated(
    individuals: List[Any],
    k: int,
    first_front_only: bool = False,
) -> List[List[Any]]:
    """Fast non-dominated sorting (Deb et al. 2002).

    Parameters
    ----------
    individuals:
        List of individuals.
    k:
        Number of individuals to sort.
    first_front_only:
        If ``True``, sort only the first front.

    Returns
    -------
    list[list]
        List of Pareto fronts (each front is a list of individuals).
    """
    if k == 0:
        return []

    map_fit_ind = defaultdict(list)
    for ind in individuals:
        map_fit_ind[ind.values].append(ind)
    fits = list(map_fit_ind.keys())

    current_front = []
    next_front = []
    dominating_fits: dict[Any, int] = defaultdict(int)
    dominated_fits: dict[Any, list[Any]] = defaultdict(list)

    for i, fit_i in enumerate(fits):
        for fit_j in fits[i + 1 :]:
            if fit_i.dominates(fit_j):
                dominating_fits[fit_j] += 1
                dominated_fits[fit_i].append(fit_j)
            elif fit_j.dominates(fit_i):
                dominating_fits[fit_i] += 1
                dominated_fits[fit_j].append(fit_i)
        if dominating_fits[fit_i] == 0:
            current_front.append(fit_i)

    fronts: list[list[Any]] = [[]]
    for fit in current_front:
        fronts[-1].extend(map_fit_ind[fit])
    pareto_sorted = len(fronts[-1])

    if not first_front_only:
        N = min(len(individuals), k)
        while pareto_sorted < N:
            fronts.append([])
            for fit_p in current_front:
                for fit_d in dominated_fits[fit_p]:
                    dominating_fits[fit_d] -= 1
                    if dominating_fits[fit_d] == 0:
                        next_front.append(fit_d)
                        pareto_sorted += len(map_fit_ind[fit_d])
                        fronts[-1].extend(map_fit_ind[fit_d])
            current_front = next_front
            next_front = []

    return fronts


def assignCrowdingDist(individuals: List[Any]) -> None:
    """Assign crowding distance to each individual.

    Stores the result on ``individual[i].crowding_dist``.
    """
    if len(individuals) == 0:
        return

    distances = [0.0] * len(individuals)
    crowd = [(ind.values, i) for i, ind in enumerate(individuals)]

    nobj = len(individuals[0].values)

    for i in range(nobj):
        crowd.sort(key=lambda element: element[0][i])
        distances[crowd[0][1]] = float("inf")
        distances[crowd[-1][1]] = float("inf")
        if crowd[-1][0][i] == crowd[0][0][i]:
            continue
        norm = nobj * float(crowd[-1][0][i] - crowd[0][0][i])
        for prev, cur, nxt in zip(crowd[:-2], crowd[1:-1], crowd[2:]):
            distances[cur[1]] += (nxt[0][i] - prev[0][i]) / norm

    for i, dist in enumerate(distances):
        individuals[i].crowding_dist = dist


def selTournamentDCD(individuals: List[Any], k: int) -> List[Any]:
    """Tournament selection based on dominance and crowding distance.

    Parameters
    ----------
    individuals:
        List of individuals (must have ``crowding_dist`` attribute).
    k:
        Number of individuals to select.

    Returns
    -------
    list
        Selected individuals.
    """

    def tourn(ind1: Any, ind2: Any) -> Any:
        if ind1.values.dominates(ind2.values):
            return ind1
        if ind2.values.dominates(ind1.values):
            return ind2

        if getattr(ind1, "crowding_dist", 0) < getattr(ind2, "crowding_dist", 0):
            return ind2
        if getattr(ind1, "crowding_dist", 0) > getattr(ind2, "crowding_dist", 0):
            return ind1

        return ind1 if random.random() <= 0.5 else ind2

    individuals_1 = random.sample(individuals, len(individuals))
    individuals_2 = random.sample(individuals, len(individuals))

    chosen = []
    for i in range(0, k, 4):
        chosen.append(tourn(individuals_1[i], individuals_1[i + 1]))
        chosen.append(tourn(individuals_1[i + 2], individuals_1[i + 3]))
        chosen.append(tourn(individuals_2[i], individuals_2[i + 1]))
        chosen.append(tourn(individuals_2[i + 2], individuals_2[i + 3]))

    return chosen


# ---------------------------------------------------------------------------
# Generalized Reduced Run-Time Complexity Non-Dominated Sorting
# ---------------------------------------------------------------------------


def identity(obj: Any) -> Any:
    """Return *obj* directly."""
    return obj


def isDominated(wvalues1: Tuple[float, ...], wvalues2: Tuple[float, ...]) -> bool:
    """Check if *wvalues1* is dominated by *wvalues2*.

    Minimization semantics: lower is better.

    Parameters
    ----------
    wvalues1:
        Fitness values that would be dominated.
    wvalues2:
        Dominant fitness values.

    Returns
    -------
    bool
        ``True`` if wvalues2 dominates wvalues1.
    """
    not_equal = False
    for self_w, other_w in zip(wvalues1, wvalues2):
        if self_w > other_w:
            return False
        if self_w < other_w:
            not_equal = True
    return not_equal


def median(seq: Sequence[Any], key=identity) -> Any:
    """Return the median value of *seq*."""
    sseq = sorted(seq, key=key)
    length = len(seq)
    if length % 2 == 1:
        return key(sseq[(length - 1) // 2])
    return (key(sseq[(length - 1) // 2]) + key(sseq[length // 2])) / 2.0


def sortLogNondominated(
    individuals: List[Any],
    k: int,
    first_front_only: bool = False,
) -> List[List[Any]]:
    """Generalized Reduced Run-Time Complexity Non-Dominated Sorting.

    Uses the algorithm of Fortin et al. (2013).

    Parameters
    ----------
    individuals:
        List of individuals (must have ``wvalues`` attribute).
    k:
        Number of individuals to sort.
    first_front_only:
        If ``True``, return only the first front.

    Returns
    -------
    list[list]
        List of Pareto fronts.
    """
    if k == 0:
        return []

    if not individuals:
        return []

    unique_fits = defaultdict(list)
    for i, ind in enumerate(individuals):
        unique_fits[ind.wvalues].append(ind)

    obj = len(individuals[0].wvalues) - 1
    fitnesses = list(unique_fits.keys())
    front = dict.fromkeys(fitnesses, 0)

    fitnesses.sort(reverse=True)
    sortNDHelperA(fitnesses, obj, front)

    nbfronts = max(front.values()) + 1
    pareto_fronts: list[list[Any]] = [[] for _ in range(nbfronts)]
    for fit in fitnesses:
        index = front[fit]
        pareto_fronts[index].extend(unique_fits[fit])

    if not first_front_only:
        count = 0
        for i, front_list in enumerate(pareto_fronts):
            count += len(front_list)
            if count >= k:
                return pareto_fronts[: i + 1]
        return pareto_fronts
    return pareto_fronts[0]


def sortNDHelperA(fitnesses: List, obj: int, front: dict) -> None:
    """Non-dominated sort on the first M objectives."""
    if len(fitnesses) < 2:
        return
    if len(fitnesses) == 2:
        s1, s2 = fitnesses[0], fitnesses[1]
        if isDominated(s2[: obj + 1], s1[: obj + 1]):
            front[s2] = max(front[s2], front[s1] + 1)
    elif obj == 1:
        sweepA(fitnesses, front)
    elif len(frozenset(map(itemgetter(obj), fitnesses))) == 1:
        sortNDHelperA(fitnesses, obj - 1, front)
    else:
        best, worst = splitA(fitnesses, obj)
        sortNDHelperA(best, obj, front)
        sortNDHelperB(best, worst, obj - 1, front)
        sortNDHelperA(worst, obj, front)


def splitA(fitnesses: List, obj: int) -> Tuple[List, List]:
    """Partition fitnesses according to the median of objective *obj*."""
    median_ = median(fitnesses, itemgetter(obj))
    best_a, worst_a, best_b, worst_b = [], [], [], []

    for fit in fitnesses:
        if fit[obj] > median_:
            best_a.append(fit)
            best_b.append(fit)
        elif fit[obj] < median_:
            worst_a.append(fit)
            worst_b.append(fit)
        else:
            best_a.append(fit)
            worst_b.append(fit)

    balance_a = abs(len(best_a) - len(worst_a))
    balance_b = abs(len(best_b) - len(worst_b))

    if balance_a <= balance_b:
        return best_a, worst_a
    return best_b, worst_b


def sweepA(fitnesses: List, front: dict) -> None:
    """Update front numbers using a geometric sweep on the first 2 objectives."""
    stairs = [-fitnesses[0][1]]
    fstairs = [fitnesses[0]]
    for fit in fitnesses[1:]:
        idx = bisect.bisect_right(stairs, -fit[1])
        if 0 < idx <= len(stairs):
            fstair = max(fstairs[:idx], key=front.__getitem__)
            front[fit] = max(front[fit], front[fstair] + 1)
        for i, fstair in enumerate(fstairs[idx:], idx):
            if front[fstair] == front[fit]:
                del stairs[i]
                del fstairs[i]
                break
        stairs.insert(idx, -fit[1])
        fstairs.insert(idx, fit)


def sortNDHelperB(best: List, worst: List, obj: int, front: dict) -> None:
    """Assign front numbers to solutions in *worst* based on *best*."""
    key = itemgetter(obj)
    if len(worst) == 0 or len(best) == 0:
        return
    if len(best) == 1 or len(worst) == 1:
        for hi in worst:
            for li in best:
                if isDominated(hi[: obj + 1], li[: obj + 1]) or hi[: obj + 1] == li[: obj + 1]:
                    front[hi] = max(front[hi], front[li] + 1)
    elif obj == 1:
        sweepB(best, worst, front)
    elif key(min(best, key=key)) >= key(max(worst, key=key)):
        sortNDHelperB(best, worst, obj - 1, front)
    elif key(max(best, key=key)) >= key(min(worst, key=key)):
        best1, best2, worst1, worst2 = splitB(best, worst, obj)
        sortNDHelperB(best1, worst1, obj, front)
        sortNDHelperB(best1, worst2, obj - 1, front)
        sortNDHelperB(best2, worst2, obj, front)


def splitB(best: List, worst: List, obj: int) -> Tuple[List, List, List, List]:
    """Split both best and worst sets according to the median."""
    median_ = median(best if len(best) > len(worst) else worst, itemgetter(obj))
    best1_a, best2_a, best1_b, best2_b = [], [], [], []
    for fit in best:
        if fit[obj] > median_:
            best1_a.append(fit)
            best1_b.append(fit)
        elif fit[obj] < median_:
            best2_a.append(fit)
            best2_b.append(fit)
        else:
            best1_a.append(fit)
            best2_b.append(fit)

    worst1_a, worst2_a, worst1_b, worst2_b = [], [], [], []
    for fit in worst:
        if fit[obj] > median_:
            worst1_a.append(fit)
            worst1_b.append(fit)
        elif fit[obj] < median_:
            worst2_a.append(fit)
            worst2_b.append(fit)
        else:
            worst1_a.append(fit)
            worst2_b.append(fit)

    balance_a = abs(len(best1_a) - len(best2_a) + len(worst1_a) - len(worst2_a))
    balance_b = abs(len(best1_b) - len(best2_b) + len(worst1_b) - len(worst2_b))

    if balance_a <= balance_b:
        return best1_a, best2_a, worst1_a, worst2_a
    return best1_b, best2_b, worst1_b, worst2_b


def sweepB(best: List, worst: List, front: dict) -> None:
    """Adjust rank of worst fitnesses based on best on first 2 objectives."""
    stairs: list[float] = []
    fstairs: list[Any] = []
    iter_best = iter(best)
    next_best: Any = next(iter_best, None)
    for h in worst:
        while next_best is not None and h[:2] <= next_best[:2]:
            insert = True
            for i, fstair in enumerate(fstairs):
                if front[fstair] == front[next_best]:
                    if fstair[1] > next_best[1]:
                        insert = False
                    else:
                        del stairs[i], fstairs[i]
                    break
            if insert:
                idx = bisect.bisect_right(stairs, -next_best[1])
                stairs.insert(idx, -next_best[1])
                fstairs.insert(idx, next_best)
            next_best = next(iter_best, None)

        idx = bisect.bisect_right(stairs, -h[1])
        if 0 < idx <= len(stairs):
            fstair = max(fstairs[:idx], key=front.__getitem__)
            front[h] = max(front[h], front[fstair] + 1)


# ---------------------------------------------------------------------------
# SPEA-II
# ---------------------------------------------------------------------------


def selSPEA2(individuals: List[Any], k: int) -> List[Any]:
    """Apply SPEA-II selection operator.

    Parameters
    ----------
    individuals:
        List of individuals.
    k:
        Number of individuals to select.

    Returns
    -------
    list
        Selected individuals.
    """
    N = len(individuals)
    L = len(individuals[0].values)
    K = math.sqrt(N)
    strength_fits: list[float] = [0.0] * N
    fits: list[float] = [0.0] * N
    dominating_inds: list[list[int]] = [list() for _ in range(N)]

    for i, ind_i in enumerate(individuals):
        for j, ind_j in enumerate(individuals[i + 1 :], i + 1):
            if ind_i.values.dominates(ind_j.values):
                strength_fits[i] += 1
                dominating_inds[j].append(i)
            elif ind_j.values.dominates(ind_i.values):
                strength_fits[j] += 1
                dominating_inds[i].append(j)

    for i in range(N):
        for j in dominating_inds[i]:
            fits[i] += strength_fits[j]

    chosen_indices = [i for i in range(N) if fits[i] < 1]

    if len(chosen_indices) < k:
        for i in range(N):
            dists = [0.0] * N
            for j in range(i + 1, N):
                dist = sum((individuals[i].values[l] - individuals[j].values[l]) ** 2 for l in range(L))
                dists[j] = dist
            kth_dist = _randomizedSelect(dists, 0, N - 1, K)
            density = 1.0 / (kth_dist + 2.0)
            fits[i] += density

        next_indices = [(fits[i], i) for i in range(N) if i not in chosen_indices]
        next_indices.sort()
        chosen_indices += [i for _, i in next_indices[: k - len(chosen_indices)]]

    elif len(chosen_indices) > k:
        N_chosen = len(chosen_indices)
        distances: list[list[float]] = [[0.0] * N_chosen for _ in range(N_chosen)]
        sorted_indices: list[list[int]] = [[0] * N_chosen for _ in range(N_chosen)]
        for i in range(N_chosen):
            for j in range(i + 1, N_chosen):
                dist = sum(
                    (individuals[chosen_indices[i]].values[l] - individuals[chosen_indices[j]].values[l]) ** 2
                    for l in range(L)
                )
                distances[i][j] = dist
                distances[j][i] = dist
            distances[i][i] = -1

        for i in range(N_chosen):
            for j in range(1, N_chosen):
                l = j
                while l > 0 and distances[i][j] < distances[i][sorted_indices[i][l - 1]]:
                    sorted_indices[i][l] = sorted_indices[i][l - 1]
                    l -= 1
                sorted_indices[i][l] = j

        size = N_chosen
        to_remove = []
        while size > k:
            min_pos = 0
            for i in range(1, N_chosen):
                for j in range(1, size):
                    dist_i = distances[i][sorted_indices[i][j]]
                    dist_min = distances[min_pos][sorted_indices[min_pos][j]]
                    if dist_i < dist_min:
                        min_pos = i
                        break
                    if dist_i > dist_min:
                        break

            for i in range(N_chosen):
                distances[i][min_pos] = float("inf")
                distances[min_pos][i] = float("inf")
                for j in range(1, size - 1):
                    if sorted_indices[i][j] == min_pos:
                        sorted_indices[i][j] = sorted_indices[i][j + 1]
                        sorted_indices[i][j + 1] = min_pos

            to_remove.append(min_pos)
            size -= 1

        for index in reversed(sorted(to_remove)):
            del chosen_indices[index]

    return [individuals[i] for i in chosen_indices]


def _randomizedSelect(array: List[float], begin: int, end: int, i: float) -> float:
    """Select the ith smallest element without full sorting."""
    if begin == end:
        return array[begin]
    q = _randomizedPartition(array, begin, end)
    k = q - begin + 1
    if i < k:
        return _randomizedSelect(array, begin, q, i)
    return _randomizedSelect(array, q + 1, end, i - k)


def _randomizedPartition(array: List[float], begin: int, end: int) -> int:
    """Partition array around a median-of-three pivot."""
    m = begin + (end - begin) // 2
    b, e = begin, end
    if end - begin > 40:
        s = (end - begin) // 8
        b = _medianIndexThree(array, begin, begin + s, begin + 2 * s)
        m = _medianIndexThree(array, m, m - s, m + s)
        e = _medianIndexThree(array, end - 1, end - 1 - s, end - 1 - 2 * s)
    m = _medianIndexThree(array, b, m, e - 1)

    array[begin], array[m] = array[m], array[begin]
    return _partition(array, begin, end)


def _partition(array: List[float], begin: int, end: int) -> int:
    """Lomuto partition scheme."""
    x = array[begin]
    i = begin - 1
    j = end + 1
    while True:
        j -= 1
        while array[j] > x:
            j -= 1
        i += 1
        while array[i] < x:
            i += 1
        if i < j:
            array[i], array[j] = array[j], array[i]
        else:
            return j


def _medianIndexThree(array: List[float], i1: int, i2: int, i3: int) -> int:
    """Return the index of the median of three elements."""
    c = _cmp(array[i1], array[i2])
    if c < 0:
        c2 = _cmp(array[i1], array[i3])
        return i1 if c2 < 0 else i2
    c2 = _cmp(array[i2], array[i3])
    return i2 if c2 < 0 else i3


def _cmp(a: float, b: float) -> int:
    """Compare two floats, returning -1, 0, or 1."""
    return (a > b) - (a < b)


__all__ = [
    "selNSGA2",
    "selSPEA2",
    "sortNondominated",
    "sortLogNondominated",
    "selTournamentDCD",
    "isDominated",
    "assignCrowdingDist",
]
