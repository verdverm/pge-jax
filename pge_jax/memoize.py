"""Expression deduplication for the PGE search loop.

Uses a simple hash-based dictionary to track which expressions have
already been seen, preventing duplicate evaluations.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import sympy

from pge_jax.search_model import SearchModel


class Memoizer:
    """Index models by expression hash for deduplication.

    Parameters
    ----------
    variables:
        Input variable symbols used to identify coefficient symbols.
    """

    def __init__(self, variables):
        self.models: list[SearchModel] = []
        self.variables = variables
        self.hmap: Dict[int, SearchModel] = {}

    def insert(self, model: SearchModel) -> bool:
        """Insert a model into the index.

        Parameters
        ----------
        model:
            The model to insert.

        Returns
        -------
        bool
            ``True`` if the expression was new (inserted),
            ``False`` if it was a duplicate.
        """
        h = model.orig.__hash__()
        r = self.hmap.get(h, None)
        if r is not None:
            return False

        model.id = len(self.models)
        self.models.append(model)
        self.hmap[h] = model
        return True

    def lookup(self, model: SearchModel) -> Tuple[bool, Optional[SearchModel]]:
        """Check if an expression is already indexed.

        Parameters
        ----------
        model:
            The model to look up.

        Returns
        -------
        tuple[bool, SearchModel | None]
            ``(True, model)`` if found, ``(False, None)`` otherwise.
        """
        h = model.orig.__hash__()
        r = self.hmap.get(h, None)
        if r is None:
            return False, None
        return True, r

    def get_by_id(self, i: int) -> SearchModel:
        """Return the model with index *i*.

        Parameters
        ----------
        i:
            Model index.

        Returns
        -------
        SearchModel
        """
        return self.models[i]
